import SimpleITK as sitk
import numpy as np

class Demons:
    
    def calc(self, fixed, moving, moving_seg=None):
        warped = np.zeros(fixed.shape)
        jac = np.zeros(fixed.shape)
        warped_seg = np.zeros(fixed.shape)
        if len(fixed.shape) == 4: # batch
            for b in range(fixed.shape[0]):
                for t in range(fixed.shape[1]):
                    if moving_seg is None:
                        w, j, ws = self._calc_single(fixed[b,t,...], 
                                                     moving[b,...],
                                                     None)
                    else:
                        w, j, ws = self._calc_single(fixed[b,t,...],
                                                     moving[b,...],
                                                     moving_seg[b,...])
                    warped[b,t,...] = w
                    jac[b,t, ...] = j.copy()
                    warped_seg[b,t,...] = ws
        elif len(fixed.shape) == 3: # sequence
            for t in range(fixed.shape[0]):
                w, j, ws = self._calc_single(fixed[t,...], moving, moving_seg)
                warped[t,...] = w
                jac[t, ...] = j.copy()
                warped_seg[t,...] = ws
        else:
            assert len(fixed.shape) == 2 # single image
            warped, jac, warped_seg = self._calc_single(fixed, moving, moving_seg)
        return warped, jac.copy(), warped_seg

class DemonsITK(Demons):    
    def __init__(self, iterations, calc_jac):
        self.iterations = iterations
        self.calc_jac = calc_jac
      
    def _calc_single(self, fixed, moving, moving_seg):
        moving_sitk = sitk.GetImageFromArray(moving*256)
        fixed_sitk = sitk.GetImageFromArray(fixed*256)
        resampler, disp, warped_sitk = self._calc(fixed_sitk, moving_sitk)
        warped = sitk.GetArrayViewFromImage(warped_sitk) / 256
        
        if self.calc_jac:
            jac_sitk = sitk.DisplacementFieldJacobianDeterminant(disp.GetDisplacementField())
            jac = sitk.GetArrayViewFromImage(jac_sitk)
        else:
            jac = np.asarray([0])
        if moving_seg is not None:
            moving_seg_sitk = sitk.GetImageFromArray(moving_seg*256)
            warped_seg_sitk = resampler.Execute(moving_seg_sitk)
            warped_seg = sitk.GetArrayViewFromImage(warped_seg_sitk) / 256
        else:
            warped_seg = None
        return warped, jac, warped_seg    
    
    def _calc(self, fixed, moving):
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOn()
        moving = matcher.Execute(moving, fixed)

        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(self.iterations)
        demons.SetStandardDeviations(1.0)

        displacementField = demons.Execute(fixed, moving)
        outTx = sitk.DisplacementFieldTransform(displacementField)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(outTx)

        warped = resampler.Execute(moving)
        return resampler, outTx, warped


import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

path = '/home/ngune07417/project/airlab'
sys.path.insert(0, path)
import airlab as al
import torch as th
from metrics import jacobian_determinant

class DemonsAL(Demons):
    def __init__(self, iterations, calc_jac, device='cpu', diffeomorphic=False):
        self.iterations = iterations
        self.calc_jac= calc_jac
        self.dtype = th.float32
        self.device = th.device(device) 
        self.diffeomorphic = diffeomorphic
    
    def _calc_single(self, fixed, moving, moving_seg):
        fixed_al = al.image_from_numpy(fixed.numpy(), [1, 1], [0, 0], dtype=self.dtype, device=self.device)
        moving_al = al.image_from_numpy(moving.numpy(), [1, 1], [0, 0], dtype=self.dtype, device=self.device)
        displacement, warped_al = self._calc(fixed_al, moving_al)
        warped = warped_al.numpy()
        
        if self.calc_jac:
            jac = jacobian_determinant(displacement.numpy())
        else:
            jac = np.asarray([0])
        if moving_seg is not None:
            moving_seg = al.image_from_numpy(moving_seg.numpy(), [1, 1], [0, 0], dtype=self.dtype, device=self.device)
            warped_seg = al.transformation.utils.warp_image(moving_seg, displacement).numpy()
        else:
            warped_seg = None
        return warped, jac, warped_seg 
    
    def _calc(self, fixed, moving):
        registration = al.DemonsRegistraion(verbose=False)
        transformation = al.transformation.pairwise.NonParametricTransformation(moving.size,
                                                                            dtype=self.dtype,
                                                                            device=self.device,
                                                                            diffeomorphic=self.diffeomorphic)
        registration.set_transformation(transformation)
        
        image_loss = al.loss.pairwise.MSE(fixed, moving)
        registration.set_image_loss([image_loss])

        # choose a regulariser for the demons
        regulariser = al.regulariser.demons.GaussianRegulariser(moving.spacing, sigma=[2, 2], dtype=self.dtype,
                                                                device=self.device)

        registration.set_regulariser([regulariser])

        # choose the Adam optimizer to minimize the objective
        optimizer = th.optim.Adam(transformation.parameters(), lr=0.01)

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(self.iterations)

        # start the registration
        registration.start()
        displacement = transformation.get_displacement()
        warped_image = al.transformation.utils.warp_image(moving, displacement)
        return displacement, warped_image
        