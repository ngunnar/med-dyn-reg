import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorlayer.cost import dice_coe, dice_hard_coe

import pystrum.pynd.ndutils as nd


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def ncc(self, Ii, Ji):
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(Ii.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims
        elif not isinstance(self.win, list):  # user specified a single number not a list
            self.win = [self.win] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # compute filters
        in_ch = list(Ji.shape)[-1]
        sum_filt = tf.ones([*self.win, in_ch, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)

        # compute local sums via convolution
        padding = 'SAME'
        I_sum = conv_fn(Ii, sum_filt, strides, padding)
        J_sum = conv_fn(Ji, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # TODO: simplify this
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = tf.maximum(cross, self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = tf.maximum(I_var, self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = tf.maximum(J_var, self.eps)

        # cc = (cross * cross) / (I_var * J_var)
        cc = (cross / I_var) * (cross / J_var)

        # return mean cc for each entry in batch
        return tf.reduce_mean(K.batch_flatten(cc), axis=-1)

    def sim(self, y_true, y_pred):
        y_pred = y_pred[...,None]
        y_true = y_true[...,None]
        result = [self.ncc(y_true[:,t,...], y_pred[:,t,...]) for t in range(y_true.shape[1])]
        return tf.stack(result, axis=1)

    
class MSE:
    """
    Sigma-weighted mean squared error for image reconstruction.
    """

    #def __init__(self, image_sigma=1.0):
    #    self.image_sigma = image_sigma

    def loss(self, y_true, y_pred):
        # Shape (bs, steps, *dim)
        #y_pred = y_pred[...,None]
        #y_true = y_true[...,None]
        #y_true = tf.repeat(y_true[:,None,...,None], y_pred.shape[1], axis=1)
        return tf.reduce_mean((y_true - y_pred)**2, axis=[2,3])
    

class DICE:
    
    def coeff(self, seg, gt):
        assert len(seg.shape) == 2, seg
        assert len(gt.shape) == 2, gt
        seg = tf.cast(seg[None,...,None]==1.0, 'float32')
        gt = gt[None,...,None]
        return dice_hard_coe(seg, gt)
    

def batch_jacobian_determinant(disp):
    all_result = []
    for b in range(disp.shape[0]):
        batch_result = []
        for t in range(disp.shape[1]):
            batch_result.append(jacobian_determinant(disp[b,t,...]))
        all_result.append(tf.stack(batch_result))
    return tf.stack(all_result)

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]