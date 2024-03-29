import os
import glob
import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from .data_utils import get_tf_data, preprocess_data

def gaussian_noise_percentage(image, percent):
    # how to: https://stackoverflow.com/questions/31834803/how-to-add-5-percent-gaussian-noise-to-image

    # stats
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    minx = stats.GetMinimum()
    maxx = stats.GetMaximum()
    meanx = stats.GetMean()
    stdx = stats.GetSigma()

    # normalize image
    img = sitk.ShiftScale( sitk.Cast( image, sitk.sitkFloat64 ), -minx, 1/(maxx - minx) )
    
    # stats to get std
    stats = sitk.StatisticsImageFilter()
    stats.Execute(img)
    sigma = stats.GetSigma()
    
    # new_sigma = sigma * percent
    new_sigma = sigma * percent * (maxx - minx) + minx

    # gaussian noise
    output = sitk.AdditiveGaussianNoise(image, new_sigma)
    return output 

def slice_image(image, slice_value, view):
    if view == 'axial':
        return image[:,:,slice_value]
    if view == 'coronal':
        return image[:, slice_value, ::-1]
    if view == 'sagittal':
        return image[slice_value,:,::-1]
    raise NotImplemented('View not implemented')


class SimLoader3D:
    @staticmethod
    def get_files(folder_output):
        folder_out3d = os.path.join(folder_output, 'image/')
        files = glob.glob(folder_out3d + '/*')
        files.sort(key = lambda x: int(x.split('/')[-1].split('.')[0][-4:]))
        return files
    
    @staticmethod
    def read_generator(directories, get_files, noise_percentage, slice_no, view, dim_y, max_val, min_val):
        def gen():
            for directory in directories:
                files = get_files(directory)
                video = [] 
                for f in files:
                    image = sitk.ReadImage(f)
                    image = slice_image(image, slice_no, view)
                    image = gaussian_noise_percentage(image, noise_percentage)
                    image = sitk.GetArrayFromImage(image)
                    if len(image.shape) == 3:
                        image = image[0,...]
                    video.append(image)
                video = np.asarray(video, dtype='float32')
                video, img_ref = preprocess_data(video, dim_y, max_val, min_val)                        
                yield {'input_ref': img_ref, 'input_video': video}
        return gen 
    
    

class SimLoader:
    @staticmethod
    def get_files(folder_output, view):
        folder = os.path.join(folder_output, view)
        files = glob.glob(folder + '/*')
        files.sort(key = lambda x: int(x.split('/')[-1].split('.')[0][-4:]))
        return files
    
    @staticmethod
    def read_generator(directories, get_files, dim_y, max_val, min_val ):
        def gen():
            for directory in directories:
                files = get_files(directory)
                video = [] 
                for f in files:
                    image = np.load(f)                                        
                    video.append(image)
                video = np.asarray(video, dtype='float32')
                video, img_ref = preprocess_data(video, dim_y, max_val, min_val)
                yield {'input_ref': img_ref, 'input_video': video}
        return gen 

class SimTestLoader:

    @staticmethod
    def get_files(folder_output, view, seg):
        img_folder = os.path.join(folder_output, view)
        seg_folder = os.path.join(folder_output, 'seg', seg)
        img_files = glob.glob(img_folder + '/*')
        img_files.sort(key = lambda x: int(x.split('/')[-1].split('.')[0][-4:]))
        seg_files = glob.glob(seg_folder + '/*')
        seg_files.sort(key = lambda x: int(x.split('/')[-1].split('.')[0][-4:]))
        return img_files, seg_files
    
    @staticmethod
    def read_generator(directories, view, seg_name, dim_y, max_val, min_val, noise_percentage):
        def gen():
            for directory in directories:
                img_files, seg_files = SimTestLoader.get_files(directory, view, seg_name)
                video = []
                segs = []
                for img_f, seg_f in zip(img_files, seg_files):
                    image = sitk.ReadImage(img_f)
                    seg = sitk.ReadImage(seg_f)
                    # Add noise
                    image = gaussian_noise_percentage(image, noise_percentage)
                    
                    image = sitk.GetArrayFromImage(image)
                    seg = sitk.GetArrayFromImage(seg)
                    video.append(image)                    
                    segs.append(seg)
                
                video = np.asarray(video, dtype='float32')
                segs = np.asarray(segs, dtype='float32')
                
                video, img_ref, segs, seg_refs = preprocess_data(video, dim_y, max_val, min_val, segs)
                yield {'input_ref': img_ref, 'input_video': video, 'input_seg_ref': seg_refs, 'input_seg': segs}
        return gen     
    
class SimDataLoader:  
    def __init__(self, length, dim_y, view):
        self.directories = glob.glob('/data/Niklas/CineMRI/train/*') 
        self.directories.sort()

        train, test = np.split(self.directories, [int(len(self.directories)*0.9)])
        self.train_directories = list(train)
        self.test_directories = list(test)

        noise_percentage = 0.1
        slice_no = 218
        max_val = 300
        min_val = 100
         
        get_files = lambda x: SimLoader.get_files(x, view)
        self.ds_train, self.ds_model_train = get_tf_data(self.train_directories, length, dim_y, lambda x: SimLoader.read_generator(x, get_files, dim_y, max_val, min_val))
        self.ds_test, self.ds_model_test = get_tf_data(self.test_directories, length, dim_y, lambda x: SimLoader.read_generator(x, get_files, dim_y, max_val, min_val))
        
class SimTestDataLoader:  
    def __init__(self, dim_y, view):
        self.directories = glob.glob('/data/Niklas/CineMRI/test/3')
        self.directories.sort()

        train, test = np.split(self.directories, [int(len(self.directories)*0.9)])
        self.train_directories = list(train)
        self.test_directories = list(test)

        noise_percentage = 0.1
        slice_no = 218
        max_val = 300
        min_val = 100
         
        #get_files = lambda x: SimLoader.get_files(x, view),
        self.ds = tf.data.Dataset.from_generator(SimTestLoader.read_generator(self.directories, 'sagittal', 'tumor', dim_y, max_val, min_val, noise_percentage), 
                                                 output_types=({'input_ref': tf.float32, 
                                                                'input_video': tf.float32, 
                                                                'input_seg_ref': tf.float32,
                                                                'input_seg': tf.float32}),
                                                 output_shapes= ({'input_ref': tf.TensorShape([None, None]), 
                                                                  'input_video': tf.TensorShape([None, None, None]),
                                                                  'input_seg_ref': tf.TensorShape([None, None]), 
                                                                  'input_seg': tf.TensorShape([None, None, None]),}))