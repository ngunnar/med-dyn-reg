import tensorflow as tf
import numpy as np
import glob
import SimpleITK as sitk
from .numpyDataset import NumpyLoader

class MhaLoader(NumpyLoader):
    
    @staticmethod
    def get_files(directory):
        if 'Volunteer' in directory:
            files = glob.glob(directory + "/*_cut0000_*.mha")
            files.sort(key = lambda x: int(x.split('/')[-1].split('_')[0]))
            return files
        if 'comodo' in directory:
            files = glob.glob(directory + "/*.mha")
            files.sort(key = lambda x: int(x.split('/')[-1].split('.')[0][3:]))
            return files
    
    @staticmethod
    def read_generator(directories):
        def gen():
            for directory in directories:
                files = MhaLoader.get_files(directory)
                images = [] 
                for f in files:
                    org_image = sitk.ReadImage(f)
                    org_image = sitk.GetArrayFromImage(org_image)
                    if len(org_image.shape) == 3:
                        org_image = org_image[0,...]
                    images.append(org_image)
                images = np.asarray(images, dtype='float32')[1:]                                
                yield images
        return gen
    
    @staticmethod
    def process_generator(dataset, dim_y, max_val, min_val=500):
        def gen():
            for images in dataset:
                # remove outliers
                images = MhaLoader.crop(images, min_val)
                images = tf.clip_by_value(images, 0, max_val)
                # resize
                images = tf.image.resize(images[...,None], dim_y)
                # normalize
                images = tf.image.per_image_standardization(images)
                images = images[...,0]
                f, _, _ = images.shape
                mask = np.zeros(f, dtype='bool')
                yield {'input_video': images, 'input_mask': mask}
        return gen
    
    @staticmethod
    def get_tf_data(data_list, ph_steps, dim_y, max_val, min_val=500):
        window = ph_steps
        shift = window
        # Read files
        ds = tf.data.Dataset.from_generator(MhaLoader.read_generator(data_list), 
                                                 tf.float32, tf.TensorShape([None, None, None]))
        
        # Process dataset
        ds_model = ds.flat_map(lambda x: MhaLoader.get_data(x, window, shift=shift, drop_remainder=True))
        ds_model = tf.data.Dataset.from_generator(MhaLoader.process_generator(ds_model, dim_y, max_val, min_val=min_val),
                                             output_types = ({"input_video": tf.float32, 
                                                             "input_mask": tf.bool}),
                                             output_shapes = ({"input_video": (window, *dim_y),
                                                               "input_mask": (window)}))
        
        return ds, ds_model
        

class VolunteerDataLoader:
    def __init__(self, ph_steps, dim_y):
        
        test_patient = 'V1_2345123140_Abdo/Fraction1'
        excepted_patients = 'Pelvis'
        
        max_val = 3000
        directories = glob.glob('/data/Niklas/Unity/Volunteer/**/**/2DSlices')
        directories = [x for x in directories if excepted_patients not in x]
        
        self.train_directories = [x for x in directories if test_patient not in x]
        self.test_directories = [x for x in directories if test_patient in x]        
               
        self.sag_train_data, self.sag_train = MhaLoader.get_tf_data(self.train_directories, ph_steps, dim_y, max_val)
        self.sag_test_data, self.sag_test = MhaLoader.get_tf_data(self.test_directories, ph_steps, dim_y, max_val)
        
class ComodoDataLoader:
    def __init__(self, ph_steps, dim_y):
        
        test_patient = 'comodo/comodo/comodo_pp/01b'        
        
        max_val = 1500
        directories = glob.glob('/data/Niklas/Unity/comodo/comodo/comodo_pp/**/bffe_cine_sagittal')
        
        self.train_directories = [x for x in directories if test_patient not in x]
        self.test_directories = [x for x in directories if test_patient in x]        
               
        self.sag_train_data, self.sag_train = MhaLoader.get_tf_data(self.train_directories, ph_steps, dim_y, max_val)
        self.sag_test_data, self.sag_test = MhaLoader.get_tf_data(self.test_directories, ph_steps, dim_y, max_val)

        
        
class MergedDataLoader:
    def __init__(self, ph_steps, dim_y):
        volunteer_ds = VolunteerDataLoader(ph_steps, dim_y)
        comodo_ds = ComodoDataLoader(ph_steps, dim_y)
        
        self.ds_train = volunteer_ds.sag_train.concatenate(comodo_ds.sag_train)
        self.ds_test = volunteer_ds.sag_test.concatenate(comodo_ds.sag_test)
        