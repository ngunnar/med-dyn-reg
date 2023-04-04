import glob
import SimpleITK as sitk
import numpy as np

from .data_utils import get_tf_data


class MhaLoader:
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
    def read_generator(directories, get_files):
        def gen():
            for directory in directories:
                files = get_files(directory)
                images = [] 
                for f in files:
                    image = sitk.ReadImage(f)
                    image = sitk.GetArrayFromImage(image)
                    if len(image.shape) == 3:
                        image = image[0,...]
                    images.append(image)
                images = np.asarray(images, dtype='float32')[1:]                                
                yield {'input_ref': images[0,...], 'input_video': images}
        return gen   

class VolunteerDataLoader:  
    def __init__(self, length, dim_y):
        
        test_patient = 'V1_2345123140_Abdo/Fraction1'
        excepted_patients = 'Pelvis'
        
        max_val = 3000
        directories = glob.glob('/data/Niklas/Unity/Volunteer/**/**/2DSlices')
        directories = [x for x in directories if excepted_patients not in x]
        
        self.train_directories = [x for x in directories if test_patient not in x]
        self.test_directories = [x for x in directories if test_patient in x]        
               
        self.sag_train_data, self.sag_train = get_tf_data(self.train_directories, length, dim_y, max_val, lambda x: MhaLoader.read_generator(x, MhaLoader.get_files))
        self.sag_test_data, self.sag_test = get_tf_data(self.test_directories, length, dim_y, max_val, lambda x: MhaLoader.read_generator(x, MhaLoader.get_files))
        
class ComodoDataLoader:
    def __init__(self, length, dim_y):
        
        test_patient = 'comodo/comodo/comodo_pp/01b'        
        
        max_val = 1500
        directories = glob.glob('/data/Niklas/Unity/comodo/comodo/comodo_pp/**/bffe_cine_sagittal')
        
        self.train_directories = [x for x in directories if test_patient not in x]
        self.test_directories = [x for x in directories if test_patient in x]        
               
        self.sag_train_data, self.sag_train = get_tf_data(self.train_directories, length, dim_y, max_val, lambda x: MhaLoader.read_generator(x, MhaLoader.get_files))
        self.sag_test_data, self.sag_test = get_tf_data(self.test_directories, length, dim_y, max_val, lambda x: MhaLoader.read_generator(x, MhaLoader.get_files))

        
        
class MergedDataLoader:
    def __init__(self, length, dim_y):
        volunteer_ds = VolunteerDataLoader(length, dim_y)
        comodo_ds = ComodoDataLoader(length, dim_y)
        
        self.ds_train = volunteer_ds.sag_train.concatenate(comodo_ds.sag_train)
        self.ds_test = volunteer_ds.sag_test.concatenate(comodo_ds.sag_test)
        