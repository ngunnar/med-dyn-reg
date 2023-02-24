import tensorflow as tf
import numpy as np
import glob
from .data_utils import get_tf_data

class NumpyLoader:

    @staticmethod
    def read_generator(files):
        def gen():
            for file in files:
                images = np.load(file).astype('float32')[1:] # TODO: First image is much brigher, why?
                yield {'input_ref': images[0,...], 'input_video': images}
        return gen
    
    @staticmethod
    def remove_expected(data_list, expected_patients):
        return [d for d in data_list if not any(patient in d for patient in expected_patients)]
        

class ComodoDataLoader:
    def __init__(self, length, dim_y):
        
        max_val = 1500
        sag_files = glob.glob('/data/Niklas/Unity/comodo/bffe_cine_sagittal/*.npy')
        sag_train_files = sag_files[1:]
        sag_test_files = [sag_files[0]]
        expected_patients = {}
        self.sag_train_files = NumpyLoader.remove_expected(sag_train_files, expected_patients)
        self.sag_test_files = NumpyLoader.remove_expected(sag_test_files, expected_patients)
        
        self.sag_train_processed, self.sag_train = get_tf_data(self.sag_train_files, length, dim_y, max_val, NumpyLoader.read_generator)
        self.sag_test_processed, self.sag_test = get_tf_data(self.sag_test_files, length, dim_y, max_val, NumpyLoader.read_generator)
        

class ProKnowDataLoader:
    def __init__(self, length, dim_y):
        
        max_val = 2000
        
        sag_files = glob.glob('/data/Niklas/ProKnow/sites/Iowa_MMRP/**/2d_slices/sagittal/numpy/data.npy')
        sag_train_files = sag_files[1:]
        sag_test_files = [sag_files[0]]
        
        cor_files = glob.glob('/data/Niklas/ProKnow/sites/Iowa_MMRP/**/2d_slices/coronal/numpy/data.npy')
        cor_train_files = cor_files[1:]
        cor_test_files = [cor_files[0]]
        
        expected_patients = {'UIHC^MMRP^009^MMRP^', 'UIHC^prostate^001^2^'}
        self.sag_train_files = NumpyLoader.remove_expected(sag_train_files, expected_patients)
        self.cor_train_files = NumpyLoader.remove_expected(cor_train_files, expected_patients)
        
        self.sag_test_files = NumpyLoader.remove_expected(sag_test_files, expected_patients)
        self.cor_test_files = NumpyLoader.remove_expected(cor_test_files, expected_patients)

        self.sag_train_processed, self.sag_train = get_tf_data(self.sag_train_files, length, dim_y, max_val, NumpyLoader.get_data, NumpyLoader.read_generator)
        self.cor_train_processed, self.cor_train = get_tf_data(self.cor_train_files, length, dim_y, max_val, NumpyLoader.get_data, NumpyLoader.read_generator)
        self.sag_test_processed, self.sag_test = get_tf_data(self.sag_test_files, length, dim_y, max_val, NumpyLoader.get_data, NumpyLoader.read_generator)
        self.cor_test_processed, self.cor_test = get_tf_data(self.cor_test_files, length, dim_y, max_val, NumpyLoader.get_data, NumpyLoader.read_generator)