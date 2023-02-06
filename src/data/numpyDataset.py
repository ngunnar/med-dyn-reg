import tensorflow as tf
import numpy as np
import glob

class NumpyLoader:    
    @staticmethod
    def get_data(sequence, size, shift=None, stride=1, drop_remainder=False):
        ds = tf.data.Dataset.from_tensor_slices(sequence)
        ds = ds.window(size=size, shift=shift, stride=stride, drop_remainder=drop_remainder)
        ds = ds.flat_map(lambda x: x.batch(size))
        return ds
    
    @staticmethod
    def crop(image, min_val=500):
        _, y_nonzero, x_nonzero = np.nonzero(image > min_val)
        return image[:, np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
    
    @staticmethod
    def read_generator(files):
        def gen():
            for file in files:
                images = np.load(file).astype('float32')[1:] # TODO: First image is much brigher, why?
                yield images
        return gen

    @staticmethod
    def process_generator(dataset, dim_y, max_val, min_val=500):
        def gen():
            for images in dataset:
                # remove outliers
                images = NumpyLoader.crop(images, min_val)
                images = tf.clip_by_value(images, min_val, max_val)
                # resize
                images = tf.image.resize(images[...,None], dim_y)
                # normalize
                images = tf.image.per_image_standardization(images)
                images = images[...,0]
                f, h, w = images.shape
                mask = np.zeros(f, dtype='bool')
                yield {'input_video': images, 'input_mask': mask}
        return gen
    
    @staticmethod
    def remove_expected(data_list, expected_patients):
        return [d for d in data_list if not any(patient in d for patient in expected_patients)]
    
    @staticmethod
    def get_tf_data(data_list, ph_steps, dim_y, max_val):
        window = ph_steps
        shift = window    
        ds_processed = tf.data.Dataset.from_generator(NumpyLoader.read_generator(data_list), 
                                                 tf.float32, tf.TensorShape([None, *dim_y]))
        ds_model = ds_processed.flat_map(lambda x: NumpyLoader.get_data(x, window, shift=shift, drop_remainder=True))
        ds_model = tf.data.Dataset.from_generator(NumpyLoader.process_generator(ds_model, dim_y, max_val, min_val=500),
                                             output_types = ({"input_video": tf.float32, 
                                                             "input_mask": tf.bool}),
                                             output_shapes = ({"input_video": (window, *dim_y),
                                                               "input_mask": (window)}))
        
        return ds_processed, ds_model
        

class ComodoDataLoader:
    def __init__(self, ph_steps, dim_y):
        
        max_val = 1500
        sag_files = glob.glob('/data/Niklas/Unity/comodo/bffe_cine_sagittal/*.npy')
        sag_train_files = sag_files[1:]
        sag_test_files = [sag_files[0]]
        expected_patients = {}
        self.sag_train_files = NumpyLoader.remove_expected(sag_train_files, expected_patients)
        self.sag_test_files = NumpyLoader.remove_expected(sag_test_files, expected_patients)
        
        self.sag_train_processed, self.sag_train = NumpyLoader.get_tf_data(self.sag_train_files, ph_steps, dim_y, max_val)
        self.sag_test_processed, self.sag_test = NumpyLoader.get_tf_data(self.sag_test_files, ph_steps, dim_y, max_val)
        

class ProKnowDataLoader:
    def __init__(self, ph_steps, dim_y):
        
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

        self.sag_train_processed, self.sag_train = NumpyLoader.get_tf_data(self.sag_train_files, ph_steps, dim_y, max_val)
        self.cor_train_processed, self.cor_train = NumpyLoader.get_tf_data(self.cor_train_files, ph_steps, dim_y, max_val)
        self.sag_test_processed, self.sag_test = NumpyLoader.get_tf_data(self.sag_test_files, ph_steps, dim_y, max_val)
        self.cor_test_processed, self.cor_test = NumpyLoader.get_tf_data(self.cor_test_files, ph_steps, dim_y, max_val)