import numpy as np

import tensorflow as tf

def crop(image, min_val=500):
    _, y_nonzero, x_nonzero = np.nonzero(image > min_val)
    return image[:, np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

def multi_crop(images, min_val=500):
    _, x_nonzero_cor, z_nonzero_cor = np.nonzero(images['coronal'] > min_val)
    _, y_nonzero_sag, z_nonzero_sag = np.nonzero(images['sagittal'] > min_val)
    _, x_nonzero_tran, y_nonzero_tran = np.nonzero(images['transversal'] > min_val)

    min_x = np.min([np.min(x_nonzero_cor), np.min(x_nonzero_tran)])
    max_x = np.max([np.max(x_nonzero_cor), np.max(x_nonzero_cor)])

    min_y = np.min([np.min(y_nonzero_sag), np.min(y_nonzero_tran)])
    max_y = np.max([np.max(y_nonzero_sag), np.max(y_nonzero_tran)])

    min_z = np.min([np.min(z_nonzero_cor), np.min(z_nonzero_sag)])
    max_z = np.max([np.max(z_nonzero_cor), np.max(z_nonzero_sag)])


    coronal = images['coronal'][:, min_x:max_x, min_z:max_z]
    sagittal = images['sagittal'][:, min_y:max_y, min_z:max_z]
    transversal = images['transversal'][:, min_x:max_x, min_y:max_y]

    return coronal, sagittal, transversal

def get_data(sequence, size, shift=None, stride=1, drop_remainder=False):
    ds = tf.data.Dataset.from_tensor_slices(sequence['input_video'])
    ds = ds.window(size=size, shift=shift, stride=stride, drop_remainder=drop_remainder)
    ds = ds.flat_map(lambda x: x.batch(size))
    ds = ds.map(lambda x: {'input_ref':sequence['input_ref'], 'input_video':x})
    return ds

def get_multi_data(sequence, size, shift=None, stride=1, drop_remainder=False):
    ds = tf.data.Dataset.from_tensor_slices({'coronal': sequence['coronal'],
                                             'sagittal': sequence['sagittal'],
                                             'transversal':sequence['transversal']})
    ds = ds.window(size=size, shift=shift, stride=stride, drop_remainder=drop_remainder)    
    ds = ds.flat_map(lambda x: tf.data.Dataset.zip((x['coronal'].batch(size),
                                                    x['sagittal'].batch(size), 
                                                    x['transversal'].batch(size))))

    ds = ds.map(lambda x, y, z: {'coronal': x, 
                                 'coronal_ref': sequence['coronal_ref'],
                                 'sagittal': y,  
                                 'sagittal_ref': sequence['sagittal_ref'],
                                 'transversal': z, 
                                 'transversal_ref': sequence['transversal_ref']})
    return ds


def preprocess_data(video, ref_img, dim_y, max_val, min_val):
    # remove outliers
    input_video = crop(video, min_val)
    input_ref = crop(ref_img[None,...], min_val)[0,...]

    input_video = tf.clip_by_value(input_video, 0, max_val)
    input_ref = tf.clip_by_value(input_ref, 0, max_val)
    
    # resize
    input_video = tf.image.resize(input_video[...,None], dim_y)
    input_ref = tf.image.resize(input_ref[...,None], dim_y)

    # normalize
    input_video = tf.image.per_image_standardization(input_video)
    input_ref = tf.image.per_image_standardization(input_ref)

    input_video = input_video[...,0]
    input_ref = input_ref[...,0]
    f, _, _ = input_video.shape
    mask = np.zeros(f, dtype='bool')
    return input_video, input_ref, mask

def process_generator(dataset, dim_y, max_val, min_val=500):
    def gen():
        for data in dataset:
            input_video, input_ref, mask = preprocess_data(data['input_video'], data['input_ref'], dim_y, max_val, min_val)            
            #yield {'input_video': input_video, 'input_ref': input_video[0,...], 'input_mask': mask} # TODO change here to use reference image instead
            yield {'input_video': input_video, 'input_ref': input_ref, 'input_mask': mask} 
    return gen

def multi_process_generator(dataset, dim_y, max_val, min_val=500):
    def gen():
        for data in dataset:            
            input_video_cor, input_ref_cor, _ = preprocess_data(data['coronal'], data['coronal_ref'], dim_y, max_val, min_val)
            input_video_sag, input_ref_sag, _ = preprocess_data(data['sagittal'], data['sagittal_ref'], dim_y, max_val, min_val)
            input_video_trans, input_ref_trans, _ = preprocess_data(data['transversal'], data['transversal_ref'], dim_y, max_val, min_val)
            yield {'coronal': input_video_cor, 
                   'coronal_ref': input_ref_cor, 
                   'sagittal': input_video_sag, 
                   'sagittal_ref': input_ref_sag, 
                   'transversal': input_video_trans,
                   'transversal_ref': input_ref_trans} 
    return gen

def get_tf_data(data_list, ph_steps, dim_y, max_val, read_generator, min_val=500):
    window = ph_steps
    shift = window
    # Read files
    ds = tf.data.Dataset.from_generator(read_generator(data_list), 
                                        output_types=({'input_ref': tf.float32, 'input_video': tf.float32}),
                                        output_shapes= ({'input_ref': tf.TensorShape([None, None]), 'input_video': tf.TensorShape([None, None, None])}))   
    
    # Process dataset
    ds_model = ds.flat_map(lambda x: get_data(x, window, shift=shift, drop_remainder=True))
    ds_model = tf.data.Dataset.from_generator(process_generator(ds_model, dim_y, max_val, min_val=min_val),
                                                output_types = ({'input_video': tf.float32, 'input_ref': tf.float32, 'input_mask': tf.bool}),
                                                output_shapes = ({'input_video': (window, *dim_y), 'input_ref': dim_y, 'input_mask': (window)}))
    
    return ds, ds_model

def get_tf_multi_data(data_list, ph_steps, dim_y, max_val, read_generator, min_val=500):
    window = ph_steps
    shift = window
    # Read files
    ds = tf.data.Dataset.from_generator(read_generator(data_list), 
                                        output_types=({'coronal': tf.float32, 'coronal_ref': tf.float32,
                                                       'sagittal': tf.float32, 'sagittal_ref': tf.float32, 
                                                       'transversal': tf.float32, 'transversal_ref': tf.float32}),
                                        output_shapes= ({'coronal': tf.TensorShape([None, None, None]), 'coronal_ref': tf.TensorShape([None, None]), 
                                                         'sagittal': tf.TensorShape([None, None, None]), 'sagittal_ref': tf.TensorShape([None, None]),
                                                         'transversal': tf.TensorShape([None, None, None]), 'transversal_ref': tf.TensorShape([None, None])}))   
    
    # Process dataset
    ds_model = ds.flat_map(lambda x: get_multi_data(x, window, shift=shift, drop_remainder=True))
    ds_model = tf.data.Dataset.from_generator(multi_process_generator(ds_model, dim_y, max_val, min_val=min_val),
                                                output_types = ({'coronal': tf.float32, 
                                                                 'coronal_ref': tf.float32, 
                                                                 'sagittal': tf.float32, 
                                                                 'sagittal_ref': tf.float32, 
                                                                 'transversal': tf.float32,
                                                                 'transversal_ref': tf.float32}),
                                                output_shapes = ({'coronal': (window, *dim_y),
                                                                  'coronal_ref': dim_y,
                                                                  'sagittal': (window, *dim_y),
                                                                  'sagittal_ref': dim_y,
                                                                  'transversal': (window, *dim_y),
                                                                  'transversal_ref': dim_y}))
    
    return ds, ds_model