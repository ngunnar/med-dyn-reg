import numpy as np

import tensorflow as tf

def crop(image, min_val=500):
    _, y_nonzero, x_nonzero = np.nonzero(image > min_val)
    return image[:, np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

def multi_crop(images, min_val=500):
    _, x_nonzero_cor, z_nonzero_cor = np.nonzero(images['coronal'] > min_val)
    _, y_nonzero_sag, z_nonzero_sag = np.nonzero(images['sagittal'] > min_val)
    _, x_nonzero_tran, y_nonzero_tran = np.nonzero(images['tranversal'] > min_val)

    min_x = np.min([np.min(x_nonzero_cor), np.min(x_nonzero_tran)])
    max_x = np.max([np.max(x_nonzero_cor), np.max(x_nonzero_cor)])

    min_y = np.min([np.min(y_nonzero_sag), np.min(y_nonzero_tran)])
    max_y = np.max([np.max(y_nonzero_sag), np.max(y_nonzero_tran)])

    min_z = np.min([np.min(z_nonzero_cor), np.min(z_nonzero_sag)])
    max_z = np.max([np.max(z_nonzero_cor), np.max(z_nonzero_sag)])


    coronal = images['coronal'][:, min_x:max_x, min_z:max_z]
    sagittal = images['sagittal'][:, min_y:max_y, min_z:max_z]
    tranversal = images['tranversal'][:, min_x:max_x, min_y:max_y]

    return coronal, sagittal, tranversal

def get_data(sequence, size, shift=None, stride=1, drop_remainder=False):
    ds = tf.data.Dataset.from_tensor_slices(sequence['input_video'])
    ds = ds.window(size=size, shift=shift, stride=stride, drop_remainder=drop_remainder)
    ds = ds.flat_map(lambda x: x.batch(size))
    ds = ds.map(lambda x: {'input_ref':sequence['input_ref'], 'input_video':x})
    return ds

def get_multi_data(sequence, size, shift=None, stride=1, drop_remainder=False):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(size=size, shift=shift, stride=stride, drop_remainder=drop_remainder)    
    ds = ds.flat_map(lambda x: tf.data.Dataset.zip((x['coronal'].batch(size),
                                                    x['sagittal'].batch(size), 
                                                    x['tranversal'].batch(size))))

    ds = ds.map(lambda x, y, z: {'coronal': x,
                                 'sagittal': y, 
                                 'tranversal': z})
    return ds

def process_generator(dataset, dim_y, max_val, min_val=500):
    def gen():
        for data in dataset:
            # remove outliers
            input_video = crop(data['input_video'], min_val)
            input_ref = crop(data['input_ref'][None,...], min_val)[0,...]

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
            yield {'input_video': input_video, 'input_ref': input_video[0,...], 'input_mask': mask} # TODO change here to use reference image instead
            #yield {'input_video': input_video, 'input_ref': input_ref, 'input_mask': mask} 
    return gen

def multi_process_generator(dataset, dim_y, max_val, min_val=500):
    def gen():
        for data in dataset:
            # remove outliers            
            #coronal, sagittal, tranversal = multi_crop(data, min_val)
            coronal = crop(data['coronal'], min_val)
            sagittal = crop(data['sagittal'], min_val)
            tranversal = crop(data['tranversal'], min_val)

            coronal = tf.clip_by_value(coronal, 0, max_val)
            sagittal = tf.clip_by_value(sagittal, 0, max_val)
            tranversal = tf.clip_by_value(tranversal, 0, max_val)
            
            # resize
            coronal = tf.image.resize(coronal[...,None], dim_y)
            sagittal = tf.image.resize(sagittal[...,None], dim_y)
            tranversal = tf.image.resize(tranversal[...,None], dim_y)

            # normalize
            coronal = tf.image.per_image_standardization(coronal)
            sagittal = tf.image.per_image_standardization(sagittal)
            tranversal = tf.image.per_image_standardization(tranversal)

            coronal = coronal[...,0]
            sagittal = sagittal[...,0]
            tranversal = tranversal[...,0]
            yield {'coronal': coronal, 'sagittal': sagittal, 'tranversal': tranversal} 
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
                                        output_types=({'coronal': tf.float32, 'sagittal': tf.float32, 'tranversal': tf.float32}),
                                        output_shapes= ({'coronal': tf.TensorShape([None, None, None]), 
                                                         'sagittal': tf.TensorShape([None, None, None]),
                                                         'tranversal': tf.TensorShape([None, None, None])}))   
    
    # Process dataset
    ds_model = ds.flat_map(lambda x: get_multi_data(x, window, shift=shift, drop_remainder=True))
    ds_model = tf.data.Dataset.from_generator(multi_process_generator(ds_model, dim_y, max_val, min_val=min_val),
                                                output_types = ({'coronal': tf.float32, 'sagittal': tf.float32, 'tranversal': tf.float32}),
                                                output_shapes = ({'coronal': (window, *dim_y),
                                                                  'sagittal': (window, *dim_y),
                                                                  'tranversal': (window, *dim_y)}))
    
    return ds, ds_model