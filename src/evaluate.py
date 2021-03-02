import numpy as np
from kvae import KVAE
from vae import VAE
from collections import namedtuple
import tensorflow as tf
import os
import numpy as np
from datasetLoader import TensorflowDatasetLoader
import pickle
import json

def get_config(path):
    print(path)
    print('%s/%s' % (os.path.dirname(path), 'config.json'))
    print(os.path.dirname(path))
    with open('%s/%s' % (os.path.dirname(path), 'config.json')) as data_file:
        config_dict = json.load(data_file)
    config = namedtuple("Config", config_dict.keys())(*config_dict.values())
    return config, config_dict

def mse(pred, true, bs, s):
    mse_i = tf.reduce_mean(tf.square(pred-true), axis=[1,2])
    mse_i = tf.reshape(mse_i, (bs, s))
    return mse_i
    
def ssim(pred, true, bs, s):
    ssim_i = tf.image.ssim(pred[...,None], true[...,None], max_val=2.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    ssim_i = tf.reshape(ssim_i, (bs, s))
    return ssim_i

path_kvae = './saved_models/kvae/'
path_vae = './saved_models/vae/'

kvae_dir = path_kvae
kvae_config, kvae_config_dict = get_config(kvae_dir)
kvae_model = KVAE(config = kvae_config)
kvae_model.load_weights(kvae_dir + '/kvae_cardiac').expect_partial()

vae_dir = path_vae
vae_config, vae_config_dict = get_config(vae_dir)
vae_model = VAE(config = vae_config)
vae_model.load_weights(vae_dir + '/vae_cardiac').expect_partial()

val_generator = TensorflowDatasetLoader(root = kvae_config.ds_path, image_shape = tuple(kvae_config.dim_y), length = kvae_config.ph_steps, split='val', period = kvae_config.period)

val_dataset = val_generator.data.batch(kvae_config.batch_size)
known_all = np.arange(kvae_config.ph_steps)
kvae = {}
for i in [1,2,3,5,7,10]:
    kvae[i] = {
        'smooth': {'ssim':[], 'mse':[]},
        'filt': {'ssim':[], 'mse':[]}
    }

obs = {'ssim': [], 'mse': []}
vae = {'ssim': [], 'mse': []}

for i, (val_y, val_mask) in enumerate(val_dataset):
    print(i)
    bs = val_y.shape[0]
    s = val_y.shape[1]
    val_y_reshaped = tf.reshape(val_y, (-1, *tf.shape(val_y)[2:]))
    for m in kvae:
        mask = np.ones((bs, s)).astype('bool')
        known = known_all[::m]
        mask[:, known] = False
        y_filt, y_smooth, y_obs = kvae_model.predict([val_y, mask])
        if m == 1:
            y_vae = vae_model.predict([val_y, mask])
            y_vae = tf.reshape(y_vae, (-1, *tf.shape(val_y)[2:]))
            y_obs = tf.reshape(y_obs, (-1, *tf.shape(y_obs)[2:]))
            obs['mse'].append(mse(y_obs, val_y_reshaped, bs, s))
            obs['ssim'].append(ssim(y_obs, val_y_reshaped, bs, s))
            vae['mse'].append(mse(y_vae, val_y_reshaped, bs, s))
            vae['ssim'].append(ssim(y_vae, val_y_reshaped, bs, s))
            
        y_smooth = tf.reshape(y_smooth, (-1, *tf.shape(y_smooth)[2:]))
        y_filt = tf.reshape(y_filt, (-1, *tf.shape(y_filt)[2:]))
        kvae[m]['smooth']['mse'].append(mse(y_smooth, val_y_reshaped, bs, s))
        kvae[m]['smooth']['ssim'].append(ssim(y_smooth, val_y_reshaped, bs, s))
        kvae[m]['filt']['mse'].append(mse(y_filt, val_y_reshaped, bs, s))
        kvae[m]['filt']['ssim'].append(ssim(y_filt, val_y_reshaped, bs, s))

for m in kvae:
    kvae[m]['smooth']['mse_tf'] = tf.concat(kvae[m]['smooth']['mse'], axis=0)
    kvae[m]['smooth']['ssim_tf'] = tf.concat(kvae[m]['smooth']['ssim'], axis=0)
    kvae[m]['filt']['mse_tf'] = tf.concat(kvae[m]['filt']['mse'], axis=0)
    kvae[m]['filt']['ssim_tf'] = tf.concat(kvae[m]['filt']['ssim'], axis=0)

    
obs['mse_tf'] = tf.concat(obs['mse'], axis=0)
obs['ssim_tf'] = tf.concat(obs['ssim'], axis=0)

vae['mse_tf'] = tf.concat(vae['mse'], axis=0)
vae['ssim_tf'] = tf.concat(vae['ssim'], axis=0)

for m in kvae:
    kvae[m]['smooth']['mse_mean'] = tf.reduce_mean(kvae[m]['smooth']['mse_tf'], axis=0)
    kvae[m]['smooth']['ssim_mean'] = tf.reduce_mean(kvae[m]['smooth']['ssim_tf'], axis=0)
    kvae[m]['filt']['mse_mean'] = tf.reduce_mean(kvae[m]['filt']['mse_tf'], axis=0)
    kvae[m]['filt']['ssim_mean'] = tf.reduce_mean(kvae[m]['filt']['ssim_tf'], axis=0)

obs['mse_mean'] = tf.reduce_mean(obs['mse_tf'], axis=0)
obs['ssim_mean'] = tf.reduce_mean(obs['ssim_tf'], axis=0)

vae['mse_mean'] = tf.reduce_mean(vae['mse_tf'], axis=0)
vae['ssim_mean'] = tf.reduce_mean(vae['ssim_tf'], axis=0)

with open('./saved_data/kvae.pickle', 'wb') as handle:
    pickle.dump(kvae, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('./saved_data/obs.pickle', 'wb') as handle:
    pickle.dump(obs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('./saved_data/vae.pickle', 'wb') as handle:
    pickle.dump(vae, handle, protocol=pickle.HIGHEST_PROTOCOL)