import os
os.environ["CUDA_VISIBLE_DEVICES"]='-1'

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from src.models.fkvae import fKVAE
from src.models.lgssm import FineTunedLGSSM
from src.result_utils import get_config

from src.data.mhaDataset import VolunteerDataLoader


model_dir = './logs/unity_sag/KVAE_32_64/'
model_path = model_dir + 'end_model'

config, config_dict = get_config(model_dir)

# Data
dataset_loader = VolunteerDataLoader(config.ph_steps, config.dim_y)

train_dataset = dataset_loader.sag_train
test_dataset = dataset_loader.sag_test

ds_type = 'train'

if ds_type == 'train':
    dataset = train_dataset
else:
    dataset = test_dataset
len_data = sum(dataset.map(lambda x: 1).as_numpy_iterator())

#dataset = dataset.shuffle(buffer_size=len_data).batch(config.batch_size, drop_remainder=True)
dataset = dataset.batch(config.batch_size, drop_remainder=True)
tqdm.write("Train size {0}".format(len_data))
tqdm.write("Number of batches: {0}".format(np.ceil(len_data/config.batch_size)))


model = fKVAE(config = config)
if model_path is not None:
    model.load_weights(model_path)

ds = np.asarray([model.encoder(inputs['input_video'], False)[0].sample() for inputs in dataset])
ds = np.reshape(ds, (-1, config.ph_steps, config.dim_x))
masks = np.zeros((ds.shape[0], ds.shape[1]), dtype='bool')
tqdm.write("Data shape {0}".format(ds.shape))

dataset = tf.data.Dataset.from_tensor_slices(({'x':ds, 'mask':masks}))
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

full_latent_model = FineTunedLGSSM(config)

full_latent_model.compile(optimizer = tf.keras.optimizers.Adam(1e-3))
full_latent_model.fit(dataset, epochs = 100)
full_latent_model.save_weights('lgssm_model')