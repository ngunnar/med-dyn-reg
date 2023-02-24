import os
import argparse

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from src.models.fkvae import fKVAE
from src.models.lgssm import FineTunedLGSSM
from src.result_utils import get_config

from src.data.mhaDataset import VolunteerDataLoader

def main(org_model_path, 
         loss_metric,
         gpu = '0',
         log_dir = './logs/lgssm/'):

    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    model_path = org_model_path + 'end_model'

    config, config_dict = get_config(org_model_path)

    # Data
    dataset_loader = VolunteerDataLoader(config.length, config.dim_y)

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
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model = fKVAE(config = config)
    if model_path is not None:
        model.load_weights(model_path)

    ds_x = []
    ds_x_ref = []
    for inputs in dataset:
        y = inputs['input_video']
        y_ref = inputs['input_ref']
        q_enc, q_ref_enc, x_ref_feat = model.encoder(y, y_ref, False)
        ds_x.append(q_enc.sample())
        ds_x_ref.append(q_ref_enc.sample())
        
    ds_x = np.asarray(ds_x)
    ds_x = np.reshape(ds_x, (-1, config.length, config.dim_x))

    ds_x_ref = np.asarray(ds_x_ref)
    ds_x_ref = np.reshape(ds_x_ref, (-1, config.dim_x))

    masks = np.zeros((ds_x.shape[0], ds_x.shape[1]), dtype='bool')
    tqdm.write("Data shape {0}".format(ds_x.shape))
    tqdm.write("Data ref shape {0}".format(ds_x_ref.shape))

    dataset = tf.data.Dataset.from_tensor_slices(({'x':ds_x, 'x_ref':ds_x_ref, 'mask':masks}))
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    full_latent_model = FineTunedLGSSM(config, loss_metric = loss_metric)

    full_latent_model.compile(optimizer = tf.keras.optimizers.Adam(1e-3))
    full_latent_model.fit(dataset, epochs = 300, callbacks=[tensorboard_callback])
    full_latent_model.save_weights(log_dir + '/end_model')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('-org_model_path', '--org_model_path', help='path to original model')
    parser.add_argument('-loss_metric', '--loss_metric', choices=['ll', 'other'], help='path to original model')
    parser.add_argument('-gpu','--gpus', help='comma separated list of GPUs (default -1 (CPU))', default='-1')
    parser.add_argument('-log_dir','--log_dir', help='Loggin directory', default='./logs/lgssm/')
    
    args = parser.parse_args()
    
    main(org_model_path = args.org_model_path,
         loss_metric = args.loss_metric,
         gpu = args.gpus,
         log_dir = args.log_dir)