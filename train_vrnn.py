import tensorflow as tf
import os
import datetime
import argparse
import json
import numpy as np
import glob

from src.models.vrnn_model import VRNN

from src.data.mhaDataset import MergedDataLoader, VolunteerDataLoader
from src.callbacks import SetLossWeightsCallback, VisualizeResultCallback
from config import get_config

def main(dim_y = (112,112),
         dim_x = 16,
         dim_z = 32, 
         gpu = '0',
         int_steps = 0,
         length = 50,
         model_path = None, 
         start_epoch = 1, 
         prefix = None,
         skip_connection = False):
    
    batch_size = 4
    config, config_dict = get_config(dim_y = dim_y, 
                                     dim_x = dim_x,
                                     length = length,
                                     enc_filters = [16, 32, 64, 128],
                                     dec_filters = [16, 32, 64, 128],
                                     skip_connection = skip_connection,
                                     init_lr = 1e-4,
                                     num_epochs = 100,
                                     batch_size = batch_size,
                                     gpu = gpu,
                                     plot_epoch = 5,
                                     int_steps = int_steps)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu

    dataset_loader = VolunteerDataLoader(config.length, config.dim_y)
    train_dataset = dataset_loader.sag_train
    test_dataset = dataset_loader.sag_test

    len_train = sum(train_dataset.map(lambda x: 1).as_numpy_iterator())
    len_test = sum(test_dataset.map(lambda x: 1).as_numpy_iterator())

    train_dataset = train_dataset.shuffle(buffer_size=len_train).batch(config.batch_size, drop_remainder=True)
    test_dataset = test_dataset.shuffle(buffer_size=len_test).batch(config.batch_size, drop_remainder=True)

    log_folder = 'vrnn'
    if prefix is not None:
        log_dir = 'logs/{0}/{1}'.format(log_folder, prefix)
    else:
        log_dir = 'logs/{0}/{1}'.format(log_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    checkpoint_filepath = log_dir + '/cp-{epoch:04d}.ckpt'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, 
        verbose=1, 
        save_weights_only=True,
        save_freq='epoch')


    # model
    model = VRNN(config)
    num_batches = np.ceil(len_train/config.batch_size)

    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(config.init_lr,
    #                                                             decay_steps=config.decay_steps*num_batches,
    #                                                             decay_rate=config.decay_rate,                                            
    #                                                             staircase=True)  

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = config.init_lr), loss_weights=[1e-5, 1., 1e-3])    
    model.save_weights(checkpoint_filepath.format(epoch=0))
    with open(log_dir + '/config.json', 'w') as f:
        json.dump(config_dict, f)
        
    model.fit(train_dataset,
              epochs = config.num_epochs, 
              verbose = 1, 
              validation_data = test_dataset,
              callbacks=[tensorboard_callback,                      
                         model_checkpoint_callback]
             )
    model.save_weights(log_dir + '/end_model')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('-y', '--dim_y', type=tuple, help='dimension of image variable (default (112,112))', default=(112,112))
    parser.add_argument('-x', '--dim_x', type=int, help='dimension of latent variable (default 16)', default=16)
    parser.add_argument('-z', '--dim_z', type=int, help='dimension of state space variable (default 32)', default=32)
    parser.add_argument('-length','--length', type=int, help='length of time sequence (default 50)', default = 50)
    parser.add_argument('-int_steps', '--int_steps', type=int, help='flow integration steps (default 0)', default=0)
    parser.add_argument('-skip_connection', '--skip_connection',choices=["False", "True"], help='skip connection (default False)', default="False")
    parser.add_argument('-saved_model','--saved_model', help='model path if continue running model (default:None)', default=None)
    parser.add_argument('-start_epoch','--start_epoch', type=int, help='start epoch', default=1)
    
    parser.add_argument('-gpu','--gpus', help='comma separated list of GPUs (default -1 (CPU))', default='-1')
    parser.add_argument('-prefix','--prefix', help='predix for log folder (default:None)', default=None)
    
    args = parser.parse_args()
    
    skip_connection = args.skip_connection == "True"
    
    main(dim_y = args.dim_y,
         dim_x = args.dim_x,
         dim_z = args.dim_z, 
         gpu = args.gpus,
         int_steps = args.int_steps,
         skip_connection = skip_connection,
         length = args.length,
         model_path = args.saved_model, 
         start_epoch = args.start_epoch,
         prefix = args.prefix)