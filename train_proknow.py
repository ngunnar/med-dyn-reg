import tensorflow as tf
import os
import datetime
import argparse
import json
import numpy as np
import glob

from config import get_config
from src.models.fkvae import fKVAE
from src.callbacks import SetLossWeightsCallback, VisualizeResultCallback
from src.data.numpyDataset import ProKnowDataLoader, ComodoDataLoader
          
def main(dim_y = (112,112),
         dim_x = 16,
         dim_z = 32, 
         dec_in = 16,
         gpu = '0',
         int_steps = 0,
         length = 50,
         model_path = None, 
         start_epoch = 1, 
         prefix = None,
         skip_connection = False,
         losses = ['kvae_loss', 'grad']):
    num_batches = 4
    config, config_dict = get_config(dim_y = dim_y, 
                                     dim_x = dim_x,
                                     dim_z = dim_z,                                     
                                     dec_input_dim = dec_in,
                                     skip_connection = skip_connection,
                                     losses = losses,
                                     int_steps = int_steps,
                                     length = length,
                                     gpu = gpu, 
                                     start_epoch = start_epoch,
                                     model_path = model_path, 
                                     init_cov = 1.0,
                                     #enc_filters = [64, 128, 256, 512],
                                     #dec_filters = [64, 128, 256, 512],
                                     enc_filters = [16, 32, 64, 128],
                                     dec_filters = [16, 32, 64, 128],
                                     init_lr = 1e-4,
                                     num_epochs = 100,
                                     batch_size = num_batches,
                                     plot_epoch = 5)

    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu
    
    # Data
    dataset_loader = ComodoDataLoader(config.length, config.dim_y)
    train_dataset = dataset_loader.sag_train
    test_dataset = dataset_loader.sag_test
    
    len_data = sum(train_dataset.map(lambda x: 1).as_numpy_iterator())
    
    # Put it before shuffle to get same plot images every time
    plot_train = list(train_dataset.take(1))[0]
    plot_train = {'input_video': plot_train['input_video'][None,...], 'input_ref': plot_train['input_ref'][None,...], 'input_mask': plot_train['input_mask'][None,...]}
    
    plot_test= list(test_dataset.take(1))[0]
    plot_test = {'input_video': plot_test['input_video'][None,...], 'input_ref': plot_test['input_ref'][None,...], 'input_mask': plot_test['input_mask'][None,...]}
    
    train_dataset = train_dataset.shuffle(buffer_size=len_data).batch(config.batch_size, drop_remainder=True)
    test_dataset = test_dataset.shuffle(buffer_size=len_data).batch(config.batch_size, drop_remainder=True)
    # Logging and callbacks
    log_folder = 'comodo'

    if prefix is not None:
        log_dir = 'logs/{0}/{1}'.format(log_folder, prefix)
    else:
        log_dir = 'logs/{0}/{1}'.format(log_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    lossweight_callback = SetLossWeightsCallback(config.kl_growth)

    checkpoint_filepath = log_dir + '/cp-{epoch:04d}.ckpt'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, 
        verbose=1, 
        save_weights_only=True,
        save_freq='epoch')

    img_log_dir = log_dir + '/img'
    file_writer = tf.summary.create_file_writer(img_log_dir)

    visualizeresult_callback = VisualizeResultCallback(file_writer, 
                                                       train_data = plot_train, 
                                                       test_data = plot_test, 
                                                       log_interval=config.plot_epoch)
    
    # model
    fkvae = fKVAE(config)
    fkvae.compile(num_batches = np.ceil(len_data/config.batch_size))

    if config.model_path is not None:
        fkvae.load_weights(config.model_path)

    fkvae.save_weights(checkpoint_filepath.format(epoch=0))
    with open(log_dir + '/config.json', 'w') as f:
        json.dump(config_dict, f)
    fkvae.fit(train_dataset, 
              epochs = config.num_epochs, 
              verbose = 1, 
              validation_data = test_dataset,
              callbacks=[lossweight_callback, 
                         tensorboard_callback, 
                         visualizeresult_callback,
                         model_checkpoint_callback])


    fkvae.save_weights(log_dir + '/end_model')    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('-y', '--dim_y', type=tuple, help='dimension of image variable (default (112,112))', default=(112,112))
    parser.add_argument('-x', '--dim_x', type=int, help='dimension of latent variable (default 16)', default=16)
    parser.add_argument('-z', '--dim_z', type=int, help='dimension of state space variable (default 32)', default=32)
    parser.add_argument('-dec_in', '--dec_in', type=int, help='input dim to decoder (default same as x)', default=None)
    parser.add_argument('-length','--length', type=int, help='length of time sequence (default 50)', default = 50)
    parser.add_argument('-int_steps', '--int_steps', type=int, help='flow integration steps (default 0)', default=0)
    parser.add_argument('-skip_connection', '--skip_connection',choices=["False", "True"], help='skip connection (default False)', default="False")
    parser.add_argument('-ncc', '--ncc',choices=["False", "True"], help='use NCC loss (default False)', default="False")
    parser.add_argument('-saved_model','--saved_model', help='model path if continue running model (default:None)', default=None)
    parser.add_argument('-start_epoch','--start_epoch', type=int, help='start epoch', default=1)
    
    parser.add_argument('-gpu','--gpus', help='comma separated list of GPUs (default -1 (CPU))', default='-1')
    parser.add_argument('-prefix','--prefix', help='predix for log folder (default:None)', default=None)
    
    
    args = parser.parse_args()
    
    skip_connection = args.skip_connection == "True"
    ncc = args.ncc == "True"
    
    if args.dec_in is None:
        args.dec_in = args.dim_x
    
    if ncc:
        losses = ['kvae_loss', 'ncc', 'grad']
    else:
        losses = ['kvae_loss', 'grad']
    
    main(dim_y = args.dim_y,
         dim_x = args.dim_x,
         dim_z = args.dim_z, 
         dec_in = args.dec_in,
         gpu = args.gpus,
         int_steps = args.int_steps,
         skip_connection = skip_connection,
         losses = losses,
         length = args.length,
         model_path = args.saved_model, 
         start_epoch = args.start_epoch,
         prefix = args.prefix)