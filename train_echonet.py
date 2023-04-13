
import tensorflow as tf
import os
import datetime
import argparse
import json

from config import get_config
from src.models.fkvae import fKVAE
from src.data.datasetLoader import TensorflowDatasetLoader
from src.callbacks import SetLossWeightsCallback, VisualizeResultCallback

def main(dim_y = (112,112),
         dim_x = 16,
         dim_z = 32, 
         gpu = '0',
         int_steps = 0,
         model_path = None, 
         start_epoch = 1, 
         prefix = None, 
         skip_connection = False,
         ds_path = '/data/Niklas/EchoNet-Dynamics', 
         ds_size = None, 
         batch_size = 4):
    
    config, config_dict = get_config(ds_path = ds_path,
                                     ds_size = ds_size, 
                                     dim_y = dim_y, 
                                     dim_x = dim_x,
                                     dim_z = dim_z,                                     
                                     int_steps = int_steps,
                                     gpu = gpu, 
                                     start_epoch = start_epoch,
                                     skip_connection = skip_connection,
                                     model_path = model_path, 
                                     init_cov = 1.0,
                                     enc_filters = [16, 32, 64, 128],
                                     dec_filters = [16, 32, 64, 128],
                                     num_epochs = 50,
                                     batch_size = batch_size,
                                     plot_epoch = 5)

    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu

    # Data
    train_generator = TensorflowDatasetLoader(root = config.ds_path,
                                              image_shape = config.dim_y, 
                                              length = config.length, 
                                              period = config.period,
                                              size = config.ds_size)
    test_generator = TensorflowDatasetLoader(root = config.ds_path,
                                             image_shape = config.dim_y,
                                             length = config.length,
                                             split='test', 
                                             period = config.period,
                                             size = config.ds_size)

    plot_train = list(train_generator.data.batch(1).take(1))[0]
    plot_test = list(test_generator.data.batch(1).take(1))[0]

    len_train = int(len(train_generator.idxs))
    num_batches = (len(train_generator.idxs)/config.batch_size)

    train_dataset = train_generator.data
    train_dataset = train_dataset.shuffle(buffer_size=len_train).batch(config.batch_size)

    test_dataset = test_generator.data
    test_dataset = test_dataset.batch(config.batch_size)

    
    
    # Logging and callbacks
    log_folder = 'EchoNet'

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
        save_freq=5*config.batch_size)

    img_log_dir = log_dir + '/img'
    file_writer = tf.summary.create_file_writer(img_log_dir)

    visualizeresult_callback = VisualizeResultCallback(file_writer, plot_train, plot_test, log_interval=config.plot_epoch)
    
    # model
    fkvae = fKVAE(config)
    fkvae.compile(num_batches = num_batches)

    if config.model_path is not None:
        fkvae.load_weights(config.model_path)

    fkvae.save_weights(checkpoint_filepath.format(epoch=0))
    with open(log_dir + '/config.json', 'w') as f:
        json.dump(config_dict, f)
    fkvae.fit(train_dataset, 
              validation_data = test_dataset,
              epochs = config.num_epochs, 
              verbose = 1, 
              callbacks=[lossweight_callback, 
                         tensorboard_callback, 
                         visualizeresult_callback,
                         model_checkpoint_callback])


    fkvae.save_weights(log_dir + '/end_model')
    fkvae.evaluate(test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('-y', '--dim_y', type=tuple, help='dimension of image variable (default %(default))', default=(112,112))
    parser.add_argument('-x', '--dim_x', type=int, help='dimension of latent variable (default %(default))', default=16)
    parser.add_argument('-z', '--dim_z', type=int, help='dimension of state space variable (default %(default))', default=32)
    parser.add_argument('-int_steps', '--int_steps', type=int, help='flow integration steps (default %(default))', default=0)
    parser.add_argument('-saved_model','--saved_model', help='model path if continue running model (default:%(default))', default=None)
    parser.add_argument('-start_epoch','--start_epoch', type=int, help='start epoch', default=1)
    parser.add_argument('-skip_connection', '--skip_connection',choices=["False", "True"], help='skip connection (default %(default))', default="False")
    parser.add_argument('-gpu','--gpus', help='comma separated list of GPUs (default %(default))', default='-1')
    parser.add_argument('-prefix','--prefix', help='predix for log folder (default:%(default))', default=None)
    
    # data set
    parser.add_argument('-ds_path','--ds_path', help='path to dataset (default:%(default))', default='/data/Niklas/EchoNet-Dynamics')
    parser.add_argument('-ds_size','--ds_size', type=int, help='Size of datasets', default=None)
    
    args = parser.parse_args()
    
    skip_connection = args.skip_connection == "True"
    
    main(dim_y = args.dim_y,
         dim_x = args.dim_x,
         dim_z = args.dim_z, 
         gpu = args.gpus,
         int_steps = args.int_steps,
         model_path = args.saved_model, 
         skip_connection = skip_connection,
         start_epoch = args.start_epoch,
         prefix = args.prefix,
         ds_path = args.ds_path, 
         ds_size = args.ds_size)