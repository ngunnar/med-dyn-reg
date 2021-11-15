import tensorflow as tf
import os
import numpy as np
import datetime
import json
import sys
from tqdm import tqdm
import argparse

from config import get_config
from src.flow_models import fKVAE
from src.datasetLoader import TensorflowDatasetLoader

def main(dim_y = (112,112),
         dim_x = 16,
         dim_z = 32, 
         dec_in = 16,
         gpu = '0',
         model_path = None, 
         start_epoch = 1, 
         prefix = None, 
         ds_path = '/data/Niklas/EchoNet-Dynamics', 
         ds_size = None, 
         batch_size = 4):
    
    config, config_dict = get_config(ds_path = ds_path,
                                     ds_size = ds_size, 
                                     dim_y = dim_y, 
                                     dim_x = dim_x,
                                     dim_z = dim_z,                                     
                                     dec_input_dim = dec_in,
                                     use_kernel = False,
                                     gpu = gpu, 
                                     start_epoch = start_epoch,
                                     model_path = model_path, 
                                     init_cov = 1.0,
                                     enc_filters = [16, 32, 64, 128],
                                     dec_filters = [16, 32, 64, 128],
                                     num_epochs = 50,
                                     batch_size = batch_size,
                                     plot_epoch = 2)

    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu

    model = fKVAE(config = config)
    log_folder = 'fkvae'    
    model.info()
    
    train_generator = TensorflowDatasetLoader(root = config.ds_path,
                                              image_shape = config.dim_y, 
                                              length = config.ph_steps, 
                                              period = config.period,
                                              size = config.ds_size)
    test_generator = TensorflowDatasetLoader(root = config.ds_path,
                                             image_shape = config.dim_y,
                                             length = config.ph_steps,
                                             split='test', 
                                             period = config.period,
                                             size = config.ds_size)
    len_train = int(len(train_generator.idxs))
    len_test = int(len(test_generator.idxs))
    tqdm.write("Model name {0}".format(model.name))
    tqdm.write("Train size {0}".format(len_train))
    tqdm.write("Test size {0}".format(len_test))
    tqdm.write("Number of batches: {0}".format(np.ceil(len(train_generator.idxs)/config.batch_size)))

    train_dataset = train_generator.data
    train_dataset = train_dataset.shuffle(buffer_size=len_train).batch(config.batch_size)
    test_dataset = test_generator.data
    test_dataset = test_dataset.batch(config.batch_size)
    
    # model training
    if config.model_path is not None:
        model.load_weights(config.model_path)
    model.compile(int(len(train_generator.idxs)/config.batch_size))
    model.epoch = config.start_epoch

    if prefix is not None:
        model_log_dir = 'logs/{0}/{1}'.format(log_folder, prefix)
    else:
        model_log_dir = 'logs/{0}/{1}'.format(log_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_log_dir = model_log_dir + '/train'
    test_log_dir = model_log_dir + '/test'
    img_log_dir = model_log_dir + '/img'
    checkpoint_prefix = os.path.join(model_log_dir, "ckpt")

    file_writer = tf.summary.create_file_writer(img_log_dir)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    best_loss = sys.float_info.max
    with open(model_log_dir + '/config.json', 'w') as f:
        json.dump(config_dict, f)

    plot_train = list(train_generator.data.take(1))[0]
    plot_train = [p[None,...] for p in plot_train]
    plot_test = list(test_generator.data.take(1))[0]
    plot_test = [p[None,...] for p in plot_test]

    loss_sum_train_metric = tf.keras.metrics.Mean()
    checkpoint = tf.train.Checkpoint(optimizer=model.opt, model=model)
    for epoch in range(model.epoch, config.num_epochs+1):
        #################### TRANING ##################################################
        #beta = tf.sigmoid((epoch%model.config.kl_cycle - 1)**2/model.config.kl_growth-model.config.kl_growth)
        beta = tf.sigmoid((epoch-1)**2/model.config.kl_growth-model.config.kl_growth)
        model.w_kl = model.config.kl_latent_loss_weight * beta
        model.w_kf = model.config.kf_loss_weight * beta
        train_log = tqdm(total=len_train//config.batch_size, desc='Train {0} '.format(epoch), position=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
        for i, inputs in enumerate(train_dataset):
            loss, metrices = model.train_step(inputs)
            metrices = {str(key): value.numpy() for key, value in metrices.items()}
            loss_sum_train_metric(loss)
            train_log.set_postfix(metrices)
            train_log.update(1)
        train_log.close()

        train_result = {m.name: m.result().numpy() for m in model.metrics}
        [m.reset_states() for m in model.metrics]
        loss_training = loss_sum_train_metric.result()
        loss_sum_train_metric.reset_states()

        #################### TESTING ##################################################
        test_log = tqdm(total=len_test//config.batch_size, desc='Test {0} '.format(epoch), position=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
        for i, inputs in enumerate(test_dataset):
            loss, metrices = model.test_step(inputs)
            metrices = {str(key): value.numpy() for key, value in metrices.items()}
            test_log.set_postfix(metrices)
            test_log.update(1)
        test_log.close()

        test_result = {m.name: m.result().numpy() for m in model.metrics}
        [m.reset_states() for m in model.metrics]

        #################### LOGGING AND PLOTTING ##################################################
        with train_summary_writer.as_default():
            [tf.summary.scalar(l, train_result[l], step=epoch) for l in train_result.keys()]
            tf.summary.scalar('weight KL', model.w_kl, step=epoch)
            tf.summary.scalar('weight KF', model.w_kf, step=epoch)
            tf.summary.scalar('learning_rate', model.opt._decayed_lr(tf.float32), step=epoch)

        with test_summary_writer.as_default():
            [tf.summary.scalar(l, test_result[l], step=epoch) for l in test_result.keys()]
          
        if epoch % config.plot_epoch == 0:
            # Prepare the plot
            result_dict = model.get_image_summary(plot_train, plot_test)
            with file_writer.as_default():
                [tf.summary.image(l, result_dict[l], step=epoch) for l in result_dict.keys()]
        
        if loss_training < best_loss and model.epoch >= config.only_vae_epochs:
            model.save_weights(model_log_dir + '/best_model')
            best_loss = loss_training
            checkpoint.write(file_prefix=checkpoint_prefix)  # overwrite best val model
        
        if epoch % 5 == 0:
            model.save_weights(model_log_dir + '/model_{0}'.format(epoch))
        
        model.epoch += 1
    model.save_weights(model_log_dir + '/model')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('-y', '--dim_y', type=tuple, help='dimension of image variable (default (112,112))', default=(112,112))
    parser.add_argument('-x', '--dim_x', type=int, help='dimension of latent variable (default 16)', default=16)
    parser.add_argument('-z', '--dim_z', type=int, help='dimension of state space variable (default 32)', default=32)
    parser.add_argument('-dec_in', '--dec_in', type=int, help='input dim to decoder (default 16)', default=16)
    parser.add_argument('-saved_model','--saved_model', help='model path if continue running model (default:None)', default=None)
    parser.add_argument('-start_epoch','--start_epoch', type=int, help='start epoch', default=1)
    
    parser.add_argument('-gpu','--gpus', help='comma separated list of GPUs (default -1 (CPU))', default='-1')
    parser.add_argument('-prefix','--prefix', help='predix for log folder (default:None)', default=None)
    
    # data set
    parser.add_argument('-ds_path','--ds_path', help='path to dataset (default:/data/Niklas/EchoNet-Dynamics)', default='/data/Niklas/EchoNet-Dynamics')
    parser.add_argument('-ds_size','--ds_size', type=int, help='Size of datasets', default=None)
    
    args = parser.parse_args()
    
    main(dim_y = args.dim_y,
         dim_x = args.dim_x,
         dim_z = args.dim_z, 
         dec_in = args.dec_in,
         gpu = args.gpus,
         model_path = args.saved_model, 
         start_epoch = args.start_epoch,
         prefix = args.prefix,
         ds_path = args.ds_path, 
         ds_size = args.ds_size)
    