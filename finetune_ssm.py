import argparse
import datetime
import os
import sys
import json
from tqdm import tqdm
import io
import matplotlib.pyplot as plt
import numpy as np
import inspect
import math

import tensorflow as tf
import tensorflow_probability as tfp

from config import get_config
from src.flow_models import fKVAE
from src.datasetLoader import TensorflowDatasetLoader
from src.lgssm import LGSSM, get_cholesky
from src.utils import latent_plot, A_to_image
    

def get_image_summary(model, data):    
    latent_data = model.get_obs_distributions(x_sample=data[0], mask=data[1])
    return {"Eigenvalues": A_to_image(model.A),
           'Latent process': latent_plot(latent_data)}

def loss(model, data):
    total_log_prob = -tf.reduce_mean(model.log_prob(data))
    return total_log_prob

def grad(model, inputs):
    with tf.GradientTape() as tape:
        lgssm = model.get_LGSSM(inputs[:,0,...])
        loss_value = loss(lgssm, inputs)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def mle_run(data_generator, config, model, org_model, optimizer, model_log_dir, epochs=1000, verbose=False):
    train_log_dir = model_log_dir + '/train'
    img_log_dir = model_log_dir + '/img'
    
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    loss_sum_train_metric = tf.keras.metrics.Mean()    
    file_writer = tf.summary.create_file_writer(img_log_dir)
    
    len_dataset = int(len(data_generator.idxs))
    
    dataset = data_generator.data
    dataset = dataset.shuffle(buffer_size=len_dataset).batch(config.batch_size)
    
    
    img_seq, _ = data_generator._read_entire_video(data_generator.idxs[0])
    img_seq = img_seq[None,...]
    mu, sigma = org_model.encoder(img_seq, training =False)
    x_vae = org_model.encoder_dist(tf.concat([mu, sigma], axis=-1)).sample()
    
    mask = np.ones(shape=(1,x_vae.shape[1])).astype('bool')
    known = np.arange(x_vae.shape[1])    
    known = known[:50]
    mask[0, known] = False    
    plot_data = [x_vae, mask]    
    
    for epoch in range(1, epochs+1):
        train_log = tqdm(total=len_dataset//config.batch_size, desc='Train {0} '.format(epoch), position=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
        
        for i, data in enumerate(dataset):
            mu, sigma = org_model.encoder(data[0])
            x_vae = org_model.encoder_dist(tf.concat([mu, sigma], axis=-1)).sample()
            
            loss_value, grads = grad(model, x_vae)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            loss_sum_train_metric(loss_value)            
            metrices = {'loss': loss_value.numpy()}
            train_log.set_postfix(metrices)
            train_log.update(1)
        
        train_log.close()
        with train_summary_writer.as_default(step=epoch):
            tf.summary.scalar('loss', loss_sum_train_metric.result())
        
        if epoch % 5 == 0:
            result_dict = get_image_summary(model, plot_data)
            with file_writer.as_default():
                [tf.summary.image(l, result_dict[l], step=epoch) for l in result_dict.keys()]
                 
    return


def main(model_path,
         num_epochs = 50,
         dim_y = (112,112),
         dim_x = 16,
         dim_z = 32, 
         dec_in = 16,
         gpu = '0',          
         start_epoch = 1, 
         prefix = None, 
         ds_path = '/data/Niklas/EchoNet-Dynamics', 
         ds_size = 1, 
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
                                     num_epochs = num_epochs,
                                     batch_size = batch_size,
                                     plot_epoch = 2)
    
    log_folder = 'finetuned_ssm' 
    if prefix is not None:
        model_log_dir = 'logs/{0}/{1}'.format(log_folder, prefix)
    else:
        model_log_dir = 'logs/{0}/{1}'.format(log_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # Load saved model
    org_model = fKVAE(config = config)
    org_model.load_weights(config.model_path).expect_partial()   
    
    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu
        
    
    # Data    
    data_generator = TensorflowDatasetLoader(root=config.ds_path,
                                             image_shape=tuple(config.dim_y),
                                             split='val',
                                             size = config.ds_size)
    
    #with open(model_log_dir + '/config.json', 'w') as f:
    #    json.dump(config_dict, f)
        
    # Create LGSSM
    ss_model = LGSSM(config)
    ss_model.set_weights(org_model.lgssm.get_weights())  
    
    optimizer = tf.optimizers.Adam(0.01)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=ss_model)
    
    mle_run(data_generator, config, ss_model, org_model, optimizer, model_log_dir, epochs=config.num_epochs, verbose=True)
    
    new_model = fKVAE(config = config)
    new_model.encoder = org_model.encoder
    new_model.lgssm = ss_model
    new_model.decoder = org_model.decoder
    
    new_model.save_weights(model_log_dir + '/model')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('-saved_model','--saved_model', help='model path if continue running model (default:None)', required=True)
    parser.add_argument('-num_epochs', '--num_epochs', type=int, help='number of epochs (default 50)', default=50)
    
    parser.add_argument('-y', '--dim_y', type=tuple, help='dimension of image variable (default (112,112))', default=(112,112))
    parser.add_argument('-x', '--dim_x', type=int, help='dimension of latent variable (default 16)', default=16)
    parser.add_argument('-z', '--dim_z', type=int, help='dimension of state space variable (default 32)', default=32)
    parser.add_argument('-dec_in', '--dec_in', type=int, help='input dim to decoder (default 16)', default=16)
    parser.add_argument('-start_epoch','--start_epoch', type=int, help='start epoch', default=1)    
    parser.add_argument('-gpu','--gpus', help='comma separated list of GPUs (default -1 (CPU))', default='-1')
    parser.add_argument('-prefix','--prefix', help='predix for log folder (default:None)', default=None)
    
    # data set
    parser.add_argument('-ds_path','--ds_path', help='path to dataset (default:/data/Niklas/EchoNet-Dynamics)', default='/data/Niklas/EchoNet-Dynamics')
    parser.add_argument('-ds_size','--ds_size', type=int, help='Size of datasets', default=None)
    
    args = parser.parse_args()
    
    main(model_path = args.saved_model,
         num_epochs = args.num_epochs,
         dim_y = args.dim_y,
         dim_x = args.dim_x,
         dim_z = args.dim_z, 
         dec_in = args.dec_in,
         gpu = args.gpus, 
         start_epoch = args.start_epoch,
         prefix = args.prefix,
         ds_path = args.ds_path, 
         ds_size = args.ds_size)