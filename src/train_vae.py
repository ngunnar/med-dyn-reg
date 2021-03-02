import tensorflow as tf
import numpy as np
from collections import namedtuple
import os
import numpy as np
import datetime
import json
import sys
import time
import tqdm
import argparse

from vae import VAE
from datasetLoader import TensorflowDatasetLoader
from utils import single_plot_to_image

def main(gpu, model_path, start_epoch, ds_path):

    config_dict = {
        # DS
        "dim_y":(64,64),
        "ph_steps":50,    
        "period": 2,
        "ds_path": ds_path,
        #VAE
        'activation':'relu',
        'filter_size': 3,
        'filters':[64, 128, 256, 512],
        'noise_pixel_var': 0.01,
        "dim_x": 16,
        "noise_emission":0.03,
        # Training
        "gpu": gpu,
        "num_epochs": 100,
        "start_epoch": start_epoch,
        "model_path": model_path,
        "batch_size": 16,
        "init_lr": 1e-4,
        "decay_steps": 20,
        "decay_rate": 0.85,
        "max_grad_norm": 150.0,
        "scale_reconstruction": 1.0,
        "kl_latent_loss_weight": 1.0,
        "kf_loss_weight": 1.0,
        "kl_growth": 3.0,
        # Plotting
        "plot_epoch": 5,
        }
    config = namedtuple("Config", config_dict.keys())(*config_dict.values())

    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu
    
    train_generator = TensorflowDatasetLoader(root = config.ds_path,
                                              image_shape = config.dim_y, 
                                              length = config.ph_steps, 
                                              period = config.period)
    test_generator = TensorflowDatasetLoader(root = config.ds_path,
                                             image_shape = config.dim_y,
                                             length = config.ph_steps,
                                             split='test', 
                                             period = config.period)
    len_train = int(len(train_generator.idxs))
    len_test = int(len(test_generator.idxs))
    print("Train size", len_train)
    print("Test size", len_test)
    print("Number of batches: ", int(len(train_generator.idxs)/config.batch_size))

    # model training
    model = VAE(config = config)
    if config.model_path is not None:
        model.load_weights(config.model_path)
    model.compile(int(len(train_generator.idxs)/config.batch_size))
    model.epoch = config.start_epoch

    loss_train_metric = tf.keras.metrics.Mean()
    loss_sum_train_metric = tf.keras.metrics.Mean()
    recon_train_metric = tf.keras.metrics.Mean()
    kl_train_metric = tf.keras.metrics.Mean()

    loss_sum_test_metric = tf.keras.metrics.Mean()
    recon_test_metric = tf.keras.metrics.Mean()
    kl_test_metric = tf.keras.metrics.Mean()
    kf_test_metric = tf.keras.metrics.Mean()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_log_dir = 'logs/vae/' + current_time
    train_log_dir = model_log_dir + '/train'
    test_log_dir = model_log_dir + '/test'
    img_log_dir = model_log_dir + '/img'


    file_writer = tf.summary.create_file_writer(img_log_dir)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    best_loss = sys.float_info.max
    with open(model_log_dir + '/config.json', 'w') as f:
        json.dump(config_dict, f)

    plot_train = list(train_generator.data.take(1).as_numpy_iterator())[0]
    plot_train = [plot_train[0][None,...], plot_train[1][None,...]]
    plot_test = list(test_generator.data.take(1).as_numpy_iterator())[0]
    plot_test = [plot_test[0][None,...], plot_test[1][None,...]]

    train_dataset = train_generator.data
    train_dataset = train_dataset.shuffle(buffer_size=len_train).batch(config.batch_size)
    test_dataset = test_generator.data
    test_dataset = test_dataset.batch(config.batch_size)
    
    epoch_log = tqdm.tqdm(config.num_epochs, desc='Epoch', position=0)

    for epoch in range(model.epoch, config.num_epochs+1):
        #################### TRANING ##################################################
        start_time = time.time()

        train_log = tqdm.tqdm(total=len_train//config.batch_size, desc='Training', position=0)
        for i, (train_y, train_mask) in enumerate(train_dataset):
            loss_sum, loss, recon_loss, recon_w, kl_loss, kl_w = model.train_step(train_y, train_mask)
            loss_sum_train_metric(loss_sum)
            loss_train_metric(loss)
            recon_train_metric(recon_loss)
            kl_train_metric(kl_loss)

            train_log.set_postfix(loss=loss.numpy(), 
                                  loss_sum = loss_sum.numpy(),
                                  recon_loss = recon_loss.numpy(), 
                                  kl_loss = kl_loss.numpy())
            train_log.update(1)
        train_log.close()

        end_time = time.time()

        elbo_training = -loss_train_metric.result()
        loss_training = loss_sum_train_metric.result()
        recon_training = recon_train_metric.result()
        kl_training = kl_train_metric.result()

        loss_train_metric.reset_states()
        loss_sum_train_metric.reset_states()
        recon_train_metric.reset_states()
        kl_train_metric.reset_states()

        #################### TESTING ##################################################
        test_log = tqdm.tqdm(total=len_test//config.batch_size, desc='Testing', position=0)
        for i, (test_y, test_mask) in enumerate(test_dataset):
            loss_sum, recon_loss, kl_loss = model.test_step(test_y, test_mask)
            loss_sum_test_metric(loss_sum)
            recon_test_metric(recon_loss)
            kl_test_metric(kl_loss)
            test_log.set_postfix(loss_sum=loss_sum.numpy(), 
                                 recon_loss = recon_loss.numpy(), 
                                 kl_loss = kl_loss.numpy())
            test_log.update(1)
        test_log.close()
        loss_testing = loss_sum_test_metric.result()
        recon_testing = recon_test_metric.result()
        kl_testing = kl_test_metric.result()

        loss_sum_test_metric.reset_states()
        recon_test_metric.reset_states()
        kl_test_metric.reset_states()

        #################### LOGGING AND PLOTTING ##################################################
        with train_summary_writer.as_default():
            tf.summary.scalar('accuracy', elbo_training, step=epoch)
            tf.summary.scalar('loss sum', loss_training, step=epoch)
            tf.summary.scalar('recon loss', recon_training, step=epoch)
            tf.summary.scalar('KL loss', kl_training, step=epoch)
            tf.summary.scalar('learning_rate', model.opt._decayed_lr(tf.float32), step=epoch)
            tf.summary.scalar('epoch', tf.convert_to_tensor(model.epoch), step=epoch) 
            tf.summary.scalar('weight reconstruction', tf.convert_to_tensor(recon_w), step=epoch)
            tf.summary.scalar('weight KL', tf.convert_to_tensor(kl_w), step=epoch)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss sum', loss_testing, step=epoch)
            tf.summary.scalar('recon loss', recon_testing, step=epoch)
            tf.summary.scalar('KL loss', kl_testing, step=epoch)

        if epoch % config.plot_epoch == 0:
            # Prepare the plot
            y_vae_train = model.predict(plot_train)
            y_vae_test = model.predict(plot_test)
            with file_writer.as_default():
                tf.summary.image("Training data", single_plot_to_image(plot_train[0], y_vae_train), step=epoch)
                tf.summary.image("Test data", single_plot_to_image(plot_test[0], y_vae_test), step=epoch)

        epoch_log.display('Epoch: {} {}, \n\t \
        Train set ELBO: {:.2f}, recon {:.2f}(x{:.2f}), vae {:.2f}(x{:.2f}), accuracy {:.2f} \n\t \
        Test set ELBO: {:.2f}, recon {:.2f}(x{:.2f}), vae {:.2f}(x{:.2f}) \n\t \
        Learnig rate {:.2e}, \n\t \
        time elapse for current epoch: {:.2f} \n'.format(
            epoch, model.epoch,
            loss_training, 
            recon_training, recon_w, 
            kl_training, kl_w, 
            elbo_training,
            loss_testing, 
            recon_testing, recon_w,
            kl_testing, kl_w, 
            model.opt._decayed_lr(tf.float32).numpy(),
            end_time - start_time))
        
        # Save best model
        if loss_training < best_loss:
            model.save_weights(model_log_dir + '/vae_cardiac_best')
            best_loss = loss_training
        model.epoch += 1

    model.save_weights(model_log_dir + '/vae_cardiac')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu','--gpus', help='GPUs', default=None)
    parser.add_argument('-model','--model', help='model path', default=None)
    parser.add_argument('-start_epoch','--start_epoch', type=int, help='start epoch', default=1)
    parser.add_argument('-ds_path','--ds_path', help='Path to dataset', default='/data/Niklas/EchoNet-Dynamics')
    args = parser.parse_args()
    main(args.gpus, args.model, args.start_epoch, args.ds_path)
    