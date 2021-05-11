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

from src.models import FLOW_KVAE
from src.datasetLoader import TensorflowDatasetLoader
from src.utils import plot_to_image, latent_plot, single_plot_to_image

def main(dim_z, gpu, model_path, start_epoch, prefix, ds_path, ds_size=None):

    config_dict = {
        # DS
        "dim_y":(64,64),
        "ph_steps":50,    
        "period": 2,
        "ds_path": ds_path,
        "ds_size": ds_size,
        #VAE
        'activation':'relu',
        'filter_size': 3,
        'filters':[64, 128, 256, 512],
        'noise_pixel_var': 0.01,
        'est_logvar':False,
        # LGSSM
        "dim_x": 16,
        "dim_z": dim_z,
        "noise_emission": 0.03,
        "noise_transition": 0.08,
        "init_cov": 30.0,
        "trainable_A":True,
        "trainable_C":True,
        "trainable_R":True,
        "trainable_Q":True,
        "trainable_mu":True,
        "trainable_sigma":True,
        "sigma_full":False,
        "sample_z": False,
        # Training
        "gpu": gpu,
        "num_epochs": 100,
        "start_epoch": start_epoch,
        "model_path": model_path,
        "batch_size": 4,
        "init_lr": 1e-4,
        "decay_steps": 20,
        "decay_rate": 0.85,
        "max_grad_norm": 150.0,
        "scale_reconstruction": 1.0,
        "kl_latent_loss_weight": 1.0,
        "kf_loss_weight": 1.0,
        "kl_growth": 3.0,
        "kl_cycle":20,
        "only_vae_epochs": 5,
        # Plotting
        "plot_epoch": 1,
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
    
    plot_train = list(train_generator.data.take(1).as_numpy_iterator())[0]
    plot_train = [plot_train[0][None,...], plot_train[1][None,...]]
    plot_test = list(test_generator.data.take(1).as_numpy_iterator())[0]
    plot_test = [plot_test[0][None,...], plot_test[1][None,...]]

    train_dataset = train_generator.data
    train_dataset = train_dataset.shuffle(buffer_size=len_train).batch(config.batch_size)
    test_dataset = test_generator.data
    test_dataset = test_dataset.batch(config.batch_size)
    
    
    # model training
    model = FLOW_KVAE(config = config)
    if config.model_path is not None:
        model.load_weights(config.model_path)
    model.compile(int(len(train_generator.idxs)/config.batch_size))
    model.epoch = config.start_epoch

    if prefix is not None:
        model_log_dir = 'logs_new/kvae_flow/{0}'.format(prefix)
    else:
        model_log_dir = 'logs_new/kvae_flow/{0}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
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

    plot_train = list(train_generator.data.take(1).as_numpy_iterator())[0]
    plot_train = [plot_train[0][None,...], plot_train[1][None,...]]
    plot_test = list(test_generator.data.take(1).as_numpy_iterator())[0]
    plot_test = [plot_test[0][None,...], plot_test[1][None,...]]

    loss_sum_train_metric = tf.keras.metrics.Mean()

    checkpoint = tf.train.Checkpoint(optimizer=model.opt, model=model)
    for epoch in range(model.epoch, config.num_epochs+1):
        #################### TRANING ##################################################
        start_time = time.time()
        beta = tf.sigmoid((epoch%model.config.kl_cycle - 1)**2/model.config.kl_growth-model.config.kl_growth)
        model.w_kl = model.config.kl_latent_loss_weight * beta
        model.w_kf = model.config.kf_loss_weight * beta
        train_log = tqdm.tqdm(total=len_train//config.batch_size, desc='Train {0} '.format(epoch), position=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
        train_loss = 0
        for i, (train_y, train_mask) in enumerate(train_dataset):
            loss = model.train_step(train_y, train_mask)
            loss_sum_train_metric(loss)
            train_log.set_postfix(loss=loss.numpy())
            train_log.update(1)
        train_log.close()

        end_time = time.time()

        train_result = {m.name: m.result().numpy() for m in model.metrics}
        [m.reset_states() for m in model.metrics]
        loss_training = loss_sum_train_metric.result()
        loss_sum_train_metric.reset_states()

        #################### TESTING ##################################################
        test_log = tqdm.tqdm(total=len_test//config.batch_size, desc='Test {0} '.format(epoch), position=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
        for i, (test_y, test_mask) in enumerate(test_dataset):
            loss = model.test_step(test_y, test_mask)
            test_log.set_postfix(loss_sum=loss.numpy())
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
            train_arg = model.predict(plot_train)
            test_arg = model.predict(plot_test)

            latent_train = model.get_latents(plot_train)
            latent_test = model.get_latents(plot_test)

            with file_writer.as_default():
                tf.summary.image("Training data", plot_to_image(plot_train[0], train_arg), step=epoch)
                tf.summary.image("Test data", plot_to_image(plot_test[0], test_arg), step=epoch)
                tf.summary.image("Latent Training data", latent_plot(latent_train), step=epoch)
                tf.summary.image("Latent Test data", latent_plot(latent_test), step=epoch)
                tf.summary.text("A", tf.strings.as_string(model.kf.kalman_filter.transition_matrix), step=epoch)
                tf.summary.text("A eig", tf.strings.as_string(tf.linalg.eig(model.kf.kalman_filter.transition_matrix)[0]), step=epoch)
                tf.summary.text("Q", tf.strings.as_string(model.kf.kalman_filter.transition_noise.stddev()), step=epoch)
                tf.summary.text("C", tf.strings.as_string(model.kf.kalman_filter.observation_matrix), step=epoch)
                tf.summary.text("R", tf.strings.as_string(model.kf.kalman_filter.observation_noise.stddev()), step=epoch)
                tf.summary.text("mu_0", tf.strings.as_string(model.kf.kalman_filter.initial_state_prior.mean()), step=epoch)
                tf.summary.text("Sigma_0", tf.strings.as_string(model.kf.kalman_filter.initial_state_prior.covariance()), step=epoch)
        else:
            with file_writer.as_default():
                tf.summary.text("A", tf.strings.as_string(model.kf.kalman_filter.transition_matrix), step=epoch)
                tf.summary.text("A eig", tf.strings.as_string(tf.linalg.eig(model.kf.kalman_filter.transition_matrix)[0]), step=epoch)
                tf.summary.text("Q", tf.strings.as_string(model.kf.kalman_filter.transition_noise.stddev()), step=epoch)
                tf.summary.text("C", tf.strings.as_string(model.kf.kalman_filter.observation_matrix), step=epoch)
                tf.summary.text("R", tf.strings.as_string(model.kf.kalman_filter.observation_noise.stddev()), step=epoch)
                tf.summary.text("mu_0", tf.strings.as_string(model.kf.kalman_filter.initial_state_prior.mean()), step=epoch)
                tf.summary.text("Sigma_0", tf.strings.as_string(model.kf.kalman_filter.initial_state_prior.covariance()), step=epoch)

        if loss_training < best_loss and model.epoch >= config.only_vae_epochs:
            model.save_weights(model_log_dir + '/kvae_flow_cardiac_best')
            best_loss = loss_training
            checkpoint.write(file_prefix=checkpoint_prefix)  # overwrite best val model
        model.epoch += 1
    model.save_weights(model_log_dir + '/kvae_flow_cardiac_{}'.format(config.dim_z))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', '--dim_z', type=int, help='dimension of state space variable (required)', default=None)
    parser.add_argument('-gpu','--gpus', help='comma separated list of GPUs (default -1 (CPU))', default=-1)
    parser.add_argument('-model','--model', help='model path if continue running model (default:None)', default=None)
    parser.add_argument('-start_epoch','--start_epoch', type=int, help='start epoch', default=1)
    parser.add_argument('-prefix','--prefix', help='predix for log folder (default:None)', default=None)
    parser.add_argument('-ds_path','--ds_path', help='path to dataset (default:/data/Niklas/EchoNet-Dynamics)', default='/data/Niklas/EchoNet-Dynamics')
    args = parser.parse_args()
    
    main(args.dim_z, args.gpus, args.model, args.start_epoch, args.prefix, args.ds_path)
    
