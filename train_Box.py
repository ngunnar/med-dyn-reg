import tensorflow as tf
import os
import numpy as np
import datetime
import json
import sys
import time
import tqdm
import argparse

from config import get_config
from src.models import KVAE
from src.flow_models import FLOW_KVAE
from src.datasetLoader import KvaeDataLoader
from src.utils import plot_to_image, latent_plot

def main(dim_y = (64,64),
         dim_x = 2,
         dim_z = 4, 
         gpu='0',
         model_path=None, 
         start_epoch=1, 
         prefix=None, 
         ds_path='/data/Niklas/kalman_vae_data',
         d_type='box',
         log_folder='box',
         K = 3,
         flow=0):
    print("FLOW", flow)
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    
    train_generator = KvaeDataLoader(root = ds_path, 
                                     d_type=d_type,
                                     image_shape = dim_y,
                                     test=False,
                                     output_first_frame=flow==1)
    test_generator = KvaeDataLoader(root = ds_path, 
                                     d_type=d_type,
                                     image_shape = dim_y,
                                     test=True,
                                     output_first_frame=flow==1)
    
    config, config_dict = get_config(
        dim_y = dim_y,
        dim_x = dim_x,
        dim_z = dim_z,
        ds_path=ds_path,
        gpu=gpu, 
        start_epoch=start_epoch, 
        model_path=model_path,
        K=K,
        ph_steps = train_generator.videos.shape[1])
    
    len_train = int(len(train_generator.videos))
    len_test = int(len(test_generator.videos))
    print("Train size", len_train)
    print("Test size", len_test)
    print("Number of batches: ", int(len(train_generator.videos)/config.batch_size))

    train_dataset = train_generator.data
    train_dataset = train_dataset.shuffle(buffer_size=len_train).batch(config.batch_size)
    test_dataset = test_generator.data
    test_dataset = test_dataset.batch(config.batch_size)
    
    
    # model training
    if flow == 1:
        model = FLOW_KVAE(config=config)
    else:
        model = KVAE(config = config)
    if config.model_path is not None:
        model.load_weights(config.model_path)
    model.compile(int(len(train_generator.videos)/config.batch_size))
    model.epoch = config.start_epoch

    if prefix is not None:
        model_log_dir = 'logs_new/{0}/{1}'.format(log_folder, prefix)
    else:
        model_log_dir = 'logs_new/{0}/{1}'.format(log_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
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
        start_time = time.time()
        beta = tf.sigmoid((epoch%model.config.kl_cycle - 1)**2/model.config.kl_growth-model.config.kl_growth)
        model.w_kl = model.config.kl_latent_loss_weight * beta
        model.w_kf = model.config.kf_loss_weight * beta
        train_log = tqdm.tqdm(total=len_train//config.batch_size, desc='Train {0} '.format(epoch), position=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
        for i, inputs in enumerate(train_dataset):
            loss, metrices = model.train_step(inputs)
            loss_sum_train_metric(loss)
            train_log.set_postfix(metrices)
            train_log.update(1)
        train_log.close()

        end_time = time.time()

        train_result = {m.name: m.result().numpy() for m in model.metrics}
        [m.reset_states() for m in model.metrics]
        loss_training = loss_sum_train_metric.result()
        loss_sum_train_metric.reset_states()

        #################### TESTING ##################################################
        test_log = tqdm.tqdm(total=len_test//config.batch_size, desc='Test {0} '.format(epoch), position=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
        for i, inputs in enumerate(test_dataset):
            loss, metrices = model.test_step(inputs)
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
            train_arg = model.predict(plot_train)
            test_arg = model.predict(plot_test)            

            latent_train = model.get_latents(plot_train)
            latent_test = model.get_latents(plot_test)
            A, A_eig, Q, C, R, mu_0, sigma_0 = model.kf.get_params()

            with file_writer.as_default():
                tf.summary.image("Training data", plot_to_image(plot_train[0], train_arg), step=epoch)
                tf.summary.image("Test data", plot_to_image(plot_test[0], test_arg), step=epoch)                  
                tf.summary.image("Latent Training data", latent_plot(latent_train), step=epoch)
                tf.summary.image("Latent Test data", latent_plot(latent_test), step=epoch)
                tf.summary.text("A", tf.strings.as_string(A), step=epoch)
                tf.summary.text("A eig", tf.strings.as_string(A_eig), step=epoch)
                tf.summary.text("Q", tf.strings.as_string(Q), step=epoch)
                tf.summary.text("C", tf.strings.as_string(C), step=epoch)
                tf.summary.text("R", tf.strings.as_string(R), step=epoch)
                tf.summary.text("mu_0", tf.strings.as_string(mu_0), step=epoch)
                tf.summary.text("Sigma_0", tf.strings.as_string(sigma_0), step=epoch)
        else:
            A, A_eig, Q, C, R, mu_0, sigma_0 = model.kf.get_params()
            with file_writer.as_default():
                tf.summary.text("A", tf.strings.as_string(A), step=epoch)
                tf.summary.text("A eig", tf.strings.as_string(A_eig), step=epoch)
                tf.summary.text("Q", tf.strings.as_string(Q), step=epoch)
                tf.summary.text("C", tf.strings.as_string(C), step=epoch)
                tf.summary.text("R", tf.strings.as_string(R), step=epoch)
                tf.summary.text("mu_0", tf.strings.as_string(mu_0), step=epoch)
                tf.summary.text("Sigma_0", tf.strings.as_string(sigma_0), step=epoch)
        
        if loss_training < best_loss and model.epoch >= config.only_vae_epochs:
            model.save_weights(model_log_dir + '/kvae_cardiac_best')
            best_loss = loss_training
            checkpoint.write(file_prefix=checkpoint_prefix)  # overwrite best val model
        model.epoch += 1
    model.save_weights(model_log_dir + '/kvae_cardiac_{}'.format(config.dim_z))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu','--gpus', help='comma separated list of GPUs (default -1 (CPU))', default=-1)
    parser.add_argument('-flow','--flow', type=int, help='flow based decoder (default:0 (False))', default=0)
    parser.add_argument('-K','--K', type=int, help='K (default:3)', default=3)
    parser.add_argument('-model','--model', help='model path if continue running model (default:None)', default=None)
    parser.add_argument('-start_epoch','--start_epoch', type=int, help='start epoch', default=1)
    parser.add_argument('-prefix','--prefix', help='predix for log folder (default:None)', default=None)
    parser.add_argument('-ds_path','--ds_path', help='path to dataset (default:/data/Niklas/kalman_vae_data)', default='/data/Niklas/kalman_vae_data')
    args = parser.parse_args()
    
    main(gpu= args.gpus,
         model_path=args.model, 
         start_epoch=args.start_epoch, 
         prefix=args.prefix, 
         ds_path=args.ds_path,
         log_folder='box',
         K = args.K,
         flow=args.flow)
    