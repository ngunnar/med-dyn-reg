from tensorflow.keras import backend as K
import tensorflow as tf
tfk = tf.keras

from .utils import plot_to_image, latent_plot

class SetLossWeightsCallback(tfk.callbacks.Callback):
    def __init__(self, kl_growth):
        self.kl_growth = kl_growth

    def on_epoch_begin(self, epoch, logs=None):
        new_beta_value = tf.sigmoid((epoch-1)**2/self.kl_growth-self.kl_growth)            
        K.set_value(self.model.w_kf, K.get_value(self.model.init_w_kf)*new_beta_value)
        K.set_value(self.model.w_kl, K.get_value(self.model.init_w_kl)*new_beta_value)


import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
import math

def plot(data, title, max_i, axs):
    for k in range(0, max_i):
        if k==0:
            #axs[k].title.set_text(title)
            axs[k].set_ylabel(title)
        if data.shape[-1] == 2:
            axs[k].imshow(draw_hsv(data[k,...]))
        else:           
            axs[k].imshow(data[k,...], cmap='gray')

def draw_hsv(flow):
    h, w, c = flow.shape
    fx, fy = flow[...,0], flow[...,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def plot_to_image(y, result):    
    step = 5
    s = 0
    l_org = y.shape[1] // step
    l = l_org

    rows = np.sum([len(v.keys()) for k, v in result.items()])
    figure, axs = plt.subplots(rows+1,l, sharex=True, sharey=True, figsize=(3.2*l, (len(result)+1)*2.4))
    axs = axs.flatten()
    [ax.axes.xaxis.set_ticks([]) for ax in axs]
    [ax.axes.yaxis.set_ticks([]) for ax in axs]
    
    if y is not None:
        plot(y[0,::step,...], 'True image', l_org, axs[s:l])
        s = l
        l = l+l_org
    
    for k, v in result.items(): #{'type': {'images': data, 'flows': data}}
        for k1, v1 in v.items():
            #plot(v[0,::step,...].numpy(), k.numpy().decode("utf-8"), l_org, axs[s:l])
            plot(v1[0,::step,...].numpy(), ' '.join([k, k1]), l_org, axs[s:l])
            s = l
            l = l+l_org
    
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def latent_plot(latent_dist, x_obs):
    x_vae = x_obs
    
    y_max = np.max(x_vae, axis=1)
    y_min = np.min(x_vae, axis=1)
    y_diff = y_max - y_min
       
    handles = {}
    
    if 'smooth' not in latent_dist:
        dims = x_vae.shape[2]
        t = np.arange(x_vae.shape[1])
        figure, axs = plt.subplots(1,dims, figsize=(6.4*dims, 4.8))
        for i in range(dims):
            axs[i].plot(t, x_vae[0,:,i],'--', label='x')
    else:
        x_mu_smooth = latent_dist['smooth'].mean()
        x_std_smooth = latent_dist['smooth'].stddev()

        x_mu_filt = latent_dist['filt'].mean()
        x_std_filt = latent_dist['filt'].stddev()

        x_mu_pred = latent_dist['pred'].mean()
        x_std_pred = latent_dist['pred'].stddev()
        
        t = np.arange(x_vae.shape[1])
        dims = x_vae.shape[2]

        k,l = math.ceil(dims/8), min(dims,8)
        figure, axs = plt.subplots(k,l, figsize=(5*l,5*k), sharey=False,sharex=True)
                
        #figure, axs = plt.subplots(2,dims//2, figsize=(6.4*dims, 4.8))
        axs = axs.flatten()
        for i in range(dims):
            mu_s = x_mu_smooth[0,:,i]
            stf_s = x_std_smooth[0,:,i]
            mu_f = x_mu_filt[0,:,i]
            stf_f = x_std_filt[0,:,i]
            mu_p = x_mu_pred[0,:,i]
            stf_p = x_std_pred[0,:,i]
            
            axs[i].plot(t, x_vae[0,:,i],'--', label='x')                          
            
            axs[i].plot(t, mu_s, 'r', label='x(t|T)')
            axs[i].fill_between(t, mu_s-stf_s, mu_s+stf_s, alpha=0.2, color='r')
              
            axs[i].plot(t, mu_f, 'g', label='x(t|t)')
            axs[i].fill_between(t, mu_f-stf_f, mu_f+stf_f, alpha=0.2, color='g')
            
            axs[i].plot(t, mu_p, 'y', label='x(t+1|t)')
            axs[i].fill_between(t, mu_p-stf_p, mu_p+stf_p, alpha=0.2, color='y')                
            
            axs[i].set_ylim([y_min[0,i] - y_diff[0,i]*0.2, y_max[0,i] + y_diff[0,i]*0.2])
            axs[i].legend(loc="upper left", ncol=1)
                    
    
    plt.tight_layout()
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

class VisualizeResultCallback(tfk.callbacks.Callback):
    def __init__(self, img_writer, train_data, test_data = None, log_interval = 5):
        self.img_writer = img_writer
        self.train_data = train_data
        self.test_data = test_data
        self.log_interval = log_interval   


    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_interval == 0:
            train_result = self.model.eval(self.train_data)
            img_figure_train = plot_to_image(self.train_data['input_video'], train_result['image_data'])
            
            if self.test_data is not None:
                test_result = self.model.eval(self.test_data)
                img_figure_test = plot_to_image(self.test_data['input_video'], test_result['image_data'])
            
            if 'latent_dist' in train_result.keys():
                train_latent_figure = latent_plot(train_result['latent_dist'], train_result['x_obs'])
                if self.test_data is not None:
                    test_latent_figure = latent_plot(test_result['latent_dist'], test_result['x_obs'])
            
            with self.img_writer.as_default():
                tf.summary.image("Images train", img_figure_train, step=epoch)
                if self.test_data is not None:
                    tf.summary.image("Images test", img_figure_test, step=epoch)
                if 'latent_dist' in train_result.keys():
                    tf.summary.image("Latents train", train_latent_figure, step=epoch)
                    if self.test_data is not None:
                        tf.summary.image("Latents test", test_latent_figure, step=epoch)

        