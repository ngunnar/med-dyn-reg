import os
import json
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np
import tensorflow as tf

def get_config(path):
    print('%s/%s' % (os.path.dirname(path), 'config.json'))
    print(os.path.dirname(path))
    with open('%s/%s' % (os.path.dirname(path), 'config.json')) as data_file:
        config_dict = json.load(data_file)
    config = namedtuple("Config", config_dict.keys())(*config_dict.values())
    return config, config_dict

def load_model(path, Model, file):
    config, config_dict = get_config(path)
    model = Model(config = config)
    model.load_weights(path + '/' + file).expect_partial()
    return model, config

def plot_latents(data, model, save_dir=None):
    latent_data = model.get_latents(data)
    x_mu_smooth, x_cov_smooth, x_mu_filt, x_covs_filt, x_mu_filt_pred, x_covs_filt_pred, x = latent_data
    
    std_smooth = tf.sqrt(tf.linalg.diag_part(x_cov_smooth[0,...]))
    std_filt = tf.sqrt(tf.linalg.diag_part(x_covs_filt[0,...]))
    std_pred = tf.sqrt(tf.linalg.diag_part(x_covs_filt_pred[0,...]))
    
    plt.rcParams.update({'font.size': 22})
    custom_ylim = (-0.2, 0.2)

    handles = {}
    t = np.arange(50)
    fig, axs = plt.subplots(2,8, figsize=(30,5), sharey=False,sharex=True)
    axs = axs.flatten()
    for i in range(len(axs)):
        mu_s = x_mu_smooth[0,:,i]
        mu_f = x_mu_filt[0,:,i]
        mu_p = x_mu_filt_pred[0,:,i]

        stf_s = std_smooth[:,i]
        stf_f = std_filt[:,i]
        stf_p = std_pred[:,i]
        
        axs[i].set_title('d={0}'.format(i))
        l1, = axs[i].plot(t, mu_s, 'r', label='Smooth')
        axs[i].fill_between(t, mu_s-stf_s, mu_s+stf_s, alpha=0.2, color='r')
        l2, = axs[i].plot(mu_f, 'g', label='Filt')
        axs[i].fill_between(t, mu_f-stf_f, mu_f+stf_f, alpha=0.2, color='g')    
        l3, = axs[i].plot(t, mu_p, 'y', label='Pred')
        axs[i].fill_between(t, mu_p-stf_p, mu_p+stf_p, alpha=0.2, color='y')


        l4, = axs[i].plot(x[0,:,i], 'b--', label='Obs')
        if 'Smooth' not in handles:
            handles['Smooth'] = l1
        if 'Filt' not in handles:
            handles['Filt'] = l2
        if 'Pred' not in handles:
            handles['Pred'] = l3
        if 'Obs' not in handles:
            handles['Obs'] = l4
        #axs[i].set_ylim(-0.3,0.3)
    plt.tight_layout()
    lgd = plt.legend(handles=handles.values(), loc='lower center', ncol=4, bbox_to_anchor=[-5.0, -0.8])
    if save_dir is not None:
        plt.savefig(save_dir, bbox_extra_artists=[lgd], bbox_inches='tight')
    else:
        plt.show()

def get_x(y_true, masks, model):
    x_vae = model.encoder(y_true).sample()
    data = []
    for mask in masks:
        latent_data = model.get_latents([y_true, mask])
        x_mu_smooth, x_cov_smooth, x_mu_filt, x_covs_filt, x_mu_filt_pred, x_covs_filt_pred, x = latent_data

        std_smooth = tf.sqrt(tf.linalg.diag_part(x_cov_smooth))
        std_filt = tf.sqrt(tf.linalg.diag_part(x_covs_filt))
        std_filt_pred = tf.sqrt(tf.linalg.diag_part(x_covs_filt_pred))
        data.append([x_mu_smooth, std_smooth, x_mu_filt, std_filt, x_mu_filt_pred, std_filt_pred])
    return x_vae, data        
        
def plot_latent(y_true, steps, last, model, y_range, dimension, save_dir=None):
    length = y_true.shape[1]
    mask_none = np.zeros(shape=length).astype('bool')
    mask_impute = np.ones(shape=length).astype('bool')
    mask_predict = np.ones(shape=length).astype('bool')
    known = np.arange(length)
    known_impute = known[::steps]
    known_predict = known[:-last]

    mask_impute[known_impute] = False
    mask_predict[known_predict] = False   
    
    x_vae, data = get_x(y_true, [mask_none, mask_impute, mask_predict], model)

    x_mu_smooth_none, std_smooth_none, x_mu_filt_none, std_filt_none, x_mu_filt_pred_none, std_filt_pred_none = data[0]
    x_mu_smooth_impute, std_smooth_impute, x_mu_filt_impute, std_filt_impute, x_mu_filt_pred_impute, std_filt_pred_impute = data[1]
    x_mu_smooth_predict, std_smooth_predict, x_mu_filt_predict, std_filt_predict, x_mu_filt_pred_predict, std_filt_pred_predict = data[2]

    #mpl.rcParams['text.usetex'] = True
    #mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

    d = dimension
    mu_s_none = x_mu_smooth_none[0,:,d]
    std_s_none = std_smooth_none[:,d]
    mu_f_none = x_mu_filt_none[0,:,d]
    std_f_none = std_filt_none[:,d]
    mu_p_none = x_mu_filt_pred_none[0,:,d]
    std_p_none = std_filt_pred_none[:,d]

    mu_s_impute = x_mu_smooth_impute[0,:,d]
    std_s_impute = std_smooth_impute[:,d]
    mu_f_impute = x_mu_filt_impute[0,:,d]
    std_f_impute = std_filt_impute[:,d]
    mu_p_impute = x_mu_filt_pred_impute[0,:,d]
    std_p_impute = std_filt_pred_impute[:,d]

    mu_s_pred = x_mu_smooth_predict[0,:,d]
    std_s_pred = std_smooth_predict[:,d]
    mu_f_pred = x_mu_filt_predict[0,:,d]
    std_f_pred = std_filt_predict[:,d]
    mu_p_pred = x_mu_filt_pred_predict[0,:,d]
    std_p_pred = std_filt_pred_predict[:,d]

    t = np.arange(50)

    fig, axs = plt.subplots(3,3, sharey=True, sharex=True, figsize=(10,5))
    for ax in axs:
        for a in ax:
            a.set_ylim(y_range[0],y_range[1])
            a.set_rasterization_zorder(1)
    handles = {}

    l1, = axs[0,0].plot(t, mu_s_none, 'r', label=r"x_t|T")
    l2, = axs[1,0].plot(t, mu_f_none, 'g', label=r"xt|t")
    l3, = axs[2,0].plot(t, mu_p_none, 'y', label=r"x_t+1|t")
    l4, = axs[0,0].plot(t, x_vae[0,:,d], 'b--', label=r"x")

    # None
    axs[0,0].set_title(r"$t = [1, \dots, T]$", fontdict={'fontsize': 20, 'fontweight': 'medium'})
    axs[0,0].fill_between(t, mu_s_none-std_s_none, mu_s_none+std_s_none, zorder=0, alpha=0.2, color='r')
    axs[1,0].fill_between(t, mu_f_none-std_f_none, mu_f_none+std_f_none, zorder=0, alpha=0.2, color='g')
    axs[1,0].plot(t, x_vae[0,:,d], 'b--')
    axs[2,0].fill_between(t, mu_p_none-std_p_none, mu_p_none+std_p_none, zorder=0, alpha=0.2, color='y')
    axs[2,0].plot(t, x_vae[0,:,d], 'b--')

    # Impute
    axs[0,1].set_title(r"$t = [1, {0}, {1}, \dots]$".format(1+steps, 1+2*steps), fontdict={'fontsize': 20, 'fontweight': 'medium'})
    axs[0,1].plot(t, mu_s_impute, 'r')
    axs[0,1].fill_between(t, mu_s_impute-std_s_impute, mu_s_impute+std_s_impute, zorder=0, alpha=0.2, color='r')
    axs[0,1].plot(t, x_vae[0,:,d], 'b--')

    axs[1,1].plot(t, mu_f_impute, 'g')
    axs[1,1].fill_between(t, mu_f_impute-std_f_impute, mu_f_impute+std_f_impute, zorder=0, alpha=0.2, color='g')
    axs[1,1].plot(t, x_vae[0,:,d], 'b--')

    axs[2,1].plot(t, mu_p_impute, 'y')
    axs[2,1].fill_between(t, mu_p_impute-std_p_impute, mu_p_impute+std_p_impute, zorder=0, alpha=0.2, color='y')
    axs[2,1].plot(t, x_vae[0,:,d], 'b--')

    # Predict
    axs[0,2].set_title(r"$t = [1, \dots, T-{0}]$".format(last), fontdict={'fontsize': 20, 'fontweight': 'medium'})
    axs[0,2].plot(t, mu_s_pred, 'r')
    axs[0,2].fill_between(t, mu_s_pred-std_s_pred, mu_s_pred+std_s_pred, zorder=0, alpha=0.2, color='r')
    axs[0,2].plot(t, x_vae[0,:,d], 'b--')

    axs[1,2].plot(t, mu_f_pred, 'g')
    axs[1,2].fill_between(t, mu_f_pred-std_f_pred, mu_f_pred+std_f_pred, zorder=0, alpha=0.2, color='g')
    axs[1,2].plot(t, x_vae[0,:,d], 'b--')

    axs[2,2].plot(t, mu_p_pred, 'y')
    axs[2,2].fill_between(t, mu_p_pred-std_p_pred, mu_p_pred+std_p_pred, zorder=0, alpha=0.2, color='y')
    axs[2,2].plot(t, x_vae[0,:,d], 'b--')

    if 'Smooth' not in handles:
         handles['Smooth'] = l1
    if 'Filt' not in handles:
        handles['Filt'] = l2
    if 'Pred' not in handles:
        handles['Pred'] = l3
    if 'Obs' not in handles:
        handles['Obs'] = l4

    plt.tight_layout()
    lgd = plt.legend(handles=handles.values(), loc='lower center', ncol=4, bbox_to_anchor=[-0.7, -1.5])
    if save_dir is not None:
        plt.savefig(save_dir, bbox_extra_artists=[lgd], bbox_inches='tight')
    else:
        plt.show()
        
def create_animation(y_true, steps, last, model, save_dir = None):
    length = y_true.shape[1]
    mask = np.ones(shape=length).astype('bool')
    known = np.arange(length)
    known = known[:-last][::steps]
    mask[known] = False    
    
    data = [y_true, mask[None,...]]
    result = model.predict(data)
    y_smooth_val = next(item for item in result if item["name"] == "smooth")['data']
    y_vae_val = next(item for item in result if item["name"] == "vae")['data']
    y_filt_val = next(item for item in result if item["name"] == "filt")['data']
    
    fig, axs = plt.subplots(1,4,figsize=(10,5))
    axs = axs.flatten()

    [ax.axis('off') for ax in axs]

    axs[1].set_title("Smooth")
    axs[2].set_title("Filter")
    axs[3].set_title("True data")
    
    def plot(ax, d, mask,i):
        if mask == True:
            cmap = 'gray'
        else:
            cmap = 'gray'

        if np.all(d[i,...] == 0.0):
            return ax.imshow(d[i,...]+1.0, cmap=cmap, vmin=-1, vmax=1)
        elif d.shape[0] > i:
            return ax.imshow(d[i,...], cmap=cmap,vmin=-1, vmax=1)
        else:
            return ax.imshow(np.zeros_like(d[0,...]), cmap=cmap,vmin=-1, vmax=1)
   
    ims = []
    k = 0
    for i in range(y_filt_val.shape[1]):
        if mask[i] == False:
            k = i
        im0 = plot(axs[0], y_true[0,...], False,k)
        im1 = plot(axs[1], y_smooth_val[0,...], mask[i],i)
        im2 = plot(axs[2], y_filt_val[0,...], mask[i],i)
        im3 = plot(axs[3], y_true[0,...], False,i)

        title = axs[0].text(0.5,1.05,"Input t={}".format(i), 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=axs[0].transAxes, )
        ims.append([im0, im1, im2, im3, title])

    anim = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
    if save_dir is not None:
        anim.save(save_dir, writer='imagemagick', fps=8, bitrate=900)
    else:
        return anim
    