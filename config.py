from collections import namedtuple
import numpy as np
def get_config(
    #DS
    dim_y = (112,112),
    ph_steps=50,    
    period= 1,
    ds_path= '/data/Niklas/EchoNet-Dynamics',
    ds_size= None,
    #VAE
    activation = 'relu',
    filter_size = 3,
    enc_filters = [64, 128, 256, 512],
    dec_filters = [64, 128, 256, 512],
    use_subpixel = [True, True, True, True], # True - more parameters in decoder, especially in layers with many filters
    # LGSSM
    dim_x = 16,
    dim_z = 32,
    noise_emission =  0.03,
    noise_transition =  0.08,
    init_cov = 1.0, #30.0
    trainable_A = True,
    trainable_C = True,
    trainable_R = True,
    trainable_Q = True,
    trainable_mu = True,
    trainable_sigma = True,
    sigma_full = False,
    sample_z = False,
    K = 1,
    # Training
    gpu = '0',
    num_epochs = 100,
    start_epoch = 1,
    model_path = None,
    batch_size = 4,
    init_lr = 1e-4,
    decay_steps = 20,
    decay_rate = 0.85,
    max_grad_norm = 150.0,
    scale_reconstruction = 1.0,#1e-4,
    kl_latent_loss_weight = 1.0,
    kf_loss_weight = 1.0,
    kl_growth = 3.0,
    kl_cycle = 20,
    only_vae_epochs = 5,
    # Plotting
    plot_epoch = 1,
):
    
    config_dict = {
        # DS
        "dim_y":dim_y,
        "ph_steps":ph_steps,    
        "period": period,
        "ds_path": ds_path,
        "ds_size": ds_size,
        #VAE
        'activation': activation,
        'filter_size': filter_size,
        'enc_filters':enc_filters,
        'dec_filters':dec_filters,
        'use_subpixel':use_subpixel,
        # LGSSM
        "dim_x": dim_x,
        "dim_z": dim_z,
        "noise_emission": noise_emission,
        "noise_transition": noise_transition,
        "init_cov": init_cov, #30.0
        "trainable_A":trainable_A,
        "trainable_C":trainable_C,
        "trainable_R":trainable_R,
        "trainable_Q":trainable_Q,
        "trainable_mu":trainable_mu,
        "trainable_sigma":trainable_sigma,
        "sigma_full":sigma_full,
        "sample_z": sample_z,
        "K": K,
        # Training
        "gpu": gpu,
        "num_epochs": num_epochs,
        "start_epoch": start_epoch,
        "model_path": model_path,
        "batch_size": batch_size,
        "init_lr": init_lr,
        "decay_steps": decay_steps,
        "decay_rate": decay_rate,
        "max_grad_norm": max_grad_norm,
        "scale_reconstruction": scale_reconstruction,#1e-4,
        "kl_latent_loss_weight": kl_latent_loss_weight,
        "kf_loss_weight": kf_loss_weight,
        "kl_growth": kl_growth,
        "kl_cycle":kl_cycle,
        "only_vae_epochs": only_vae_epochs,
        # Plotting
        "plot_epoch": plot_epoch,
        }
    config = namedtuple("Config", config_dict.keys())(*config_dict.values())
    return config, config_dict