from collections import namedtuple

def get_config(ds_path = '/data/Niklas/EchoNet-Dynamics', ds_size=None, dim_z=32, gpu='0', start_epoch=1, model_path=None):
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
        "init_cov": 5.0, #30.0
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
    return config