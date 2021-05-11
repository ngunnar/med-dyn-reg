import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from .vae import Encoder, Decoder
from .kalman_filter import KalmanFilter
from .losses import loss_function

class KVAE(tf.keras.Model):
    def __init__(self, config, name="kvae", **kwargs):
        super(KVAE, self).__init__(name=name, **kwargs)
        self.config = config
        
        self.encoder = Encoder(self.config)
        self.kf = KalmanFilter(self.config)
        self.decoder = Decoder(self.config)

    def compile(self, num_batches):
        super(KVAE, self).compile()
        self.num_batches = num_batches
        self.epoch = 1
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.config.init_lr, 
                                                                     decay_steps=self.config.decay_steps*num_batches, 
                                                                     decay_rate=self.config.decay_rate, 
                                                                     staircase=True)
        self.opt = tf.keras.optimizers.Adam(lr_schedule) 
    
    def call(self, inputs):
        y_true = inputs[0]
        mask = inputs[1]
        x_vae, x_mu, x_logvar = self.encoder(y_true)
        
        mu_smooth, Sigma_smooth= self.kf([x_vae, mask])
        x_smooth = x_vae
        if self.config.sample_z:
            C = self.kf.kalman_filter.get_observation_matrix_for_timestep
            Q = self.kf.kalman_filter.get_observation_noise_for_timestep

            x_smooth = list()
            for n in range(y_true.shape[1]):
                z = mu_smooth[:,n]
                z = tf.expand_dims(z,2)

                # Output for the current time step
                x = C(n).matmul(z) + Q(n).mean()[...,None]
                x = tf.squeeze(x,2)
                x_smooth.append(x)

            x_smooth = tf.stack(x_smooth, 1)
        
        y_hat, y_mu, y_logvar = self.decoder(x_vae)
        
        return y_hat, y_mu, y_logvar, x_vae, x_smooth, x_mu, x_logvar, mu_smooth, Sigma_smooth
    
    def predict(self, inputs):
        y_true = inputs[0]
        mask = inputs[1]
        x_vae, x_mu, x_logvar = self.encoder(y_true)        
        _, mu_filt, Sigma_filt, mu_pred, Sigma_pred, x_mu_filt, x_covs_filt = self.kf.kalman_filter.forward_filter(x_vae, mask=mask)
        mu_smooth, Sigma_smooth = self.kf.kalman_filter.posterior_marginals(x_vae, mask = mask)
        
        mu_filt_x, Sigma_filt_x = self.kf.kalman_filter.latents_to_observations(mu_filt, Sigma_filt)
        filt_dist = tfp.distributions.MultivariateNormalFullCovariance(mu_filt_x, Sigma_filt_x)
        x_filt = filt_dist.sample()
        
        C = self.kf.kalman_filter.get_observation_matrix_for_timestep
        Q = self.kf.kalman_filter.get_observation_noise_for_timestep
        x_smooth = list()
        for n in range(y_true.shape[1]):
            z = mu_smooth[:,n]
            z = tf.expand_dims(z,2)
            
            # Output for the current time step
            x = C(n).matmul(z) + Q(n).sample()[...,None]
            x = tf.squeeze(x,2)
            x_smooth.append(x)
         
        x_smooth = tf.stack(x_smooth, 1)
                
        y_hat_filt, _, _ = self.decoder(x_filt)
        y_hat_smooth, _, _ = self.decoder(x_smooth)
        y_hat_vae, _, _ = self.decoder(x_vae)
        
        return y_hat_filt, y_hat_smooth, y_hat_vae
    
    def get_latents(self, inputs):
        y_true = inputs[0]
        mask = inputs[1]
        
        x_vae, x_mu, x_logvar = self.encoder(y_true)
        
        mu_smooth, Sigma_smooth = self.kf.kalman_filter.posterior_marginals(x_vae, mask = mask)
        _, z_mu_filt, z_cov_filt, z_mu_filt_pred, z_cov_filt_pred, x_mu_filt_t, x_covs_filt_t = self.kf.kalman_filter.forward_filter(x_vae, mask=mask)
    
        x_mu_smooth, x_cov_smooth = self.kf.kalman_filter.latents_to_observations(mu_smooth, Sigma_smooth)
        x_mu_filt_pred, x_covs_filt_pred = self.kf.kalman_filter.latents_to_observations(z_mu_filt_pred, z_cov_filt_pred)
        x_mu_filt, x_covs_filt = self.kf.kalman_filter.latents_to_observations(z_mu_filt, z_cov_filt)
        return x_mu_smooth, x_cov_smooth, x_mu_filt, x_covs_filt, x_mu_filt_pred, x_covs_filt_pred, x_vae
    
    def sample(self, samples):
        x_samples = self.kf.kalman_filter.sample(sample_shape=samples)
        y_hat_sample, _,_ = self.decoder(x_samples)
        return y_hat_sample


    def train_step(self, y_true, mask):
        with tf.GradientTape() as tape:
            w_recon = self.config.scale_reconstruction
            beta = tf.sigmoid((self.epoch%self.config.kl_cycle - 1)**2.0/self.config.kl_growth-self.config.kl_growth)
            w_kl = self.config.kl_latent_loss_weight * beta
            w_kf = self.config.kf_loss_weight
            y_hat, y_mu, y_logvar, x_vae, x_seq, x_mu, x_logvar, mu_smooth, Sigma_smooth = self([y_true, mask])
            loss_sum, recon_loss, kl_loss, kf_loss = loss_function(self.config, y_true, mask, y_hat, y_mu, y_logvar, 
                                                                   x_vae, x_mu, x_logvar,
                                                                   x_seq, mu_smooth, Sigma_smooth, self.kf.kalman_filter)
            if self.epoch <= self.config.only_vae_epochs:
                variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            else:                               
                variables = self.trainable_variables
            
            loss = recon_loss * w_recon
            loss += kl_loss * w_kl
            loss += kf_loss * w_kf
        gradients = tape.gradient(loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
        self.opt.apply_gradients(zip(gradients, variables))
        return loss_sum, loss, recon_loss, w_recon, kl_loss, w_kl, kf_loss, w_kf
    
    def test_step(self, y_true, mask):
        y_hat, y_mu, y_logvar, x_vae, x_seq, x_mu, x_logvar, mu_smooth, Sigma_smooth = self([y_true, mask], training=False)
        return loss_function(self.config, y_true, mask, y_hat, y_mu, y_logvar, 
                                  x_vae, x_mu, x_logvar,
                                  x_seq, mu_smooth, Sigma_smooth, 
                                  self.kf.kalman_filter)
