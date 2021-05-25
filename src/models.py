import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa

from .layers import Encoder, Decoder
from .kalman_filter import KalmanFilter
from .losses import log_p_kalman

tfd = tfp.distributions
tfk = tf.keras
tfpl = tfp.layers

import numpy as np
import os
        
class VAE(tfk.Model):
    def __init__(self, 
                 config,
                 output_channels=1,
                 unet_decoder=False,
                 elbo_name='elbo = log p(y|x) + log p(x) - log q(x|y)',
                 name="vae",
                 debug = False,
                 **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.config = config
        self.debug = debug
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config, output_channels = output_channels, unet=unet_decoder)
        self.prior = tfd.Normal(loc=tf.zeros(config.dim_x, dtype='float32'),
                                scale = tf.ones(config.dim_x, dtype='float32'))

        self.log_py_x_metric = tfk.metrics.Mean(name = 'log p(y|x) ↑')
        self.log_px_metric = tfk.metrics.Mean(name = 'log p(x) ↑')
        self.log_qx_y_metric = tfk.metrics.Mean(name = 'log q(x|y) ↓')
        self.elbo_metric = tfk.metrics.Mean(name = elbo_name)

        self.ssim_metric = tfk.metrics.Mean(name = 'ssim ↑')

        self.loss_metric = tfk.metrics.Mean(name = 'loss')
        
        self.w_recon = self.config.scale_reconstruction
        self.w_kl = self.config.kl_latent_loss_weight
    
    def get_loss(self, py_x, y, qx_y, px, x, mask):
        logpx = tf.reduce_sum(tf.multiply(tf.reduce_sum(px.log_prob(x), axis=[2]), mask), axis=-1)
        logqx_y = tf.reduce_sum(tf.multiply(tf.reduce_sum(qx_y.log_prob(x), axis=[2]), mask), axis=-1)
        logpy_x = tf.reduce_sum(tf.multiply(tf.reduce_sum(py_x.log_prob(y), axis=[2,3]), mask), axis=-1)
                
        self.log_py_x_metric.update_state(logpy_x)
        self.log_px_metric.update_state(logpx)
        self.log_qx_y_metric.update_state(logqx_y)
        
        y_pred = py_x.sample()
        if self.debug:
            tf.debugging.assert_equal(y_pred.shape, y.shape)
        ssim = tf.image.ssim(y_pred, y, max_val=2.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        self.ssim_metric.update_state(ssim)
        
        return logpy_x, logpx, logqx_y
       

    def call(self, inputs):
        y = inputs[0]
        mask = inputs[1]
        p_y_x, q_x_y, x = self.forward(y, mask)
        
        logpy_x, logpx, logqx_y = self.get_loss(p_y_x, y, q_x_y, self.prior, x, tf.cast(mask == False, dtype='float32'))
        elbo = logpy_x + logpx - logqx_y
        self.elbo_metric.update_state(elbo)
        loss = -(self.w_recon * logpy_x + self.w_kl*(logpx - logqx_y))
        self.loss_metric.update_state(loss)
        self.add_loss(loss)
        
        metrices = {'log p(y|x)':tf.reduce_mean(logpy_x).numpy(), 
                    'log p(x)': tf.reduce_mean(logpx).numpy(), 
                    'log q(x|y)': tf.reduce_mean(logqx_y).numpy()
                   }
        
        return p_y_x, metrices
    
    def forward(self, y, mask):
        q_x_y = self.encoder(y)
        x = q_x_y.sample()
        p_y_x = self.decoder(x)
        if self.debug:
            tf.debugging.assert_equal(p_y_x.batch_shape, y.shape, "{0} vs {1}".format(p_y_x.batch_shape, y.shape))
            tf.debugging.assert_equal(q_x_y.batch_shape, (*y.shape[0:2], self.config.dim_x), "{0} vs {1}".format(q_x_y.batch_shape, (*y.shape[0:2], self.config.dim_x)))
            tf.debugging.assert_equal(x.shape, (*y.shape[0:2], self.config.dim_x), "{0} vs {1}".format(x.shape, (*y.shape[0:2], self.config.dim_x)))
        return p_y_x, q_x_y, x
    
    @tf.function
    def predict(self, inputs):
        y_true = inputs[0]
        mask = inputs[1]
        p_y_x, q_x_y, x = self.forward(y_true, mask)
        y_hat = p_y_x.sample()
        if self.debug:
            tf.debugging.assert_equal(y_true.shape, y_hat.shape, "{0} vs {1}".format(y_true.shape, y_hat.shape))
        return [{'name':'vae', 'data': y_hat}]
    
    def compile(self, num_batches):
        super(VAE, self).compile()
        self.num_batches = num_batches
        self.epoch = 1
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.config.init_lr, 
                                                                     decay_steps=self.config.decay_steps*num_batches, 
                                                                     decay_rate=self.config.decay_rate, 
                                                                     staircase=True)
        self.opt = tf.keras.optimizers.Adam(lr_schedule)  
    
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            _, metrices = self(inputs)
            loss = tf.reduce_mean(sum(self.losses))
            variables = self.trainable_variables
            if hasattr(self.config, 'only_vae_epochs'):
                if self.epoch <= self.config.only_vae_epochs:
                    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
                    if hasattr(self, 'u_encoder'):
                        variables += self.u_encoder.trainable_variables
                else:                               
                    variables = self.trainable_variables
        
        gradients = tape.gradient(loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
        self.opt.apply_gradients(zip(gradients, variables))
        metrices['loss'] = loss.numpy()
        return loss, metrices
    
    def test_step(self, inputs):
        _, metrices = self(inputs, training=False)
        loss = tf.reduce_mean(sum(self.losses))
        metrices['loss'] = loss.numpy()
        return loss, metrices 
    
    def model(self):
        inputs = tf.keras.layers.Input(shape=(self.config.dim_y))
        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))
    
    def info(self):
        x = tf.keras.layers.Input(shape=(self.config.ph_steps, *self.config.dim_y))        
        vae = tf.keras.Model(inputs=[x], outputs=self.call(x))
        vae.summary()
        
        encoder = tf.keras.Model(inputs=[x], outputs=self.encoder.call(x))
        encoder.summary()
        
        x_decoder = tf.keras.layers.Input(shape=(self.config.ph_steps, self.config.dim_x))
        decoder = tf.keras.Model(inputs=[x_decoder], outputs=self.decoder.call(x_decoder))
        decoder.summary()

class KVAE(VAE):
    def __init__(self, config, name="kvae", elbo_name='elbo = log p(y|x) - log q(x|y) + log p(x,z) - log p(z|x)', **kwargs):
        super(KVAE, self).__init__(name=name, elbo_name = elbo_name, config=config, **kwargs)
        self.log_pz_x_metric = tfk.metrics.Mean(name = 'log p(z|x) ↓')
        self.log_pxz_metric = tfk.metrics.Mean(name = 'log p(x,z) ↑')
        self.kf = KalmanFilter(self.config)
        self.w_kf = self.config.kf_loss_weight
    
    def get_loss(self, p_y_x, y, q_x_y, p_x, x, mask, x_smooth, p_zt_xT):
        logpy_x, logpx, logqx_y = super(KVAE, self).get_loss(p_y_x, y, q_x_y, p_x, x, mask)        
        log_pz_z, log_px_z, log_p0, log_pz_x = log_p_kalman(x_smooth, p_zt_xT, self.kf.kalman_filter)

        log_pz_z = tf.multiply(log_pz_z, mask[:,1:])
        log_px_z = tf.multiply(log_px_z, mask)
        log_p0 = tf.multiply(log_p0, mask[:,0])
        log_pz_x = tf.multiply(log_pz_x, mask)

        log_pz_x = tf.reduce_sum(log_pz_x, axis=-1)
        log_pxz = tf.reduce_sum(log_px_z, axis=-1) + log_p0 + tf.reduce_sum(log_pz_z, axis=-1)
        
        self.log_pz_x_metric.update_state(log_pz_x)
        self.log_pxz_metric.update_state(log_pxz)

        return logpy_x, logpx, logqx_y, log_pxz, log_pz_x
    
    def call(self, inputs):
        y_true = inputs[0]
        mask = inputs[1]
        p_y_x, q_x_y, x, x_smooth, p_zt_xT = self.forward(y_true, mask)
        logpy_x, logpx, logqx_y, log_pxz, log_pz_x = self.get_loss(p_y_x, y_true, q_x_y, self.prior, x, tf.cast(mask == False, dtype='float32'), x_smooth, p_zt_xT)
        
        elbo = logpy_x - logqx_y + log_pxz - log_pz_x
        loss = -(self.w_recon * logpy_x - self.w_kl*logqx_y + self.w_kf * (log_pxz - log_pz_x))
        
        self.log_py_x_metric.update_state(logpy_x)
        self.log_px_metric.update_state(logpx)
        self.log_qx_y_metric.update_state(logpy_x)        
        
        self.elbo_metric.update_state(elbo)
        self.loss_metric.update_state(loss)
        self.add_loss(loss)
        
        metrices = {'log p(y|x)': tf.reduce_mean(logpy_x).numpy(), 
                    'log q(x|y)': tf.reduce_mean(logqx_y).numpy(), 
                    'log p(x,z)': tf.reduce_mean(log_pxz).numpy(), 
                    'log p(z|x)': tf.reduce_mean(log_pz_x).numpy()
                   }

        return p_y_x, metrices
    
    def forward(self, y_true, mask):
        q_x_y = self.encoder(y_true)

        x = q_x_y.sample()
        x_kf = x
        
        p_zt_xT = self.kf([x_kf, mask])
        x_smooth = x_kf
        if self.config.sample_z:
            x_mu_smooth, x_cov_smooth = self.kf.kalman_filter.latents_to_observations(p_zt_xT.mean(), p_zt_xT.covariance())
            p_xt_xT = tfp.distributions.MultivariateNormalTriL(x_mu_smooth, tf.linalg.cholesky(x_cov_smooth))
            x_smooth = p_xt_xT.sample()
            '''
            C = self.kf.kalman_filter.get_observation_matrix_for_timestep
            Q = self.kf.kalman_filter.get_observation_noise_for_timestep

            x_smooth = list()
            for n in range(y_true.shape[1]):
                z = mu_smooth[:,n]
                z = tf.expand_dims(z,2)

                # Output for the current time step
                x_n = C(n).matmul(z) + Q(n).mean()[...,None]
                x_n = tf.squeeze(x_n,2)
                x_smooth.append(x_n)

            x_smooth = tf.stack(x_smooth, 1)
            '''
        p_y_x = self.decoder(x)
        return p_y_x, q_x_y, x, x_smooth, p_zt_xT

    @tf.function
    def predict(self, inputs):
        y_true = inputs[0]
        mask = inputs[1]
        q_x_y = self.encoder(y_true) 
        x = q_x_y.sample()
        
        #Smooth
        mu_smooth, Sigma_smooth = self.kf.kalman_filter.posterior_marginals(x, mask = mask)
        x_mu_smooth, x_cov_smooth = self.kf.kalman_filter.latents_to_observations(mu_smooth, Sigma_smooth)
        smooth_dist = tfp.distributions.MultivariateNormalTriL(loc=x_mu_smooth, scale_tril=tf.linalg.cholesky(x_cov_smooth))
        if self.debug:
            tf.debugging.assert_equal(self.config.dim_x, x_mu_smooth.shape[-1], "{0} vs {1}".format(self.config.dim_x, x_mu_smooth.shape[-1]))
            tf.debugging.assert_equal(x_cov_smooth.shape[-2], x_cov_smooth.shape[-1],"{0} vs {1}".format(x_cov_smooth.shape[-2],x_cov_smooth.shape[-1]))
            tf.debugging.assert_equal(self.config.dim_x, x_cov_smooth.shape[-1],"{0} vs {1}".format(self.config.dim_x, x_cov_smooth.shape[-1]))
        
        # Filter        
        kalman_data = self.kf.kalman_filter.forward_filter(x, mask=mask)
        _, mu_filt, Sigma_filt, mu_pred, Sigma_pred, x_mu_filt, x_covs_filt = kalman_data
        filt_dist = tfp.distributions.MultivariateNormalTriL(loc=x_mu_filt, scale_tril=tf.linalg.cholesky(x_covs_filt))
        if self.debug:
            tf.debugging.assert_equal(self.config.dim_x, x_mu_filt.shape[-1],"{0} vs {1}".format(self.config.dim_x, x_mu_filt.shape[-1]))
            tf.debugging.assert_equal(x_covs_filt.shape[-2], x_covs_filt.shape[-1],"{0} vs {1}".format(x_covs_filt.shape[-2], x_covs_filt.shape[-1]))
            tf.debugging.assert_equal(self.config.dim_x, x_covs_filt.shape[-1],"{0} vs {1}".format(self.config.dim_x, x_covs_filt.shape[-1]))        
         
        y_hat_filt = self.decoder(filt_dist.sample()).sample()
        y_hat_smooth = self.decoder(smooth_dist.sample()).sample()
        y_hat_vae = self.decoder(x).sample()
        
        return [{'name':'filt', 'data': y_hat_filt},
               {'name':'smooth', 'data': y_hat_smooth},
               {'name':'vae', 'data': y_hat_vae}]
    
    @tf.function
    def get_latents(self, inputs):
        y_true = inputs[0]
        mask = inputs[1]
        
        q_x_y = self.encoder(y_true)
        x = q_x_y.sample()
        
        # Smooth
        mu_smooth, Sigma_smooth = self.kf.kalman_filter.posterior_marginals(x, mask = mask)
        x_mu_smooth, x_cov_smooth = self.kf.kalman_filter.latents_to_observations(mu_smooth, Sigma_smooth)
        
        # Filter
        kalman_data = self.kf.kalman_filter.forward_filter(x, mask=mask)
        _, mu_filt, Sigma_filt, mu_pred, Sigma_pred, x_mu_filt, x_covs_filt = kalman_data  
        x_mu_filt_pred, x_covs_filt_pred = self.kf.kalman_filter.latents_to_observations(mu_pred, Sigma_pred)
        
        return x_mu_smooth, x_cov_smooth, x_mu_filt, x_covs_filt, x_mu_filt_pred, x_covs_filt_pred, x
    
    def sample(self, samples):
        x_samples = self.kf.kalman_filter.sample(sample_shape=samples)
        y_hat_sample, _,_ = self.decoder(x_samples)
        return y_hat_sample

                                  