import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import tensorflow_addons as tfa

tfk = tf.keras

from .models import VAE, KVAE
from .losses import grad_loss

def warp(phi, y_0):
    bs, ph_steps, dim_y, _, channels = phi.shape
    y_0 = tf.repeat(y_0[:,None,...], ph_steps, axis=1)
    images = tf.reshape(y_0, (-1, *(dim_y,dim_y), 1))
    flows = tf.reshape(phi, (-1, *(dim_y,dim_y), 2))
    y_pred = tfa.image.dense_image_warp(images,
                                        flows)
    y_pred = tf.reshape(y_pred, (-1, ph_steps, *(dim_y,dim_y)))
    return y_pred

class FLOW_VAE(VAE):
    def __init__(self, config, name='flow_vae', output_channels = 2, **kwargs):
        super(FLOW_VAE, self).__init__(name=name, output_channels = output_channels, config=config, **kwargs)
        self.grad_flow_metric = tfk.metrics.Mean(name = 'grad flow ↓')
        
    def call(self, inputs):
        y = inputs[0]
        mask = inputs[1]
        p_y_x, phi_y_x, q_x_y, x = self.forward(y, mask)        
        logpy_x, logpx, logqx_y = self.get_loss(p_y_x, y, q_x_y, x, tf.cast(mask == False, dtype='float32'))
        elbo = logpy_x + logpx - logqx_y
        self.elbo_metric.update_state(elbo)
        loss = -(self.w_recon * logpy_x + self.w_kl*(logpx - logqx_y))
        self.loss_metric.update_state(loss)
        self.add_loss(loss)

        grad = grad_loss('l2', tf.reshape(phi_y_x.mean(), (y.shape[0]*y.shape[1], y.shape[2], y.shape[3], 2)))
        self.grad_flow_metric.update_state(grad)
        self.add_loss(grad)

        return p_y_x

    def forward(self, y_true, mask):
        y_0 = y_true[:,0,...]
        q_x_y = self.encoder(y_true)
        x = q_x_y.sample()
        phi_y_x = self.decoder(x)
        if self.debug:
            tf.debugging.assert_equal(phi_y_x.batch_shape, (*y_true.shape, 2), "{0} vs {1}".format(phi_y_x.batch_shape, (*y_true.shape, 2)))
            tf.debugging.assert_equal(q_x_y.batch_shape, (*y_true.shape[0:2], self.config.dim_x), "{0} vs {1}".format(q_x_y.batch_shape, (*y_true.shape[0:2], self.config.dim_x)))
            tf.debugging.assert_equal(x.shape, (*y_true.shape[0:2], self.config.dim_x), "{0} vs {1}".format(x.shape, (*y_true.shape[0:2], self.config.dim_x)))

        phi_mu = phi_y_x.mean() #bs, t, w, h, 2
        y_mu = warp(phi_mu, y_0)
        y_sigma = tf.ones_like(y_mu, dtype='float32') * 0.01
        p_y_x = tfp.distributions.Normal(loc=y_mu, scale=y_sigma)
        if self.debug:
            tf.debugging.assert_equal(p_y_x.batch_shape, y_true.shape, "{0} vs {1}".format(p_y_x.batch_shape, y_true.shape))
        return p_y_x, phi_y_x, q_x_y, x
    
    def predict(self, inputs):
        y_true = inputs[0]
        mask = inputs[1]
        p_y_x, phi_y_x, q_x_y, x = self.forward(y_true, mask)
        phi_hat = phi_y_x.sample()
        y_hat = p_y_x.sample()
        if self.debug:
            tf.debugging.assert_equal(y_true.shape, y_hat.shape, "{0} vs {1}".format(y_true.shape, y_hat.shape))
            tf.debugging.assert_equal((*y_true.shape, 2), phi_hat.shape, "{0} vs {1}".format((*y_true.shape, 2), phi_hat.shape))
        return [{'name':'vae', 'data': y_hat},
               {'name':'flow', 'data': phi_hat}]         


class FLOW_KVAE(KVAE):
    def __init__(self, config, name="flow_kvae", output_channels=2, **kwargs):
        super(FLOW_KVAE, self).__init__(name=name, output_channels = output_channels, config=config, **kwargs)
        self.grad_flow_metric = tfk.metrics.Mean(name = 'grad flow ↓')
    
    def call(self, inputs):
        y_true = inputs[0]
        mask = inputs[1]
        p_y_x, phi_y_x, q_x_y, x, x_smooth, mu_smooth, Sigma_smooth = self.forward(y_true, mask)
        
        logpy_x, logpx, logqx_y, log_pxz, log_pz_x = self.get_loss(p_y_x, y_true, q_x_y, x, tf.cast(mask == False, dtype='float32'), x_smooth, mu_smooth, Sigma_smooth)
        
        elbo = logpy_x + logpx - logqx_y + log_pxz - log_pz_x
        loss = -(self.w_recon * logpy_x + self.w_kl*(logpx - logqx_y) + self.w_kf * (log_pxz - log_pz_x))
        
        self.log_py_x_metric.update_state(logpy_x)
        self.log_px_metric.update_state(logpx)
        self.log_qx_y_metric.update_state(logpy_x)        
        
        self.elbo_metric.update_state(elbo)
        self.loss_metric.update_state(loss)
        self.add_loss(loss)

        grad = grad_loss('l2', tf.reshape(phi_y_x.mean(), (y_true.shape[0]*y_true.shape[1], y_true.shape[2], y_true.shape[3], 2)))
        self.grad_flow_metric.update_state(grad)
        self.add_loss(grad)

        return p_y_x
    
    def forward(self, y_true, mask):
        y_0 = y_true[:,0,...]
        phi_y_x, q_x_y, x, x_smooth, mu_smooth, Sigma_smooth =  super().forward(y_true, mask)
        phi_sample = phi_y_x.mean() #bs, t, w, h, 2
        y_mu = warp(phi_sample, y_0)
        y_sigma = tf.ones_like(y_mu, dtype='float32') * 0.01
        p_y_x = tfp.distributions.Normal(loc=y_mu, scale=y_sigma)

        return p_y_x, phi_y_x, q_x_y, x, x_smooth, mu_smooth, Sigma_smooth

    def predict(self, inputs):
        y_0 = inputs[0][:,0,...]
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
         
        phi_hat_filt = self.decoder(filt_dist.sample()).sample()
        phi_hat_smooth = self.decoder(smooth_dist.sample()).sample()
        phi_hat_vae = self.decoder(x).sample()  
        
        y_hat_filt = warp(phi_hat_filt, y_0)
        y_hat_smooth = warp(phi_hat_smooth, y_0)
        y_hat_vae = warp(phi_hat_vae, y_0)

        return [{'name':'filt', 'data': y_hat_filt},
                {'name':'filt_flow', 'data': phi_hat_filt},
                {'name':'smooth', 'data': y_hat_smooth},
                {'name':'smooth_flow', 'data': phi_hat_smooth},
                {'name':'vae', 'data': y_hat_vae},
                {'name':'vae_flow', 'data': phi_hat_vae}]
    
    def sample(self, samples):
        x_samples = self.kf.kalman_filter.sample(sample_shape=samples)
        return self.decoder(x_samples).sample()