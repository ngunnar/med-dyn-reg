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
        y_0 = inputs[2]
        p_y_x, phi_y_x, q_x_y, x = self.forward(y, mask, y_0)        
        logpy_x, logpx, logqx_y = self.get_loss(p_y_x, y, q_x_y, self.prior, x, tf.cast(mask == False, dtype='float32'))
        elbo = logpy_x + logpx - logqx_y
        self.elbo_metric.update_state(elbo)
        loss = -(self.w_recon * logpy_x + self.w_kl*(logpx - logqx_y))
        self.loss_metric.update_state(loss)
        self.add_loss(loss)

        grad = grad_loss('l2', tf.reshape(phi_y_x.mean(), (y.shape[0]*y.shape[1], y.shape[2], y.shape[3], 2)))
        self.grad_flow_metric.update_state(grad)
        self.add_loss(grad)
        
        metrices = {'log p(y|x)': tf.reduce_mean(logpy_x).numpy(), 
                    'log p(x)': tf.reduce_mean(logpx).numpy(), 
                    'log q(x|y)': tf.reduce_mean(logqx_y).numpy()
                   }
        return p_y_x, metrices

    def forward(self, y, mask, y_0):
        q_x_y = self.encoder(y)
        x = q_x_y.sample()
        phi_y_x = self.decoder(x)
        if self.debug:
            tf.debugging.assert_equal(phi_y_x.batch_shape, (*y.shape, 2), "{0} vs {1}".format(phi_y_x.batch_shape, (*y.shape, 2)))
            tf.debugging.assert_equal(q_x_y.batch_shape, (*y.shape[0:2], self.config.dim_x), "{0} vs {1}".format(q_x_y.batch_shape, (*y.shape[0:2], self.config.dim_x)))
            tf.debugging.assert_equal(x.shape, (*y.shape[0:2], self.config.dim_x), "{0} vs {1}".format(x.shape, (*y.shape[0:2], self.config.dim_x)))

        phi_mu = phi_y_x.mean() #bs, t, w, h, 2
        y_mu = warp(phi_mu, y_0)
        y_sigma = tf.ones_like(y_mu, dtype='float32') * 0.01
        p_y_x = tfp.distributions.Normal(loc=y_mu, scale=y_sigma)
        if self.debug:
            tf.debugging.assert_equal(p_y_x.batch_shape, y.shape, "{0} vs {1}".format(p_y_x.batch_shape, y.shape))
        return p_y_x, phi_y_x, q_x_y, x
    
    #@tf.function
    def predict(self, inputs):
        y = inputs[0]
        mask = inputs[1]
        y_0 = inputs[2]
        p_y_x, phi_y_x, q_x_y, x = self.forward(y, mask, y_0)
        phi_hat = phi_y_x.sample()
        y_hat = p_y_x.sample()
        if self.debug:
            tf.debugging.assert_equal(y.shape, y_hat.shape, "{0} vs {1}".format(y.shape, y_hat.shape))
            tf.debugging.assert_equal((*y.shape, 2), phi_hat.shape, "{0} vs {1}".format((*y.shape, 2), phi_hat.shape))
        return [{'name':'vae', 'data': y_hat},
               {'name':'flow', 'data': phi_hat}]         


class FLOW_KVAE(KVAE):
    def __init__(self, config, name="flow_kvae", output_channels=2, **kwargs):
        super(FLOW_KVAE, self).__init__(name=name, output_channels = output_channels, config=config, **kwargs)
        self.grad_flow_metric = tfk.metrics.Mean(name = 'grad flow ↓')
    
    def call(self, inputs):
        y = inputs[0]
        mask = inputs[1]
        y_0 = inputs[2]
        p_y_x, phi_y_x, q_x_y, x, x_smooth, p_zt_xT = self.forward(y, mask, y_0)
        
        logpy_x, logpx, logqx_y, log_pxz, log_pz_x = self.get_loss(p_y_x, y, q_x_y, self.prior, x, tf.cast(mask == False, dtype='float32'), x_smooth, p_zt_xT)
        
        elbo = logpy_x - logqx_y + log_pxz - log_pz_x
        loss = -(self.w_recon * logpy_x - self.w_kl* - logqx_y + self.w_kf * (log_pxz - log_pz_x))
        
        self.log_py_x_metric.update_state(logpy_x)
        self.log_px_metric.update_state(logpx)
        self.log_qx_y_metric.update_state(logpy_x)        
        
        self.elbo_metric.update_state(elbo)
        self.loss_metric.update_state(loss)
        self.add_loss(loss)

        grad = grad_loss('l2', tf.reshape(phi_y_x.mean(), (y.shape[0]*y.shape[1], y.shape[2], y.shape[3], 2)))
        self.grad_flow_metric.update_state(grad)
        self.add_loss(grad)
        
        metrices = {'log p(y|x)': tf.reduce_mean(logpy_x).numpy(), 
                    'log q(x|y)': tf.reduce_mean(logqx_y).numpy(), 
                    'log p(x,z)': tf.reduce_mean(log_pxz).numpy(), 
                    'log p(z|x)': tf.reduce_mean(log_pz_x).numpy()
                   }

        return p_y_x, metrices
    
    def forward(self, y, mask, y_0):
        phi_y_x, q_x_y, x, x_smooth, p_zt_xT =  super().forward(y, mask)
        phi_sample = phi_y_x.mean() #bs, t, w, h, 2
        y_mu = warp(phi_sample, y_0)
        y_sigma = tf.ones_like(y_mu, dtype='float32') * 0.01
        p_y_x = tfp.distributions.Normal(loc=y_mu, scale=y_sigma)

        return p_y_x, phi_y_x, q_x_y, x, x_smooth, p_zt_xT

    #@tf.function
    def predict(self, inputs):
        y_0 = inputs[2]#[:,0,...]
        y = inputs[0]
        mask = inputs[1]
        q_x_y = self.encoder(y) 
        x = q_x_y.sample()
        
        #Smooth
        smooth_dist = self.kf.get_smooth_dist(x, mask)
        
        # Filter        
        filt_dist = self.kf.get_filter_dist(x, mask)
         
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



from .layers import Encoder
class UFLOW_KVAE(FLOW_KVAE):
    def __init__(self, config, name='UFLOW_KVAE', **kwargs):
        super(UFLOW_KVAE, self).__init__(name=name, config=config, unet_decoder = True, **kwargs)
        self.u_encoder = Encoder(self.config, unet=True)
    
    def forward(self, y, mask, y_0):
        # Encoder
        q_x_y = self.encoder(y)
        p_x_y0, feats = self.u_encoder(y_0[:,None,:,:])
        
        x_0 = p_x_y0.sample()

        # Kalman
        x = q_x_y.sample()
        x_kf = x
        
        p_zt_xT = self.kf([x_kf, mask])
        x_smooth = x_kf
        
        # Decoder
        x_in = tf.concat([x, tf.repeat(x_0, x.shape[1], axis=1)], axis=-1)
        phi_y_x = self.decoder([x_in, feats])
        phi = phi_y_x.mean()
        y_mu = warp(phi, y_0)
        y_sigma = tf.ones_like(y_mu, dtype='float32') * 0.01
        p_y_x = tfp.distributions.Normal(loc=y_mu, scale=y_sigma)
        return p_y_x, phi_y_x, q_x_y, p_x_y0, x, x_smooth, p_zt_xT
    
    def call(self, inputs):
        y = inputs[0]
        mask = inputs[1]
        y_0 = inputs[2]
        p_y_x, phi_y_x, q_x_y, p_x_y0, x, x_smooth, p_zt_xT = self.forward(y, mask, y_0)
        logpy_x, logpx, logqx_y, log_pxz, log_pz_x = self.get_loss(p_y_x, 
                                                                   y, 
                                                                   q_x_y, 
                                                                   p_x_y0, 
                                                                   x, 
                                                                   tf.cast(mask == False, dtype='float32'), 
                                                                   x_smooth, 
                                                                   p_zt_xT)
        
        elbo = logpy_x + logpx - logqx_y + log_pxz - log_pz_x
        loss = -(self.w_recon * logpy_x + self.w_kl*(logpx - logqx_y) + self.w_kf * (log_pxz - log_pz_x))
        
        self.log_py_x_metric.update_state(logpy_x)
        self.log_px_metric.update_state(logpx)
        self.log_qx_y_metric.update_state(logpy_x)        
        
        self.elbo_metric.update_state(elbo)
        self.loss_metric.update_state(loss)
        self.add_loss(loss)

        grad = grad_loss('l2', tf.reshape(phi_y_x.mean(), (y.shape[0]*y.shape[1], y.shape[2], y.shape[3], 2)))
        self.grad_flow_metric.update_state(grad)
        self.add_loss(grad)
        
        metrices = {'log p(y|x)': tf.reduce_mean(logpy_x).numpy(), 
                    'log p(x)': tf.reduce_mean(logpx).numpy(),
                    'log q(x|y)': tf.reduce_mean(logqx_y).numpy(), 
                    'log p(x,z)': tf.reduce_mean(log_pxz).numpy(), 
                    'log p(z|x)': tf.reduce_mean(log_pz_x).numpy()
                   }

        return p_y_x, metrices
    
    #@tf.function
    def predict(self, inputs):
        y_0 = inputs[2]#[:,0,...]
        y = inputs[0]
        mask = inputs[1]
        q_x_y = self.encoder(y) 
        p_x_y0, feats = self.u_encoder(y_0[:,None,:,:])
        x_0 = p_x_y0.sample()
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
        
        x_filt = filt_dist.sample()
        X_0 = tf.repeat(x_0, x_filt.shape[1], axis=1)
        x_in = tf.concat([x_filt, X_0], axis=-1)
        phi_hat_filt = self.decoder([x_in, feats]).sample()
        x_smooth = smooth_dist.sample()
        x_in = tf.concat([x_smooth, X_0], axis=-1)
        phi_hat_smooth = self.decoder([x_in, feats]).sample()
        x_in = tf.concat([x, X_0], axis=-1)
        phi_hat_vae = self.decoder([x_in, feats]).sample()  
        
        y_hat_filt = warp(phi_hat_filt, y_0)
        y_hat_smooth = warp(phi_hat_smooth, y_0)
        y_hat_vae = warp(phi_hat_vae, y_0)

        return [{'name':'filt', 'data': y_hat_filt},
                {'name':'filt_flow', 'data': phi_hat_filt},
                {'name':'smooth', 'data': y_hat_smooth},
                {'name':'smooth_flow', 'data': phi_hat_smooth},
                {'name':'vae', 'data': y_hat_vae},
                {'name':'vae_flow', 'data': phi_hat_vae}]