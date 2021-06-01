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

class fVAE(VAE):
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
    
    @tf.function
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


class fKVAE(KVAE):
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

    @tf.function
    def predict(self, inputs):
        y_0 = inputs[2]#[:,0,...]
        y = inputs[0]
        mask = inputs[1]
        q_x_y = self.encoder(y) 
        x = q_x_y.sample()
        
        #Smooth
        smooth_dist = self.kf.get_smooth_dist(x, mask)
        
        # Filter        
        filt_dist, filt_pred_dist = self.kf.get_filter_dist(x, mask, True)
         
        phi_hat_filt = self.decoder(filt_dist.sample()).sample()
        phi_hat_pred = self.decoder(filt_pred_dist.sample()).sample()
        phi_hat_smooth = self.decoder(smooth_dist.sample()).sample()
        phi_hat_vae = self.decoder(x).sample()  
        
        y_hat_filt = warp(phi_hat_filt, y_0)
        y_hat_pred = warp(phi_hat_pred, y_0)
        y_hat_smooth = warp(phi_hat_smooth, y_0)
        y_hat_vae = warp(phi_hat_vae, y_0)

        return [{'name':'filt', 'data': y_hat_filt},
                {'name':'filt_flow', 'data': phi_hat_filt},
                {'name':'pred', 'data': y_hat_pred},
                {'name':'pred_flow', 'data': phi_hat_pred},
                {'name':'smooth', 'data': y_hat_smooth},
                {'name':'smooth_flow', 'data': phi_hat_smooth},
                {'name':'vae', 'data': y_hat_vae},
                {'name':'vae_flow', 'data': phi_hat_vae}]
    
    def sample(self, samples):
        x_samples = self.kf.kalman_filter.sample(sample_shape=samples)
        return self.decoder(x_samples).sample()

class Bspline(tf.keras.layers.Layer):
    def __init__(self, dim_y, dim_x, **kwargs):
        super(Bspline,self).__init__(**kwargs)
        y_range = tf.range(start=0, limit=dim_y[0])/dim_y[0]
        x_range = tf.range(start=0, limit=dim_y[1])/dim_y[1]
        y_grid, x_grid = tf.meshgrid(y_range, x_range, indexing="ij")
        self.grid = tf.cast(tf.stack((y_grid, x_grid), -1), 'float32')
        self.dim_x = dim_x   
    
    def body(i, d, p, s):
        d_new = tfa.image.interpolate_bilinear(p[...,i:i+1], s)
        return (i+1, tf.concat([d,d_new], axis=-1), p, s)
    
    def call(self, inputs):
        image = inputs[0] # (bs, weigth, heigth, 1)
        bs = tf.shape(image)[0]
        parameters = inputs[1] /10 #(bs, t, w, h, 2)
        ph_steps = parameters.shape[1]
        
        org_image = tf.repeat(image[:,None,...], ph_steps, axis=1)
        image = tf.reshape(org_image, (-1, org_image.shape[2],org_image.shape[3], 1))
        parameters = tf.reshape(parameters, (bs*ph_steps, np.sqrt(self.dim_x/2), np.sqrt(self.dim_x/2), 2))
                 
        transform_grid = tf.repeat(self.grid[None,...], bs*ph_steps, axis=0)
        transform_grid = tf.reshape(transform_grid, (bs*ph_steps, -1, self.grid.shape[-1]))
        
        scaled_points = transform_grid * (np.array(parameters.shape[1:-1], dtype='float32') - 1)[None,None,...]
        
        cond = lambda i, d, p, s: i < p.shape[-1]
        i0=tf.constant(1)
        tmp = tfa.image.interpolate_bilinear(parameters[...,0:1], scaled_points)
        _, d,_,_ = tf.while_loop(cond, 
                          Bspline.body, 
                          loop_vars=[i0, 
                                     tfa.image.interpolate_bilinear(parameters[...,0:1], scaled_points), 
                                     parameters,
                                     scaled_points],
                          shape_invariants=[i0.get_shape(), 
                                            tf.TensorShape([None, image.shape[1]*image.shape[2], None]), #bs 
                                            parameters.get_shape(), 
                                            scaled_points.get_shape()])        
        transform_grid = (transform_grid + d)
        transform_grid = tf.reshape(transform_grid, (bs*ph_steps, *self.grid.shape))
        
        scaled_grid = transform_grid * np.asarray(image.shape[1:-1])[None, None,...]
        scaled_grid = tf.reshape(scaled_grid, (bs*ph_steps, -1, 2))
        sample = tfa.image.interpolate_bilinear(image, scaled_grid)
        
        mu = tf.reshape(sample, (org_image.shape))
        sigma = tf.ones_like(mu, dtype='float32') * 0.01
        p_y_x = tfp.distributions.Normal(mu, sigma)
        
        return p_y_x
    
class bKVAE(FLOW_KVAE):
    def __init__(self, config, name='spline_kvae', **kwargs):
        super(FLOW_KVAE, self).__init__(name=name, config=config, **kwargs)
        assert config.dim_x == 32, "dim x 32 is the only supported atm"
        self.decoder = Bspline(config.dim_y, config.dim_x)
    
    def call(self, inputs):
        y = inputs[0]
        mask = inputs[1]
        y_0 = inputs[2]
        
        q_x_y = self.encoder(y)

        x = q_x_y.sample()
        x_kf = x
        
        p_zt_xT = self.kf([x_kf, mask])
        x_smooth = x_kf
        if self.config.sample_z:
            x_mu_smooth, x_cov_smooth = self.kf.kalman_filter.latents_to_observations(p_zt_xT.mean(), p_zt_xT.covariance())
            p_xt_xT = tfp.distributions.MultivariateNormalTriL(x_mu_smooth, tf.linalg.cholesky(x_cov_smooth))
            x_smooth = p_xt_xT.sample()
        p_y_x = self.decoder([y_0, x])
        
        logpy_x, logpx, logqx_y, log_pxz, log_pz_x = self.get_loss(p_y_x, y, q_x_y, self.prior, x, tf.cast(mask == False, dtype='float32'), x_smooth, p_zt_xT)
        
        elbo = logpy_x - logqx_y + log_pxz - log_pz_x
        loss = -(self.w_recon * logpy_x - self.w_kl* - logqx_y + self.w_kf * (log_pxz - log_pz_x))
        
        self.elbo_metric.update_state(elbo)
        self.loss_metric.update_state(loss)
        self.add_loss(loss)

        
        metrices = {'log p(y|x)': tf.reduce_mean(logpy_x).numpy(), 
                    'log q(x|y)': tf.reduce_mean(logqx_y).numpy(), 
                    'log p(x,z)': tf.reduce_mean(log_pxz).numpy(), 
                    'log p(z|x)': tf.reduce_mean(log_pz_x).numpy()
                   }

        return p_y_x, metrices
    
    @tf.function
    def predict(self, inputs):
        y_0 = inputs[2]#[:,0,...]
        y = inputs[0]
        mask = inputs[1]
        q_x_y = self.encoder(y) 
        x = q_x_y.sample()
        
        #Smooth
        smooth_dist = self.kf.get_smooth_dist(x, mask)
        
        # Filter        
        filt_dist, filt_pred_dist = self.kf.get_filter_dist(x, mask, True)
         
        y_hat_filt = self.decoder([y_0, filt_dist.sample()]).sample()
        y_hat_pred = self.decoder([y_0, filt_pred_dist.sample()]).sample()
        y_hat_smooth = self.decoder([y_0, smooth_dist.sample()]).sample()
        y_hat_vae = self.decoder([y_0, x]).sample()  
        
        return [{'name':'filt', 'data': y_hat_filt},
                {'name':'pred', 'data': y_hat_pred},
                {'name':'smooth', 'data': y_hat_smooth},
                {'name':'vae', 'data': y_hat_vae}]      

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
    
    @tf.function
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