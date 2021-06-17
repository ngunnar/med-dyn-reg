import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import tensorflow_addons as tfa

tfk = tf.keras
tfpl = tfp.layers

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

def flow2grid(flow, d = 10):
    bs = flow.shape[0]
    ph_steps = flow.shape[1]
    dim_y = flow.shape[-3:-1]
    
    l = np.array(dim_y.as_list()) // d
    # grid image
    #img_grid = np.zeros((bs, ph_steps, *dim_y), dtype='float32')
    img_grid = np.zeros((bs, *dim_y), dtype='float32')
    img_grid[:,10::10,:] = 1.0
    img_grid[:,:,10::10] = 1.0
    return warp(flow, img_grid)
    '''
    # meshgrid
    y_range = tf.range(start=0, limit=dim_y[0])/dim_y[0]
    x_range = tf.range(start=0, limit=dim_y[1])/dim_y[1]
    y_grid, x_grid = tf.meshgrid(y_range, x_range, indexing="ij")
    grid = tf.cast(tf.stack((y_grid, x_grid), -1), 'float32')
    
    # flow2grid
    f = -tf.reshape(flow, (bs*ph_steps, -1, flow.shape[-1]))
    transform_grid = tf.repeat(grid[None,...], bs*ph_steps, axis=0)
    transform_grid = tf.reshape(transform_grid, (bs*ph_steps, -1, grid.shape[-1]))
    
    scaled_grid = transform_grid*np.asarray(dim_y)[None, None,...] + f
    
    transform_grid = tfa.image.interpolate_bilinear(tf.reshape(img_grid, (bs*ph_steps, *dim_y, 1)), scaled_grid)
    transform_grid = tf.reshape(transform_grid, flow.shape[:-1])
    return transform_grid
    '''
class fVAE(VAE):
    def __init__(self, config, name='fKVAE', output_channels = 2, **kwargs):
        super(fVAE, self).__init__(name=name, output_channels = output_channels, config=config, **kwargs)
        self.grad_flow_metric = tfk.metrics.Mean(name = 'grad flow ↓')
        self.dist = tfpl.IndependentNormal(config.dim_y)
        
    def call(self, inputs, training):
        y = inputs[0]
        mask = inputs[1]
        y_0 = inputs[2]
        p_y_x, phi_y_x, q_x_y, x = self.forward(y, mask, y_0, training)        
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

    def forward(self, y, mask, y_0, training):
        q_x_y = self.encoder(y, training)
        x = q_x_y.sample()
        phi_y_x = self.decoder(x, training)
        if self.debug:
            tf.debugging.assert_equal(phi_y_x.batch_shape, (*y.shape, 2), "{0} vs {1}".format(phi_y_x.batch_shape, (*y.shape, 2)))
            tf.debugging.assert_equal(q_x_y.batch_shape, (*y.shape[0:2], self.config.dim_x), "{0} vs {1}".format(q_x_y.batch_shape, (*y.shape[0:2], self.config.dim_x)))
            tf.debugging.assert_equal(x.shape, (*y.shape[0:2], self.config.dim_x), "{0} vs {1}".format(x.shape, (*y.shape[0:2], self.config.dim_x)))

        phi_mu = phi_y_x.mean() #bs, t, w, h, 2
        y_mu = warp(phi_mu, y_0)
        y_sigma = tf.ones_like(y_mu, dtype='float32') * tfp.math.softplus_inverse(0.01) # softplus is used so a -4.6 approx std 0.01
        y_mu = tf.reshape(y_mu, (-1, y_mu.shape[1], np.prod(y_mu.shape[2:])))
        y_sigma = tf.reshape(y_sigma, (-1, y_mu.shape[1], np.prod(y_mu.shape[2:])))
        p_y_x = self.dist(tf.concat([y_mu, y_sigma], axis=-1))
        if self.debug:
            tf.debugging.assert_equal(p_y_x.batch_shape, y.shape, "{0} vs {1}".format(p_y_x.batch_shape, y.shape))
        return p_y_x, phi_y_x, q_x_y, x
    
    @tf.function
    def predict(self, inputs):
        y = inputs[0]
        mask = inputs[1]
        y_0 = inputs[2]
        p_y_x, phi_y_x, q_x_y, x = self.forward(y, mask, y_0, False)
        phi_hat = phi_y_x.sample()
        
        y_hat = p_y_x.sample()
        if self.debug:
            tf.debugging.assert_equal(y.shape, y_hat.shape, "{0} vs {1}".format(y.shape, y_hat.shape))
            tf.debugging.assert_equal((*y.shape, 2), phi_hat.shape, "{0} vs {1}".format((*y.shape, 2), phi_hat.shape))
        return [{'name':'vae', 'data': y_hat},
               {'name':'flow', 'data': phi_hat},
               {'name':'grid', 'data': flow2grid(phi_hat)}] 

    def info(self):
        y = tf.keras.layers.Input(shape=(self.config.ph_steps, *self.config.dim_y), batch_size=1)
        mask = tf.keras.layers.Input(shape=(self.config.ph_steps), batch_size=1)
        first_frame = tf.keras.layers.Input(shape=self.config.dim_y, batch_size=1)
        inputs = [y, mask, first_frame]

        self._print_info(inputs)        


class fKVAE(KVAE):
    def __init__(self, config, name="fKVAE", output_channels=2, **kwargs):
        super(fKVAE, self).__init__(name=name, output_channels = output_channels, config=config, **kwargs)
        self.grad_flow_metric = tfk.metrics.Mean(name = 'grad flow ↓')
        self.dist = tfpl.IndependentNormal(config.dim_y)
    
    def call(self, inputs, training):
        y = inputs[0]
        mask = inputs[1]
        y_0 = inputs[2]
        p_y_x, phi_y_x, q_x_y, x, x_smooth, p_zt_xT = self.forward(y, mask, y_0, training)
        
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
    
    def forward(self, y, mask, y_0, training):
        phi_y_x, q_x_y, x, x_smooth, p_zt_xT =  super().forward(y, mask, training)
        phi_sample = phi_y_x.mean() #bs, t, w, h, 2
        y_mu = warp(phi_sample, y_0)
        y_sigma = tf.ones_like(y_mu, dtype='float32') * tfp.math.softplus_inverse(0.01) # softplus is used so a -4.6 approx std 0.01
        y_mu = tf.reshape(y_mu, (-1, y_mu.shape[1], np.prod(y_mu.shape[2:])))
        y_sigma = tf.reshape(y_sigma, (-1, y_mu.shape[1], np.prod(y_mu.shape[2:])))
        p_y_x = self.dist(tf.concat([y_mu, y_sigma], axis=-1))

        return p_y_x, phi_y_x, q_x_y, x, x_smooth, p_zt_xT

    @tf.function
    def predict(self, inputs):
        y_0 = inputs[2]#[:,0,...]
        y = inputs[0]
        mask = inputs[1]
        q_x_y = self.encoder(y, False) 
        x = q_x_y.sample()
        
        #Smooth
        smooth_dist = self.kf.get_smooth_dist(x, mask)
        
        # Filter        
        filt_dist, filt_pred_dist = self.kf.get_filter_dist(x, mask, True)
         
        phi_hat_filt = self.decoder(filt_dist.sample(), False).sample()
        phi_hat_pred = self.decoder(filt_pred_dist.sample(), False).sample()
        phi_hat_smooth = self.decoder(smooth_dist.sample(), False).sample()
        phi_hat_vae = self.decoder(x).sample()  
        
        y_hat_filt = warp(phi_hat_filt, y_0)
        y_hat_pred = warp(phi_hat_pred, y_0)
        y_hat_smooth = warp(phi_hat_smooth, y_0)
        y_hat_vae = warp(phi_hat_vae, y_0)

        return [{'name':'filt', 'data': y_hat_filt},
                {'name':'filt_flow', 'data': phi_hat_filt},
                {'name':'filt_grid', 'data': flow2grid(phi_hat_filt)},                
                {'name':'pred', 'data': y_hat_pred},
                {'name':'pred_flow', 'data': phi_hat_pred},
                {'name':'pred_grid', 'data': flow2grid(phi_hat_pred)},
                {'name':'smooth', 'data': y_hat_smooth},
                {'name':'smooth_flow', 'data': phi_hat_smooth},
                {'name':'smooth_grid', 'data': flow2grid(phi_hat_smooth)},
                {'name':'vae', 'data': y_hat_vae},
                {'name':'vae_flow', 'data': phi_hat_vae},
                {'name':'vae_grid', 'data': flow2grid(phi_hat_vae)}]
    
    
    def info(self):
        y = tf.keras.layers.Input(shape=(self.config.ph_steps, *self.config.dim_y), batch_size=1)
        mask = tf.keras.layers.Input(shape=(self.config.ph_steps), batch_size=1)
        first_frame = tf.keras.layers.Input(shape=self.config.dim_y, batch_size=1)
        inputs = [y, mask, first_frame]

        self._print_info(inputs)

    def sample(self, samples):
        x_samples = self.kf.kalman_filter.sample(sample_shape=samples)
        return self.decoder(x_samples).sample()

class Bspline(tf.keras.layers.Layer):
    def __init__(self, dim_y, dim_x, name='Bspline', **kwargs):
        super(Bspline,self).__init__(name=name, **kwargs)
        y_range = tf.range(start=0, limit=dim_y[0])/dim_y[0]
        x_range = tf.range(start=0, limit=dim_y[1])/dim_y[1]
        y_grid, x_grid = tf.meshgrid(y_range, x_range, indexing="ij")
        self.grid = tf.cast(tf.stack((y_grid, x_grid), -1), 'float32')
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dist = tfpl.IndependentNormal((*dim_y,2))
        assert np.sqrt(self.dim_x/2).is_integer()
        
    def body(i, d, p, s):
        d_new = tfa.image.interpolate_bilinear(p[...,i:i+1], s)
        return (i+1, tf.concat([d,d_new], axis=-1), p, s)
    
    def call(self, inputs):
        parameters = inputs
        bs = tf.shape(parameters)[0]
        ph_steps = parameters.shape[1]
        parameters = tf.reshape(parameters, (bs*ph_steps, int(np.sqrt(self.dim_x/2)), int(np.sqrt(self.dim_x/2)), 2))
                 
        transform_grid = tf.repeat(self.grid[None,...], bs*ph_steps, axis=0)
        transform_grid = tf.reshape(transform_grid, (bs*ph_steps, -1, self.grid.shape[-1]))
        
        scaled_points = transform_grid * (np.array(parameters.shape[1:-1], dtype='float32') - 1)[None,None,...]
        
        cond = lambda i, d, p, s: i < p.shape[-1]
        i0 = tf.constant(1)
        _, d,_,_ = tf.while_loop(cond, 
                          Bspline.body, 
                          loop_vars=[i0, 
                                     tfa.image.interpolate_bilinear(parameters[...,0:1], scaled_points), 
                                     parameters,
                                     scaled_points],
                          shape_invariants=[i0.get_shape(), 
                                            tf.TensorShape([None, np.prod(self.dim_y), None]), #bs 
                                            parameters.get_shape(), 
                                            scaled_points.get_shape()])        
        flow = -d*np.asarray(self.dim_y)[None, None,...]
        flow = tf.reshape(flow, (bs, ph_steps, *self.dim_y, 2))
        mu = flow
        sigma = tf.ones_like(mu, dtype='float32') * tfp.math.softplus_inverse(0.01) # softplus is used so a -4.6 approx std 0.01

        mu = tf.reshape(mu, (-1, mu.shape[1], np.prod(mu.shape[2:])))
        sigma = tf.reshape(sigma, (-1, mu.shape[1], np.prod(mu.shape[2:])))
        phi_y_x = self.dist(tf.concat([mu, sigma], axis=-1))
        
        return phi_y_x
    
class bKVAE(fKVAE):
    def __init__(self, config, name='bKVAE', **kwargs):
        super(bKVAE, self).__init__(name=name, config=config, **kwargs)
        assert config.dim_x == 32, "dim x 32 is the only supported atm"
        self.decoder = Bspline(config.dim_y, config.dim_x)
        
from .layers import Encoder
class ufKVAE(fKVAE):
    def __init__(self, config, name='UFLOW_KVAE', **kwargs):
        super(ufKVAE, self).__init__(name=name, config=config, unet_decoder = True, **kwargs)
        self.u_encoder = Encoder(self.config, unet=True)
    
    def forward(self, y, mask, y_0, training):
        # Encoder
        q_x_y = self.encoder(y, training)
        p_x_y0, feats = self.u_encoder(y_0[:,None,:,:], traning)
        
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
        y_sigma = tf.ones_like(y_mu, dtype='float32') * tfp.math.softplus_inverse(0.01) # softplus is used so a -4.6 approx std 0.01
        y_mu = tf.reshape(y_mu, (-1, y_mu.shape[1], np.prod(y_mu.shape[2:])))
        y_sigma = tf.reshape(y_sigma, (-1, y_mu.shape[1], np.prod(y_mu.shape[2:])))
        p_y_x = self.normal(tf.concat([y_mu, y_sigma], axis=-1))
        return p_y_x, phi_y_x, q_x_y, p_x_y0, x, x_smooth, p_zt_xT
    
    def call(self, inputs, traning):
        y = inputs[0]
        mask = inputs[1]
        y_0 = inputs[2]
        p_y_x, phi_y_x, q_x_y, p_x_y0, x, x_smooth, p_zt_xT = self.forward(y, mask, y_0, traning)
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
        q_x_y = self.encoder(y, False) 
        p_x_y0, feats = self.u_encoder(y_0[:,None,:,:], False)
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
        phi_hat_filt = self.decoder([x_in, feats], False).sample()
        x_smooth = smooth_dist.sample()
        x_in = tf.concat([x_smooth, X_0], axis=-1)
        phi_hat_smooth = self.decoder([x_in, feats], False).sample()
        x_in = tf.concat([x, X_0], axis=-1)
        phi_hat_vae = self.decoder([x_in, feats], False).sample()  
        
        y_hat_filt = warp(phi_hat_filt, y_0)
        y_hat_smooth = warp(phi_hat_smooth, y_0)
        y_hat_vae = warp(phi_hat_vae, y_0)

        return [{'name':'filt', 'data': y_hat_filt},
                {'name':'filt_flow', 'data': phi_hat_filt},
                {'name':'smooth', 'data': y_hat_smooth},
                {'name':'smooth_flow', 'data': phi_hat_smooth},
                {'name':'vae', 'data': y_hat_vae},
                {'name':'vae_flow', 'data': phi_hat_vae}]
