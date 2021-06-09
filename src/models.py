import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from tqdm import tqdm

from .layers import Encoder, Decoder
from .kalman_filter import KalmanFilter as Linear_KF
from .kalman_filter_k import KalmanFilterK as K_KF

from .utils import plot_to_image, latent_plot

tfd = tfp.distributions
tfk = tf.keras
tfpl = tfp.layers

import numpy as np
import os
def save_if_error(gradients, inputs, model):
    if any([np.any(np.isnan(g.numpy())) for g in gradients]):
        path = './error'
        os.makedirs(path, exist_ok=True)
        [np.save(path + '/input_{0}.npy'.format(j), inputs[j]) for j in range(len(inputs))]
        model.save_weights(path + '/error_model')
        raise Exception("Metrices is NaN")
        
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
        #self.prior = tfd.Normal(loc=tf.zeros(config.dim_x, dtype='float32'),
        #                        scale = tf.ones(config.dim_x, dtype='float32'))
        self.prior = tfd.Independent(
            tfd.Normal(
                loc=tf.zeros(config.dim_x, dtype='float32'),
                scale=tf.ones(config.dim_x, dtype='float32')),
                        reinterpreted_batch_ndims=1)

        self.log_py_x_metric = tfk.metrics.Mean(name = 'log p(y|x) ↑')
        self.log_px_metric = tfk.metrics.Mean(name = 'log p(x) ↑')
        self.log_qx_y_metric = tfk.metrics.Mean(name = 'log q(x|y) ↓')
        self.elbo_metric = tfk.metrics.Mean(name = elbo_name)

        self.ssim_metric = tfk.metrics.Mean(name = 'ssim ↑')

        self.loss_metric = tfk.metrics.Mean(name = 'loss')
        
        self.w_recon = self.config.scale_reconstruction
        self.w_kl = self.config.kl_latent_loss_weight
    
    def get_loss(self, py_x, y, qx_y, px, x, mask):
        logpx = tf.reduce_sum(tf.multiply(px.log_prob(x), mask), axis=-1)
        logqx_y = tf.reduce_sum(tf.multiply(qx_y.log_prob(x), mask), axis=-1)
        logpy_x = tf.reduce_sum(tf.multiply(py_x.log_prob(y), mask), axis=-1)
                
        self.log_py_x_metric.update_state(logpy_x)
        self.log_px_metric.update_state(logpx)
        self.log_qx_y_metric.update_state(logqx_y)
        
        y_pred = py_x.sample()
        if self.debug:
            tf.debugging.assert_equal(y_pred.shape, y.shape)
        
        pred_imgs = tf.reshape(y_pred, (-1, y_pred.shape[2], y_pred.shape[3], 1))
        true_imgs = tf.reshape(y, (-1, y.shape[2], y.shape[3], 1))
        ssim = tf.image.ssim(pred_imgs, true_imgs, max_val=tf.reduce_max(true_imgs), filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        ssim = tf.reshape(ssim, (-1, y.shape[1]))
        self.ssim_metric.update_state(tf.reduce_mean(ssim, axis=-1))
        
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
    
    @tf.function
    def get_latents(self, inputs):
        y_true = inputs[0]        
        q_x_y = self.encoder(y_true)
        x = q_x_y.sample()
        return {"x":x}

    def compile(self, num_batches):
        super(VAE, self).compile()
        self.num_batches = num_batches
        self.epoch = 1
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.config.init_lr, 
                                                                     decay_steps=self.config.decay_steps*num_batches, 
                                                                     decay_rate=self.config.decay_rate, 
                                                                     staircase=True)
        self.opt = tf.keras.optimizers.Adam(lr_schedule)  
    
    def get_image_summary(self, train_data, test_data):
        train_args = self.predict(train_data)
        train_latent = self.get_latents(train_data)
        
        test_args = self.predict(test_data)
        test_latent = self.get_latents(test_data)
        
        return {"Training data": plot_to_image(train_data[0], train_args),
                "Test data":  plot_to_image(test_data[0], test_args),
                "Latent Training data":latent_plot(train_latent),
                "Latent Test data":latent_plot(test_latent)}
    
    
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
        save_if_error(gradients, inputs, self)
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
        y = tf.keras.layers.Input(shape=(self.config.ph_steps, *self.config.dim_y), batch_size=1)
        mask = tf.keras.layers.Input(shape=(self.config.ph_steps), batch_size=1)
        inputs = [y,mask]
        self._print_info(inputs)
    
    def _print_info(self, inputs):    
        y = inputs[0]
        mask = inputs[1]    
        encoder = tf.keras.Model(inputs=y, outputs=self.encoder.call(y), name='Encoder')
        encoder.summary()
        
        x_decoder = tf.keras.layers.Input(shape=(self.config.ph_steps, self.config.dim_x))
        decoder = tf.keras.Model(inputs=[x_decoder], outputs=self.decoder.call(x_decoder), name='Decoder')
        decoder.summary()
        bs = 2
        if len(inputs) > 2:            
            _ = self([np.zeros((bs,self.config.ph_steps,*self.config.dim_y), dtype='float32'), 
                      np.zeros((bs,self.config.ph_steps), dtype='bool'),
                      np.zeros((bs,*self.config.dim_y), dtype='float32')])
        else:
            _ = self([np.zeros((bs,self.config.ph_steps,*self.config.dim_y), dtype='float32'), np.zeros((bs,self.config.ph_steps), dtype='bool')])
        self.summary()

class KVAE(VAE):
    def __init__(self, config, name="kvae", elbo_name='elbo = log p(y|x) - log q(x|y) + log p(x,z) - log p(z|x)', **kwargs):
        super(KVAE, self).__init__(name=name, elbo_name = elbo_name, config=config, **kwargs)
        self.log_pz_x_metric = tfk.metrics.Mean(name = 'log p(z|x) ↓')
        self.log_pxz_metric = tfk.metrics.Mean(name = 'log p(x,z) ↑')
        if config.K == 1:
            self.kf = Linear_KF(self.config)
        else:
            self.kf = K_KF(self.config)
        self.w_kf = self.config.kf_loss_weight
    
    def get_loss(self, p_y_x, y, q_x_y, p_x, x, mask, x_smooth, p_zt_xT):
        logpy_x, logpx, logqx_y = super(KVAE, self).get_loss(p_y_x, y, q_x_y, p_x, x, mask)        
        log_pz_z, log_px_z, log_p0, log_pz_x = self.kf.get_loss(x_smooth, p_zt_xT)

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
        smooth_dist = self.kf.get_smooth_dist(x, mask)
        
        # Filter        
        filt_dist = self.kf.get_filter_dist(x, mask)       
         
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
        
        #Smooth
        smooth_dist = self.kf.get_smooth_dist(x, mask)
        
        # Filter        
        filt_dist, pred_dist = self.kf.get_filter_dist(x, mask, True)   
        
        return {"smooth_mean": smooth_dist.mean(),
                "smooth_cov": smooth_dist.covariance(),
                "filt_mean": filt_dist.mean(),
                "filt_cov": filt_dist.covariance(),
                "pred_mean": pred_dist.mean(), 
                "pred_cov": pred_dist.covariance(),
                "x": x}

    def _print_info(self, inputs):    
        from tabulate import tabulate
        _ = self.kf([np.zeros((1,self.config.ph_steps,self.config.dim_x), dtype='float32'), np.zeros((1,self.config.ph_steps), dtype='bool')])
        info = []
        [info.append([t.name, t.shape, t.trainable]) for t in self.kf.trainable_variables]
        tqdm.write("Model: {0}".format(self.kf.name))
        tqdm.write(tabulate(info, headers=['Name', 'Shape', 'Trainable']))

        super(KVAE, self)._print_info(inputs)

    def sample(self, samples):
        x_samples = self.kf.kalman_filter.sample(sample_shape=samples)
        y_hat_sample, _,_ = self.decoder(x_samples)
        return y_hat_sample

                                  