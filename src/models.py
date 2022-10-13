import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from tqdm import tqdm

from .layers import Encoder, Decoder
from .lgssm import LGSSM

from .utils import plot_to_image, latent_plot

tfd = tfp.distributions
tfk = tf.keras
tfpl = tfp.layers

import numpy as np
import os
#@tf.function
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
                 elbo_name='elbo = log p(y|x) + log p(x) - log q(x|y)',
                 name="vae",
                 debug = False,
                 **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.config = config
        self.debug = debug
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config, output_channels = output_channels)
        self.prior = tfd.Independent(
            tfd.Normal(
                loc=tf.zeros(config.dim_x, dtype='float32'),
                scale=tf.ones(config.dim_x, dtype='float32')),
                        reinterpreted_batch_ndims=1)
        
        self.encoder_dist = tfpl.IndependentNormal(config.dim_x)
        
        decoder_dist = tfpl.IndependentNormal
        if output_channels > 1:
            self.decoder_dist = decoder_dist((*config.dim_y,output_channels))
        else:
            self.decoder_dist = decoder_dist(config.dim_y)
        
        self.log_pdec_metric = tfk.metrics.Mean(name = 'log p(y|x) ↑')
        self.log_prior_metric = tfk.metrics.Mean(name = 'log p(x) ↑')
        self.log_qenc_metric = tfk.metrics.Mean(name = 'log q(x|y) ↓')
        self.elbo_metric = tfk.metrics.Mean(name = elbo_name)

        self.ssim_metric = tfk.metrics.Mean(name = 'ssim ↑')

        self.loss_metric = tfk.metrics.Mean(name = 'loss')
        
        self.w_recon = self.config.scale_reconstruction
        self.w_kl = self.config.kl_latent_loss_weight
    
    def get_loss(self, p_dec, y, q_enc, prior, x_sample, mask):
        mask_ones = tf.cast(mask == False, dtype='float32')
        log_prior = tf.reduce_sum(tf.multiply(prior.log_prob(x_sample), mask_ones), axis=-1)
        log_qenc = tf.reduce_sum(tf.multiply(q_enc.log_prob(x_sample), mask_ones), axis=-1)
        log_pdec = tf.reduce_sum(tf.multiply(p_dec.log_prob(y), mask_ones), axis=-1)
        #shape = (x.shape[0], x.shape[1], -1)
        #ssd = tf.reduce_sum((tf.reshape(py_x.mean(), shape) - tf.reshape(y, shape))**2, axis=-1)
        #logpy_x = -tf.reduce_sum(tf.multiply(ssd, mask), axis=-1)
                
        self.log_pdec_metric.update_state(log_pdec)
        self.log_prior_metric.update_state(log_prior)
        self.log_qenc_metric.update_state(log_qenc)
        
        y_pred = p_dec.sample()
        if self.debug:
            tf.debugging.assert_equal(y_pred.shape, y.shape)
        
        pred_imgs = tf.reshape(y_pred, (-1, y_pred.shape[2], y_pred.shape[3], 1))
        true_imgs = tf.reshape(y, (-1, y.shape[2], y.shape[3], 1))
        ssim = tf.image.ssim(pred_imgs, true_imgs, max_val=tf.reduce_max(true_imgs), filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        ssim = tf.reshape(ssim, (-1, y.shape[1]))
        
        self.ssim_metric.update_state(tf.reduce_mean(ssim, axis=-1))

        if "vae" in self.config.losses:
            loss = -(self.w_recon * log_pdec + self.w_kl*(log_prior - log_qenc))
            self.loss_metric.update_state(loss)
            self.add_loss(loss)

        metrices = {'log_pdec': log_pdec,
                    'log_prior': log_prior,
                    'log_qenc': log_qenc,
                    'elbo': log_pdec + log_prior - log_qenc
                   }
        return metrices
       

    def call(self, inputs, traning):
        y = inputs[0]
        mask = inputs[1]
        p_dec, q_enc, x_sample = self.forward(y, mask, traning)
        
        m = self.get_loss(p_dec, y, q_enc, self.prior, x_sample, mask)
        self.elbo_metric.update_state(m['elbo'])
        
        metrices = {'log p(y|x)':tf.reduce_mean(m['log_pdec']).numpy(), 
                    'log p(x)': tf.reduce_mean(m['log_prior']).numpy(), 
                    'log q(x|y)': tf.reduce_mean(m['log_qenc']).numpy()
                   }
        
        return p_dec, metrices
    
    def forward(self, y, mask, traning):
        mu, sigma = self.encoder(y, traning)
        q_enc = self.encoder_dist(tf.concat([mu, sigma], axis=-1))
        x_sample = q_enc.sample()
        mu, sigma = self.decoder(x_sample, traning)
        p_dec = self.decoder_dist(tf.concat([mu, sigma], axis=-1))
        
        if self.debug:
            tf.debugging.assert_equal(p_dec.batch_shape, y.shape, "{0} vs {1}".format(p_dec.batch_shape, y.shape))
            tf.debugging.assert_equal(q_enc.batch_shape, (*y.shape[0:2], self.config.dim_x), "{0} vs {1}".format(q_enc.batch_shape, (*y.shape[0:2], self.config.dim_x)))
            tf.debugging.assert_equal(x_sample.shape, (*y.shape[0:2], self.config.dim_x), "{0} vs {1}".format(x_sample.shape, (*y.shape[0:2], self.config.dim_x)))
        return p_dec, q_enc, x_sample
    
    @tf.function
    def predict(self, inputs):
        y = inputs[0]
        mask = inputs[1]
        p_dec, q_enc, x_sample = self.forward(y, mask, False)
        y_sample = p_dec.sample()
        if self.debug:
            tf.debugging.assert_equal(y.shape, y_sample.shape, "{0} vs {1}".format(y.shape, y_sample.shape))
        return [{'name':'vae', 'data': y_sample}]
    
    @tf.function
    def get_latents(self, inputs):
        y = inputs[0]
        mu, sigma = self.encoder(y, False)
        q_enc = self.encoder_dist(tf.concat([mu, sigma], axis=-1))
        x_sample = q_enc.sample()
        return {"x":x_sample}

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
    
    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            _, metrices = self(inputs, training=True)
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
        #save_if_error(gradients, inputs, self)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
        self.opt.apply_gradients(zip(gradients, variables))
        metrices['loss'] = loss#.numpy()
        return loss, metrices
    
    @tf.function
    def test_step(self, inputs):
        _, metrices = self(inputs, training=False)
        loss = tf.reduce_mean(sum(self.losses))
        metrices['loss'] = loss#.numpy()
        return loss, metrices 
    
    def model(self):
        inputs = tf.keras.layers.Input(shape=(self.config.dim_y))
        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))
    
    def info(self):
        y = tf.keras.layers.Input(shape=(self.config.ph_steps, *self.config.dim_y), batch_size=self.config.batch_size)
        mask = tf.keras.layers.Input(shape=(self.config.ph_steps), batch_size=self.config.batch_size)
        inputs = [y,mask]
        self._print_info(inputs)
    
    def _print_info(self, inputs):    
        y = inputs[0]
        mask = inputs[1]    
        encoder = tf.keras.Model(inputs=y, outputs=self.encoder.call(y, False), name='Encoder')
        encoder.summary()
        
        x_sample = tf.keras.layers.Input(shape=(self.config.ph_steps, self.config.dec_input_dim), batch_size=self.config.batch_size)
        decoder = tf.keras.Model(inputs=[x_sample], outputs=self.decoder.call(x_sample, False), name='Decoder')
        decoder.summary()
        
        bs = self.config.batch_size
        if len(inputs) > 2:            
            _ = self([np.zeros((bs,self.config.ph_steps,*self.config.dim_y), dtype='float32'), 
                      np.zeros((bs,self.config.ph_steps), dtype='bool'),
                      np.zeros((bs,*self.config.dim_y), dtype='float32')], training=False)
        else:
            _ = self([np.zeros((bs,self.config.ph_steps,*self.config.dim_y), dtype='float32'), np.zeros((bs,self.config.ph_steps), dtype='bool')], training=False)
        self.summary()

class KVAE(VAE):
    def __init__(self, config, name="kvae", elbo_name='elbo = log p(y|x) - log q(x|y) + log p(x,z) - log p(z|x)', **kwargs):
        super(KVAE, self).__init__(name=name, elbo_name = elbo_name, config=config, **kwargs)
        self.log_p_smooth_metric = tfk.metrics.Mean(name = 'log p(z|x) ↓')
        self.log_p_joint_metric = tfk.metrics.Mean(name = 'log p(x,z) ↑')
        self.lgssm = LGSSM(self.config)
        self.w_kf = self.config.kf_loss_weight
    
    def get_loss(self, y, x_sample, z_sample, q_enc, p_smooth, p_dec, mask, prior):
        m = super(KVAE, self).get_loss(p_dec, y, q_enc, prior, x_sample, mask)        
        log_pdec = m['log_pdec']
        log_qenc = m['log_qenc']
        
        log_prob_z_z1, log_prob_x_z, log_pz1, log_psmooth, log_px = self.lgssm.get_loss(x_sample, z_sample, p_smooth)
        
        mask_ones = tf.cast(mask == False, dtype='float32')
        log_prob_z_z1 = tf.multiply(log_prob_z_z1, mask_ones[:,1:])
        log_prob_x_z = tf.multiply(log_prob_x_z, mask_ones)
        log_pz1 = tf.multiply(log_pz1, mask_ones[:,0])
        log_psmooth = tf.multiply(log_psmooth, mask_ones)

        log_psmooth = tf.reduce_sum(log_psmooth, axis=-1)
        log_pjoint = tf.reduce_sum(log_prob_x_z, axis=-1) + log_pz1 + tf.reduce_sum(log_prob_z_z1, axis=-1)
        
        self.log_p_smooth_metric.update_state(log_psmooth)
        self.log_p_joint_metric.update_state(log_pjoint)
        
        elbo = log_pdec - log_qenc + log_pjoint - log_psmooth
        
        #shape = (y.shape[0], y.shape[1], -1)
        #y_pred = p_dec.mean()#*self.external_mask
        #ssd = tf.reduce_sum((tf.reshape(y_pred, shape) - tf.reshape(y, shape))**2, axis=-1)
        #log_pdec_loss = -tf.reduce_sum(tf.multiply(ssd, mask_ones), axis=-1)
        if 'kvae_loss' in self.config.losses:
            loss = -(self.w_recon * log_pdec - self.w_kl*log_qenc + self.w_kf * (log_pjoint - log_psmooth))
            self.loss_metric.update_state(loss)
            self.add_loss(loss)
        if 'lgssm_ml' in self.config.losses:
            loss = -tf.reduce_mean(log_px)
            self.loss_metric.update_state(loss)
            self.add_loss(loss)
        
        self.elbo_metric.update_state(elbo)        
        
        metrices = {'log p(y|z)': tf.reduce_mean(log_pdec), 
                    'log q(x|y)': tf.reduce_mean(log_qenc), 
                    'log p(x,z)': tf.reduce_mean(log_pjoint), 
                    'log p(z|x)': tf.reduce_mean(log_psmooth)
                   }
        
        return metrices
    
    def call(self, inputs, training):
        y = inputs[0]
        mask = inputs[1]
        q_enc, p_smooth, p_dec, x_sample, z_sample = self.forward(y, mask, training)
        
        metrices = self.get_loss(y, x_sample, z_sample, q_enc, p_smooth, p_dec, mask = mask, prior = self.prior)

        return p_dec, metrices
    
    def forward(self, y, mask, training):
        mu, sigma = self.encoder(y, training)
        q_enc = self.encoder_dist(tf.concat([mu, sigma], axis=-1))

        x_sample = q_enc.sample()
        p_smooth = self.lgssm([x_sample, mask], training=training)
        
        z_sample = p_smooth.sample()
        if self.config.dec_input_dim == self.config.dim_x:
            mu, sigma = self.decoder(x_sample, training)
            p_dec = self.decoder_dist(tf.concat([mu, sigma], axis=-1))
        else:
            mu, sigma = self.decoder(z_sample, training)
            p_dec = self.decoder_dist(tf.concat([mu, sigma], axis=-1))
        return q_enc, p_smooth, p_dec, x_sample, z_sample

    @tf.function
    def predict(self, inputs):
        y_filt_sample, y_pred_sample, y_smooth_sample = self._predict(inputs)
        
        return [{'name':'filt', 'data': y_filt_sample},
               {'name':'smooth', 'data': y_smooth_sample}]
    
    @tf.function
    def _predict(self, inputs):
        y = inputs[0]
        mask = inputs[1]
        mu, sigma = self.encoder(y, False)
        q_enc = self.encoder_dist(tf.concat([mu, sigma], axis=-1))
        
        x_sample = q_enc.sample()
                
        #Smooth
        p_smooth, p_obssmooth = self.lgssm.get_smooth_dist(x_sample, mask)
        
        # Filter        
        p_filt, p_obsfilt, p_pred, p_obspred = self.lgssm.get_filter_dist(x_sample, mask, get_pred=True)
                
        if self.config.dec_input_dim == self.config.dim_x:
            filt_sample = p_obsfilt.sample()
            pred_sample = p_obspred.sample()
            smooth_sample = p_obssmooth.sample()
        else: # self.config.dec_input_dim == self.config.dim_z:
            filt_sample = p_filt.sample()
            pred_sample = p_pred.sample()
            smooth_sample = p_smooth.sample()    
        
        mu, sigma = self.decoder(filt_sample, False)
        p_filt = self.decoder_dist(tf.concat([mu, sigma], axis=-1))
        
        mu, sigma = self.decoder(pred_sample, False)
        p_pred = self.decoder_dist(tf.concat([mu, sigma], axis=-1))
        
        mu, sigma = self.decoder(smooth_sample, False)
        p_smooth = self.decoder_dist(tf.concat([mu, sigma], axis=-1))
        return p_filt.sample(), p_pred.sample(), p_smooth.sample()
    
    @tf.function
    def _get_filt(self, inputs):
        y = inputs[0]
        mask = inputs[1]
        mu, sigma = self.encoder(y, False)
        q_enc = self.encoder_dist(tf.concat([mu, sigma], axis=-1))
        
        x_sample = q_enc.sample()
        
        # Filter        
        p_filt, p_obsfilt = self.lgssm.get_filter_dist(x_sample, mask, get_pred=False)
                
        if self.config.dec_input_dim == self.config.dim_x:
            filt_sample = p_obsfilt.sample()
        else: # self.config.dec_input_dim == self.config.dim_z:
            filt_sample = p_filt.sample()
        
        mu, sigma = self.decoder(filt_sample, False)
        p_filt = self.decoder_dist(tf.concat([mu, sigma], axis=-1))
        
        return p_filt.sample()
    
    @tf.function
    def get_latents(self, inputs):
        y = inputs[0]
        mask = inputs[1]
        
        mu, sigma = self.encoder(y, False)
        q_enc = self.encoder_dist(tf.concat([mu, sigma], axis=-1))
        x_sample = q_enc.sample()
        
        return self.lgssm.get_obs_distributions(x_sample, mask)

    def _print_info(self, inputs):    
        from tabulate import tabulate
        _ = self.lgssm([
            np.zeros((self.config.batch_size,self.config.ph_steps,self.config.dim_x), dtype='float32'),
            np.zeros((self.config.batch_size,self.config.ph_steps), dtype='bool')], training=False)
        info = []
        [info.append([t.name, t.shape, t.trainable]) for t in self.lgssm.trainable_variables]
        tqdm.write("Model: {0}".format(self.lgssm.name))
        tqdm.write(tabulate(info, headers=['Name', 'Shape', 'Trainable']))

        super(KVAE, self)._print_info(inputs)
                                  