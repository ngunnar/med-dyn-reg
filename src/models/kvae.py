import tensorflow as tf
from voxelmorph.tf.losses import NCC, MSE
import numpy as np

from .layers_skip import Encoder, Decoder
from .lgssm import LGSSM
from .utils import get_log_dist

tfk = tf.keras

class KVAE(tfk.Model):
    def __init__(self, config, name='kvae', **kwargs):
        super(KVAE, self).__init__(self, name = name, **kwargs)
        self.config = config
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config, output_channels = 1)
        self.lgssm = LGSSM(self.config)
        
        self.init_w_kf = tf.Variable(initial_value=config.kf_loss_weight, trainable=False, dtype="float32", name="init_w_kf")
        self.w_kf = tf.Variable(initial_value=config.kf_loss_weight, trainable=False, dtype="float32", name="w_kf")

        self.init_w_recon = tf.Variable(initial_value=config.scale_reconstruction, trainable=False, dtype="float32", name="init_w_kf")
        self.w_recon = tf.Variable(initial_value=config.scale_reconstruction, trainable=False, dtype="float32", name="w_kf")

        self.init_w_kl = tf.Variable(initial_value=config.kl_latent_loss_weight, trainable=False, dtype="float32", name="init_w_kf")
        self.w_kl = tf.Variable(initial_value=config.kl_latent_loss_weight, trainable=False, dtype="float32", name="w_kf")
        
        #self.loss_metric = tfk.metrics.Mean(name="loss")

        self.mse_metric = tfk.metrics.Mean(name = 'MSE ↓')
        self.ncc_metric = tfk.metrics.Mean(name = 'NCC ↑')
        self.log_pdec_metric = tfk.metrics.Mean(name = 'log p(y[t]|x[t]) ↑')
        self.log_qenc_metric = tfk.metrics.Mean(name = 'log q(x[t]|y[t]) ↓')     
        self.log_pzx_metric = tfk.metrics.Mean(name = 'log p(z[t],x[t]) ↑')     
        self.log_pz_x_metric = tfk.metrics.Mean(name = 'log p(z[t]|x[t]) ↓') 
        self.log_px_x_metric = tfk.metrics.Mean(name = 'log p(x[t]|x[:t-1]) ↑')
        
        self.elbo_metric = tfk.metrics.Mean(name = 'ELBO ↑')
        self.ssim_metric = tfk.metrics.Mean(name = 'ssim ↑')

    def call(self, inputs, training=None):
        y = inputs['input_video']
        mask = inputs['input_mask'] 

        q_enc, x, x_ref, log_pred, log_filt, log_p_1, log_smooth, ll, p_dec = self.forward(inputs, training)

        self.set_loss(y, mask, p_dec, q_enc, x, x_ref, log_pred, log_filt, log_p_1, log_smooth, ll)
        #self.loss_metric.update_state(tf.reduce_sum(self.losses))
        return

    #@tf.function
    def forward(self, inputs, training):
        y = inputs['input_video'] # (bs, length, h, w) 
        y_ref = inputs['input_ref'] # (bs, h, w) 
        mask = inputs['input_mask'] # (bs, length)
        length = y.shape[1]
                       
        q_enc, q_ref_enc, x_ref_feat = self.encoder(y, y_ref, training)  
        
        x = q_enc.sample()
        x_ref = q_ref_enc.sample()

        p_dec = self.dec(x, x_ref, x_ref_feat, length, training)

        log_pred, log_filt, log_p_1, log_smooth, ll = self.lgssm([x, x_ref, mask])
        
        return q_enc, x, x_ref, log_pred, log_filt, log_p_1, log_smooth, ll, p_dec
    
    #@tf.function
    def dec(self, x, x_ref, x_ref_feat, length, training):
        return self.decoder([tf.concat([x, tf.repeat(x_ref[:,None,:], length, axis=1)], axis=2), x_ref_feat], training)

    #@tf.function
    def sim_metric(self, y_true, y_pred, mask_ones, metric, length):
        val = metric(y_true, y_pred) # (bs*length, *dim_y, 1)        
        val = tf.reduce_sum(val, axis=np.arange(1, len(val.shape))) # (bs*length)
        val = tf.reshape(val, (-1, length)) # (bs, length)
        val = tf.multiply(val, mask_ones)  # (bs, length)
        return tf.reduce_sum(val, axis=1) # (bs)

    #@tf.function
    def set_loss(self, y, mask, p_dec, q_enc, x, x_ref, log_pred, log_filt, log_p_1, log_smooth, ll):
        length = y.shape[1]
        mask_ones = tf.cast(mask == False, dtype='float32')

        # log p(y|x)        
        log_p_y_x = get_log_dist(p_dec, y, mask_ones)       
        
        # log q(x|y)       
        log_q_x_y = get_log_dist(q_enc, x, mask_ones)          
        
        # log p(x, z)
        log_p_xz = tf.reduce_sum(log_filt, axis=1) + log_p_1 + tf.reduce_sum(log_pred, axis=1)
        
        # log p(z|x)
        log_p_z_x = tf.reduce_sum(log_smooth, axis=1)
        
        # Image simularity metrics
        y_true = tf.reshape(y, (-1, *y.shape[-2:], 1)) # (bs*length, *dim_y, 1)
        y_pred = tf.reshape(p_dec.sample(), (-1, *y.shape[-2:], 1)) # (bs*length, *dim_y, 1)
        
        ncc = self.sim_metric(y_true, y_pred, mask_ones, NCC().ncc, length)
        mse = self.sim_metric(y_true, y_pred, mask_ones, MSE().mse, length)
        ssim_m = lambda t, p: tf.image.ssim(p, t, max_val=tf.reduce_max(t), filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, return_index_map=True)
        ssim = self.sim_metric(y_true, y_pred, mask_ones, ssim_m, length)

        if 'kvae_loss' in self.config.losses:
            if 'ncc' in self.config.losses:
                recon = ncc
            elif 'mse' in self.config.losses:
                recon = -mse  
            else:
                recon = log_p_y_x

            loss = -(self.w_recon * recon - self.w_kl*log_q_x_y + self.w_kf * (log_p_xz - log_p_z_x))
            #loss = -(self.w_recon * recon - self.w_kl*log_q_x_y + self.w_kf * tf.reduce_sum(ll, axis=1))
            #self.loss_metric.update_state(loss)
            self.add_loss(loss)
        
        if 'lgssm_ml' in self.config.losses:                        
            loss = -tf.reduce_mean(ll, axis=1)
            #self.loss_metric.update_state(loss)
            self.add_loss(loss)

        self.add_metric(self.w_kl, "W_kl")
        self.add_metric(self.w_kf, "W_kf")
        self.add_metric(self.w_recon, "W_recon")
        
        # METRICES
        elbo = log_p_y_x - log_q_x_y + log_p_xz - log_p_z_x
        
        self.mse_metric.update_state(mse)
        self.ncc_metric.update_state(ncc)
        self.log_pdec_metric.update_state(log_p_y_x)
        self.log_qenc_metric.update_state(log_q_x_y)
        self.log_pzx_metric.update_state(log_p_xz)
        self.log_pz_x_metric.update_state(log_p_z_x)
        self.elbo_metric.update_state(elbo)     
        self.log_px_x_metric.update_state(tf.reduce_sum(ll, axis=1))

        y_pred = p_dec.sample()
        self.ssim_metric.update_state(ssim)

    def eval(self, inputs):
        y = inputs['input_video']
        y_ref = inputs['input_ref']
        mask = inputs['input_mask'] 
        length = y.shape[1]

        q_enc, q_ref_enc, x_ref_feat = self.encoder(y, y_ref, training=False)        
        
        x = q_enc.sample()
        x_ref = q_ref_enc.sampel()
        
        p_dec = self.dec(x, x_ref, x_ref_feat, length, False)     

        latent_dist = self.lgssm.get_distribtions(x, x_ref, mask)
        p_dec_smooth = self.dec(latent_dist['smooth'].sample(), x_ref, x_ref_feat, length, False)
        p_dec_filt = self.dec(latent_dist['filt'].sample(), x_ref, x_ref_feat, length, False)
        p_dec_pred = self.dec(latent_dist['pred'].sample(), x_ref, x_ref_feat, length, False)    

        y_vae = p_dec.sample()
        y_smooth = p_dec_smooth.sample()
        y_filt = p_dec_filt.sample()
        y_pred = p_dec_pred.sample()

        return {'image_data': {'vae': {'images' : y_vae},
                        'smooth': {'images': y_smooth},
                        'filt': {'images': y_filt},
                        'pred': {'images': y_pred}},
                'latent_dist': latent_dist,
                'x_obs': x,
                'x_ref': x_ref,
                'x_ref_feat': x_ref_feat}


    def compile(self, num_batches, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None, **kwargs):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.config.init_lr, 
                                                                     decay_steps=self.config.decay_steps*num_batches, 
                                                                     decay_rate=self.config.decay_rate, 
                                                                     staircase=True)        
        
        optimizer = tf.keras.optimizers.Adam(lr_schedule)

        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)

    def train_step(self, inputs):                    
        with tf.GradientTape() as tape:
            self(inputs, training=True)  # Forward pass

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(self.losses, trainable_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))        

        out = {self.loss_metric.name: self.loss_metric.result()}
        out.update({m.name: m.result() for m in self.metrics})    
        return {m.name: m.result() for m in self.metrics}