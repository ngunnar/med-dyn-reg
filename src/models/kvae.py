import tensorflow as tf
from voxelmorph.tf.losses import NCC, MSE

from .layers_skip import Encoder, Decoder
from .lgssm import LGSSM
from .utils import get_log_dist, set_name

tfk = tf.keras

class KVAE(tfk.Model):
    def __init__(self, config, name='kvae', prefix=None, **kwargs):
        super(KVAE, self).__init__(self, name = set_name(name, prefix), **kwargs)
        self.config = config
        self.encoder = Encoder(self.config, prefix=prefix)
        self.decoder = Decoder(self.config, output_channels = 1, prefix=prefix)
        self.lgssm = LGSSM(self.config, prefix=prefix)
        
        self.init_w_kf = tf.Variable(initial_value=config.kf_loss_weight, trainable=False, dtype="float32", name=set_name("init_w_kf", prefix))
        self.w_kf = tf.Variable(initial_value=config.kf_loss_weight, trainable=False, dtype="float32", name=set_name("w_kf", prefix))

        self.init_w_recon = tf.Variable(initial_value=config.scale_reconstruction, trainable=False, dtype="float32", name=set_name("init_w_kf", prefix))
        self.w_recon = tf.Variable(initial_value=config.scale_reconstruction, trainable=False, dtype="float32", name=set_name("w_kf", prefix))

        self.init_w_kl = tf.Variable(initial_value=config.kl_latent_loss_weight, trainable=False, dtype="float32", name=set_name("init_w_kf", prefix))
        self.w_kl = tf.Variable(initial_value=config.kl_latent_loss_weight, trainable=False, dtype="float32", name=set_name("w_kf", prefix))
        
        #self.loss_metric = tfk.metrics.Mean(name=set_name("loss", prefix))

        self.mse_metric = tfk.metrics.Mean(name = set_name('MSE ↓', prefix))
        self.ncc_metric = tfk.metrics.Mean(name = set_name('NCC ↑', prefix))
        self.log_py_metric = tfk.metrics.Mean(name = set_name('log p(y[t]|x[t]) ↑', prefix))
        self.log_qx_metric = tfk.metrics.Mean(name = set_name('log q(x[t]|y[t]) ↓', prefix))
        self.log_pzx_metric = tfk.metrics.Mean(name = set_name('log p(z[t],x[t]) ↑', prefix))
        self.log_pz_x_metric = tfk.metrics.Mean(name = set_name('log p(z[t]|x[t]) ↓', prefix))
        self.log_px_x_metric = tfk.metrics.Mean(name = set_name('log p(x[t]|x[:t-1]) ↑', prefix))
        
        self.elbo_metric = tfk.metrics.Mean(name = set_name('ELBO ↑', prefix))
        self.ssim_metric = tfk.metrics.Mean(name = set_name('ssim ↑', prefix))

    def call(self, inputs, training=None):
        y = inputs['input_video']
        mask = inputs['input_mask'] 

        q_x, x, log_pred, log_filt, log_p_1, log_smooth, ll, p_y = self.forward(inputs, training)

        self.set_loss(y, mask, p_y, q_x, x, log_pred, log_filt, log_p_1, log_smooth, ll)
        #self.loss_metric.update_state(tf.reduce_sum(self.losses))
        return

    #@tf.function
    def forward(self, inputs, training):
        y = inputs['input_video'] # (bs, length, h, w) 
        y_ref = inputs['input_ref'] # (bs, h, w) 
        mask = inputs['input_mask'] # (bs, length)
        length = y.shape[1]
                       
        q_x, q_s, s_feat = self.encoder(y, y_ref, training)  
        
        x = q_x.sample()
        s = q_s.sample()

        p_y = self.dec(x, s, s_feat, length, training)

        log_pred, log_filt, log_p_1, log_smooth, ll = self.lgssm([x, mask])
        
        return q_x, x, log_pred, log_filt, log_p_1, log_smooth, ll, p_y
    
    #@tf.function
    def dec(self, x, s, s_feat, length, training):
        return self.decoder([tf.concat([x, tf.repeat(s[:,None,:], length, axis=1)], axis=2), s_feat], training)

    #@tf.function
    def sim_metric(self, y_true, y_pred, mask_ones, metric, length):
        val = metric(y_true, y_pred) # (bs*length, *dim_y, 1)        
        val = tf.reduce_sum(val, axis=tf.range(1, len(val.shape))) # (bs*length)
        val = tf.reshape(val, (-1, length)) # (bs, length)
        val = tf.multiply(val, mask_ones)  # (bs, length)
        return tf.reduce_sum(val, axis=1) # (bs)

    #@tf.function
    def set_loss(self, y, mask, p_y, q_x, x, log_pred, log_filt, log_p_1, log_smooth, ll):
        length = y.shape[1]
        mask_ones = tf.cast(mask == False, dtype='float32')

        # log p(y|x)        
        log_p_y_x = get_log_dist(p_y, y, mask_ones)       
        
        # log q(x|y)       
        log_q_x_y = get_log_dist(q_x, x, mask_ones)          
        
        # log p(x, z)
        log_p_xz = tf.reduce_sum(log_filt, axis=1) + log_p_1 + tf.reduce_sum(log_pred, axis=1)
        
        # log p(z|x)
        log_p_z_x = tf.reduce_sum(log_smooth, axis=1)
        
        # Image simularity metrics
        y_true = tf.reshape(y, (-1, *y.shape[-2:], 1)) # (bs*length, *dim_y, 1)
        y_pred = tf.reshape(p_y.sample(), (-1, *y.shape[-2:], 1)) # (bs*length, *dim_y, 1)
        
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
        self.log_py_metric.update_state(log_p_y_x)
        self.log_qx_metric.update_state(log_q_x_y)
        self.log_pzx_metric.update_state(log_p_xz)
        self.log_pz_x_metric.update_state(log_p_z_x)
        self.elbo_metric.update_state(elbo)     
        self.log_px_x_metric.update_state(tf.reduce_sum(ll, axis=1))

        y_pred = p_y.sample()
        self.ssim_metric.update_state(ssim)

    def eval(self, inputs):
        y = inputs['input_video']
        y_ref = inputs['input_ref']
        mask = inputs['input_mask'] 
        length = y.shape[1]

        q_x, q_s, s_feat = self.encoder(y, y_ref, training=False)        
        
        x = q_x.sample()
        s = q_s.sampel()
        
        p_dec = self.dec(x, s, s_feat, length, False)     

        p_obssmooth, p_obsfilt, p_obspred = self.lgssm.get_distribtions(x, mask)
        p_y_smooth = self.dec(p_obssmooth.sample(), s, s_feat, length, False)
        p_y_filt = self.dec(p_obsfilt.sample(), s, s_feat, length, False)
        p_y_pred = self.dec(p_obspred.sample(), s, s_feat, length, False)    

        y_vae = p_dec.sample()
        y_smooth = p_y_smooth.sample()
        y_filt = p_y_filt.sample()
        y_pred = p_y_pred.sample()

        return {'image_data': {'vae': {'images' : y_vae},
                        'smooth': {'images': y_smooth},
                        'filt': {'images': y_filt},
                        'pred': {'images': y_pred}},
                'latent_dist': {'smooth': p_obssmooth, 'filt': p_obsfilt, 'pred': p_obspred},
                'x_obs': x,
                's': s,
                's_feat': s_feat}


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

        #out = {self.loss_metric.name: self.loss_metric.result()}
        #out.update({m.name: m.result() for m in self.metrics})    
        return {m.name: m.result() for m in self.metrics}