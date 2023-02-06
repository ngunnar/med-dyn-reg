import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from .layers_skip import Encoder, Decoder
from .utils import ssim_calculation

tfd = tfp.distributions
tfk = tf.keras
tfpl = tfp.layers

class VAE(tfk.Model):
    def __init__(self, 
                 config,
                 name="vae",
                 **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.config = config
        
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config, output_channels = 1)

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self.config.dim_x), scale=1.), 
                                reinterpreted_batch_ndims=1)

        self.loss_metric = tfk.metrics.Mean(name="loss")
        self.log_pdec_metric = tfk.metrics.Mean(name = 'log p(y|x) ↑')        
        self.kld_metric = tfk.metrics.Mean(name = 'KLD ↓')
        self.elbo_metric = tfk.metrics.Mean(name = 'ELBO ↑')
        self.ssim_metric = tfk.metrics.Mean(name = 'ssim ↑')

    def call(self, inputs, training=None):
        y = inputs['input_video']
        mask = inputs['input_mask']   
        p_dec, q_enc, x_sample = self.forward(y, training)        
        self.set_loss(y, mask, p_dec, q_enc)
        return p_dec, q_enc, x_sample
    
    def forward(self, y, training=None):
        q_enc, x_ref_feat = self.encoder(y, training)        
        x_sample = q_enc.sample()        
        p_dec = self.decoder([x_sample, x_ref_feat], training)        
        
        return p_dec, q_enc, x_sample

    def set_loss(self, y, mask, p_dec, q_enc):
        mask_ones = tf.cast(mask == False, dtype='float32') # (bs, ph_steps)    
        log_p_dec = p_dec.log_prob(y) # (bs, ph_steps)
        log_p_dec = tf.multiply(log_p_dec, mask_ones) 
        log_p_dec = tf.reduce_sum(log_p_dec, axis=1)

        kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        kld = kl(q_enc.sample(), self.prior.sample()) #(bs, ph_steps)
        kld = tf.multiply(kld, mask_ones)
        kld = tf.reduce_sum(kld, axis=1)

        elbo = log_p_dec - kld
        loss = -log_p_dec + kld
        
        # LOSS
        self.add_loss(loss) # on batch level

        # METRICES
        self.log_pdec_metric.update_state(log_p_dec)
        self.kld_metric.update_state(kld)
        self.elbo_metric.update_state(elbo)
        
        y_pred = p_dec.sample()
        self.ssim_metric.update_state(ssim_calculation(y, y_pred))

    def eval(self, inputs):
        y = inputs['input_video']
        mask = inputs['input_mask'] 
        p_dec, q_enc, x_sample = self.forward(y, False)
        y_vae = p_dec.sample()

        return {'image_data': {'vae': {'images' : y_vae}},
                'x_obs': x_sample}

    def sample(self, y):
        x_sample = self.prior((tf.shape(y)[0:2]))
        p_dec = self.decoder(x_sample, training=False)
        return p_dec

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
        loss = self.losses
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))      
        self.loss_metric.update_state(loss)

        out = {self.loss_metric.name: self.loss_metric.result()}
        out.update({m.name: m.result() for m in self.metrics})
        return out