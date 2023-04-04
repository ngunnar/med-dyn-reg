import tensorflow as tf
import tensorflow_probability as tfp

from .layers_skip import Encoder, Decoder
from .utils import ssim_calculation, set_name

tfd = tfp.distributions
tfk = tf.keras
tfpl = tfp.layers

class VAE(tfk.Model):
    def __init__(self, 
                 config,
                 name="vae",
                 prefix=None,
                 **kwargs):
        super(VAE, self).__init__(name=set_name(name, prefix), **kwargs)
        self.config = config
        
        self.encoder = Encoder(self.config, prefix=prefix)
        self.decoder = Decoder(self.config, output_channels = 1, prefix=prefix)

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self.config.dim_x), scale=1.), 
                                reinterpreted_batch_ndims=1)

        self.loss_metric = tfk.metrics.Mean(name=set_name("loss", prefix))
        self.log_py_metric = tfk.metrics.Mean(name = set_name('log p(y|x) ↑', prefix))
        self.kld_metric = tfk.metrics.Mean(name = set_name('KLD ↓', prefix))
        self.elbo_metric = tfk.metrics.Mean(name = set_name('ELBO ↑', prefix))
        self.ssim_metric = tfk.metrics.Mean(name = set_name('ssim ↑', prefix))

    def call(self, inputs, training=None):
        y = inputs['input_video']
        y_ref = inputs['input_ref']
        mask = inputs['input_mask']   
        p_y, q_x, x = self.forward(y, y_ref, training)        
        self.set_loss(y, mask, p_y, q_x)
        return p_y, q_x, x
    
    def forward(self, y, y_ref, training=None):
        q_x, q_s, s_feat = self.encoder(y, y_ref, training)        
        x = q_x.sample()        
        p_y = self.decoder([x, s_feat], training)        
        
        return p_y, q_x, x

    def set_loss(self, y, mask, p_y, q_x):
        mask_ones = tf.cast(mask == False, dtype='float32') # (bs, length)    
        log_p_y = p_y.log_prob(y) # (bs, length)
        log_p_y = tf.multiply(log_p_y, mask_ones) 
        log_p_y = tf.reduce_sum(log_p_y, axis=1)

        kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        kld = kl(q_x.sample(), self.prior.sample()) #(bs, length)
        kld = tf.multiply(kld, mask_ones)
        kld = tf.reduce_sum(kld, axis=1)

        elbo = log_p_y - kld
        loss = -log_p_y + kld
        
        # LOSS
        self.add_loss(loss) # on batch level

        # METRICES
        self.log_py_metric.update_state(log_p_y)
        self.kld_metric.update_state(kld)
        self.elbo_metric.update_state(elbo)
        
        y_pred = log_p_y.sample()
        self.ssim_metric.update_state(ssim_calculation(y, y_pred))

    def eval(self, inputs):
        y = inputs['input_video']
        mask = inputs['input_mask'] 
        p_y, q_x, x = self.forward(y, False)
        y_vae = p_y.sample()

        return {'image_data': {'vae': {'images' : y_vae}},
                'x_obs': x}

    def sample(self, y):
        x = self.prior((tf.shape(y)[0:2]))
        p_y = self.decoder(x, training=False)
        return p_y

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