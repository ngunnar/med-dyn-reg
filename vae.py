import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import numpy as np
from losses import loss_function_vae

class Cnn_block(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 #activation, 
                 kernel=(3,3), 
                 strides = (2,2), 
                 name='CNN_block', **kwargs):
        super(Cnn_block, self).__init__(name=name, **kwargs)
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=kernel,
                                       strides=strides,
                                       padding='same',
                                       use_bias=True,
                                       kernel_initializer='glorot_normal',
                                       bias_initializer='zeros',
                                       activation=None),
            tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            tf.keras.layers.LeakyReLU(0.2)])
    def call(self, x):
        x = self.conv(x)
        return x

class CnnT_block(tf.keras.layers.Layer):
    def __init__(self, 
                 filters,
                 factor = 2,
                 kernel=(5,5),
                 strides=(2,2),
                 #activation=None,
                 name='CnnT_block', **kwargs):
        super(CnnT_block, self).__init__(name=name, **kwargs)
        self.factor = factor
        self.convT = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=kernel,
                                   strides=strides,
                                   padding='same',
                                   use_bias=True,
                                   kernel_initializer='glorot_normal',
                                   bias_initializer='zeros',
                                   activation=None),
            tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            tf.keras.layers.LeakyReLU(0.2)])
    
    def subpixel_reshape(x, factor):
        # input and output shapes
        bs, ih, iw, ic = x.get_shape().as_list()
        oh, ow, oc = ih * factor, iw * factor, ic // factor ** 2

        assert ic % factor == 0, "Number of input channels must be divisible by factor"

        intermediateshp = (-1, iw, iw, oc, factor, factor)
        x = tf.reshape(x, intermediateshp)
        x = tf.transpose(x, (0, 1, 4, 2, 5, 3))
        x = tf.reshape(x, (-1, oh, ow, oc))
        return x
    def call(self, x):
        x = self.convT(x)
        x = CnnT_block.subpixel_reshape(x, self.factor)
        return x

class Fc_block(tf.keras.layers.Layer):
    def __init__(self, h, w, c, name='fc_block', **kwargs):
        super(Fc_block, self).__init__(name=name, **kwargs)
        self.l = tf.keras.Sequential([
            tf.keras.layers.Dense(h*w*c, 
                                  use_bias=True,
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer='zeros'),
            tf.keras.layers.Reshape((h,w,c))])
        
    def call(self, x):
        x = self.l(x)
        return x

class Sampler(tf.keras.layers.Layer):
    def call(self, inputs):
        mu = inputs[0]
        logvar = inputs[1]
        shape = tf.shape(mu)
        bs = shape[0]
        dim = shape[1]
        eps = tf.random.normal(shape=(bs,dim))
        std_dev = tf.math.sqrt(tf.math.exp(logvar))
        return mu + std_dev*eps
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, config, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dim_x = config.dim_x
        self.dim_y = config.dim_y
        self.filters = config.filters
        self.noise_emission = config.noise_emission
        self.activation_fn = config.activation
        
        
        self.cnn_blocks = []
        for i in range(len(self.filters)):
            self.cnn_blocks.append(Cnn_block(filters = self.filters[i], 
                                             kernel = config.filter_size,
                                             strides = (2,2),
                                             name="Cnn_block{0}".format(i)))
        
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.dense_mu = tf.keras.layers.Dense(self.dim_x, name='dense_mu')
        self.dense_logvar = tf.keras.layers.Dense(self.dim_x, name='dense_logvar')
        self.sampler = Sampler()
    
    def call(self, x):
        ph_steps = tf.shape(x)[1]
        x = tf.reshape(x, (-1, *self.dim_y, 1)) # (b,s,h,w) -> (b*s, h, w, 1)
        for l in self.cnn_blocks:
            x = l(x)
        
        x = self.flatten(x)
        
        x_mu = self.dense_mu(x)
        x_logvar = self.dense_logvar(x) + tf.math.log(self.noise_emission)
        
        x = tf.reshape(self.sampler([x_mu, x_logvar]), (-1, ph_steps, self.dim_x))
        return x, x_mu, x_logvar
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, config, name='Decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dim_y = config.dim_y
        self.dim_x = config.dim_x
        self.est_logvar = config.est_logvar
        self.filters = config.filters
                                   
        activation_y_mu = None # For Gaussian
        
        h = int(config.dim_y[0] / (2**(len(self.filters))))
        w = int(config.dim_y[1] / (2**(len(self.filters))))
        self.fc_block = Fc_block(h,w,self.filters[-1], name='fc_block')
        self.cnnT_blocks = []
        for i in reversed(range(len(self.filters))):
            self.cnnT_blocks.append(CnnT_block(filters = self.filters[i]*4, 
                                               factor = 2,
                                               kernel = config.filter_size,
                                               strides = (1,1),
                                               name="CnnT_block{0}".format(i)))
        
        if self.est_logvar:
            f = 2
            self.sampler = Sampler()
        else:
            f = 1
        self.cnnT_block_last = tf.keras.layers.Conv2D(filters=f,
                                                    kernel_size=1,
                                                    strides=1,
                                                    padding='same',
                                                    use_bias=True,
                                                    kernel_initializer='glorot_normal',
                                                    bias_initializer='zeros',
                                                    activation=None,
                                                    name='y_last'
                                                   )
        self.mu_activation = tf.keras.layers.Activation(activation_y_mu)
        
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, x):
        ph_steps = tf.shape(x)[1]
        x = tf.reshape(x, (-1, self.dim_x)) # (b,s,latent_dim) -> (b*s, latent_dim)
        x = self.fc_block(x)
        for l in self.cnnT_blocks:
            x = l(x)
        
        y = self.cnnT_block_last(x) # (bs*s, h, w, 2)
        if self.est_logvar:
            y_mu = self.mu_activation(y[...,0])
            y_mu = self.flatten(y_mu) # (bs*s, h * w)
            y_logvar = y[...,1] # (bs*s, h, w)     
            y_logvar = self.flatten(y_logvar) # (bs*s, h * w)
            y = tf.reshape(self.sampler([y_mu, y_logvar]), (-1, ph_steps, *self.dim_y)) # (bs, s, h, w)
        else:
            y_mu = self.mu_activation(y)
            y_mu = self.flatten(y_mu) # (bs*s, h * w)
            y_logvar = tf.zeros_like(y_mu, dtype='float32')
            y = tf.reshape(y_mu, (-1, ph_steps, *self.dim_y))
        
        return y, y_mu, y_logvar
    
    
class VAE(tf.keras.Model):
    def __init__(self, config, name="autoencoder", **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.config = config
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
    
    def call(self, inputs):
        y = inputs[0]
        mask = inputs[1]
        x, x_mu, x_logvar = self.encoder(y)
        y_hat, y_mu, y_logvar = self.decoder(x)
        
        return y_hat, y_mu, y_logvar, x, x_mu, x_logvar
    
    def predict(self, inputs):
        y_true = inputs[0]
        mask = inputs[1]
        x_vae, x_mu, x_logvar = self.encoder(y_true)
        y_hat, y_mu, y_logvar = self.decoder(x_vae)
        
        return y_hat
    
    def compile(self, num_batches):
        super(VAE, self).compile()
        self.num_batches = num_batches
        self.epoch = 1
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.config.init_lr, 
                                                                     decay_steps=self.config.decay_steps*num_batches, 
                                                                     decay_rate=self.config.decay_rate, 
                                                                     staircase=True)
        self.opt = tf.keras.optimizers.Adam(lr_schedule)  
    
    def train_step(self, y_true, mask):
        w_recon = self.config.scale_reconstruction
        w_kl = self.config.kl_latent_loss_weight * tf.sigmoid((self.epoch - 1)**2/self.config.kl_growth-self.config.kl_growth)
        with tf.GradientTape() as tape:
            y_hat, y_mu, y_logvar, x_vae, x_mu, x_logvar = self([y_true, mask])
            loss_sum, recon_loss, kl_loss = loss_function_vae(self.config, y_true, mask, y_hat, y_mu, y_logvar, x_vae, x_mu, x_logvar)
            loss = recon_loss * w_recon
            loss += kl_loss * w_kl       
            variables = self.trainable_variables
        
        gradients = tape.gradient(loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
        self.opt.apply_gradients(zip(gradients, variables))
        return loss_sum, loss, recon_loss, w_recon, kl_loss, w_kl    
    
    def test_step(self, y_true, mask):
        y_hat, y_mu, y_logvar, x_vae, x_mu, x_logvar = self([y_true, mask], training=False)
        return loss_function_vae(self.config, y_true, mask, y_hat, y_mu, y_logvar, x_vae, x_mu, x_logvar)  
    
    def model(self):
        inputs = tf.keras.layers.Input(shape=(self.config.dim_y))
        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))
    
    def info(self):
        x = tf.keras.layers.Input(shape=(self.config.ph_steps, *self.config.dim_y))        
        vae = tf.keras.Model(inputs=[x], outputs=self.call(x))
        vae.summary()
        
        encoder = tf.keras.Model(inputs=[x], outputs=self.encoder.call(x))
        encoder.summary()
        
        x_decoder = tf.keras.layers.Input(shape=(self.config.ph_steps, self.config.dim_x))
        decoder = tf.keras.Model(inputs=[x_decoder], outputs=self.decoder.call(x_decoder))
        decoder.summary()