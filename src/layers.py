import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
tfkl = tf.keras.layers
tfk = tf.keras
tfpl = tfp.layers

class Cnn_block(tfkl.Layer):
    def __init__(self, 
                 filters, 
                 kernel=(3,3), 
                 strides = (2,2), 
                 name='CNN_block', **kwargs):
        super(Cnn_block, self).__init__(name=name, **kwargs)
        self.conv2d = tfkl.Conv2D(filters=filters,
                                       kernel_size=kernel,
                                       strides=strides,
                                       padding='same',
                                       use_bias=True,
                                       kernel_initializer='glorot_normal',
                                       bias_initializer='zeros',
                                       activation=None,
                                       name = name + '_conv')
        self.batchNorm = tfkl.BatchNormalization(momentum=0.99, epsilon=0.001,name = name + '_bn')
        self.leakyRelu = tfkl.LeakyReLU(0.2, name = name + '_leakyRelu')
    def call(self, x):
        x = self.conv2d(x)
        x = self.batchNorm(x)
        x = self.leakyRelu(x)
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
        self.convT = tfkl.Conv2D(filters=filters,
                                   kernel_size=kernel,
                                   strides=strides,
                                   padding='same',
                                   use_bias=True,
                                   kernel_initializer='glorot_normal',
                                   bias_initializer='zeros',
                                   activation=None,
                                   name = name+'_convT')
        self.batchNorm = tfkl.BatchNormalization(momentum=0.99, epsilon=0.001,name = name+'_bn')
        self.leakyReLU = tfkl.LeakyReLU(0.2, name = name+'_leakyRelu')
             
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
        x = self.batchNorm(x)
        x = self.leakyReLU(x)
        x = CnnT_block.subpixel_reshape(x, self.factor)
        return x

class Fc_block(tfkl.Layer):
    def __init__(self, h, w, c, name='FC_block', **kwargs):
        super(Fc_block, self).__init__(name=name, **kwargs)
        self.dense = tfkl.Dense(h*w*c, 
                                use_bias=True,
                                kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                name = name + '_dense')
        self.reshape =  tfkl.Reshape((h,w,c), name=name + '_reshape')
        
    def call(self, x):
        x = self.dense(x)
        x = self.reshape(x)
        return x

class Encoder(tfk.Model):
    def __init__(self, config, unet=False, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dim_x = config.dim_x
        self.dim_y = config.dim_y
        self.filters = config.filters
        self.activation_fn = config.activation
        self.unet = unet
        
        self.cnn_blocks = []
        for i in range(len(self.filters)):
            self.cnn_blocks.append(Cnn_block(filters = self.filters[i], 
                                             kernel = config.filter_size,
                                             strides = (2,2),
                                             name="{0}_Cnn_block{1}".format(self.name, i)))
        
        self.flatten = tfkl.Flatten(name='{0}_Flatten'.format(self.name))

        encoder_dist = tfpl.IndependentNormal
        self.mu_layer = tfkl.Dense(self.dim_x, name='{0}_Dense_mu'.format(self.name))
        self.sigma_layer = tfkl.Dense(encoder_dist.params_size(self.dim_x) - self.dim_x, activation='softplus', name='{0}_Dense_sigma'.format(self.name))
        self.dist = encoder_dist(self.dim_x)
    
    def call(self, y):
        ph_steps = tf.shape(y)[1]
        x = tf.reshape(y, (-1, *self.dim_y, 1)) # (b,s,h,w) -> (b*s, h, w, 1)
        if self.unet:
            outs = []
        
        for l in self.cnn_blocks:
            x = l(x)
            if self.unet:
                outs.append(x)
        
        x = self.flatten(x)

        mu = tf.reshape(self.mu_layer(x), (-1, ph_steps, self.dim_x))
        # if variance is zero, log_p(x) will be NaN and gradients will be NaN, therefore adding epsilson + softmax
        sigma = tf.reshape(self.sigma_layer(x), (-1, ph_steps, self.dim_x)) + tf.keras.backend.epsilon()
        
        event = tf.concat([mu, sigma], axis=-1)
        q_x_y = self.dist(event)

        if self.unet:
            return q_x_y, outs
        return q_x_y
    
class Decoder(tfk.Model):
    def __init__(self, config, output_channels, unet=False, name='Decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.output_channels = output_channels
        self.unet = unet
        self.dim_y = config.dim_y
        self.dim_x = config.dim_x if not unet else config.dim_x * 2
        self.filters = config.filters
                                   
        activation_y_mu = None # For Gaussian
        
        h = int(config.dim_y[0] / (2**(len(self.filters))))
        w = int(config.dim_y[1] / (2**(len(self.filters))))
        self.fc_block = Fc_block(h,w,self.filters[-1], name='{0}_FC_block'.format(self.name))
        self.cnnT_blocks = []
        for i in reversed(range(len(self.filters))):
            self.cnnT_blocks.append(CnnT_block(filters = self.filters[i]*2 if self.unet else self.filters[i]*4, 
                                               factor = 2,
                                               kernel = config.filter_size,
                                               strides = (1,1),
                                               name="{0}_CnnT_block{1}".format(self.name, i)))
        self.cnnT_block_last = tfkl.Conv2D(filters=self.output_channels,
                                                    kernel_size=1,
                                                    strides=1,
                                                    padding='same',
                                                    use_bias=True,
                                                    kernel_initializer='glorot_normal',
                                                    bias_initializer='zeros',
                                                    activation=None,
                                                    name='{0}_Y_last'.format(self.name)
                                                   )
        self.mu_activation = tfkl.Activation(activation_y_mu)
        self.flatten = tfkl.Flatten(name='{0}_Flatten'.format(self.name))
        decoder_dist = tfpl.IndependentNormal
        if self.output_channels > 1:
            self.dist = decoder_dist((*self.dim_y,self.output_channels))
        else:
            self.dist = decoder_dist(self.dim_y)
        
    def call(self, inputs):
        if self.unet:
            x = inputs[0]
            feats = inputs[1]
        else:
            x = inputs
        
        ph_steps = tf.shape(x)[1]
        bs = tf.shape(x)[0]
        x = tf.reshape(x, (-1, self.dim_x)) # (b,s,latent_dim) -> (b*s, latent_dim)
        x = self.fc_block(x)
        i = 1
        for l in self.cnnT_blocks:
            if self.unet:
                f = feats[-i][:,None,...]
                f = tf.reshape(tf.repeat(f, ph_steps, axis=1), (-1, *f.shape[2:]))
                x = tf.concat([x, f], axis=-1)
            x = l(x)          
            i += 1
        
        y = self.cnnT_block_last(x) # (bs*s, h, w, 2)
        y_mu = self.mu_activation(y)
        y_sigma = tf.ones_like(y_mu, dtype='float32') * 0.01
        
        y_mu = tf.reshape(y_mu, (bs, ph_steps, *self.dim_y, -1)) # (b*s,dim_y, dim_y, c) -> (b,s,dim_y, dim_y, c)
        y_sigma = tf.reshape(y_sigma, (bs, ph_steps, *self.dim_y,-1))  + tf.keras.backend.epsilon()
        #if self.output_channels == 1:
        #    y_mu = y_mu[...,0] # (b*s,dim_y, dim_y, 1) -> (b,s,dim_y, dim_y)
        #    y_sigma = y_sigma[...,0]

        y_mu = tf.reshape(y_mu, (-1, ph_steps, np.prod(self.dim_y)*self.output_channels))
        y_sigma = tf.reshape(y_sigma, (-1, ph_steps, np.prod(self.dim_y)*self.output_channels))

        p_y_x = self.dist(tf.concat([y_mu, y_sigma], axis=-1))
        return p_y_x
