import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa

import numpy as np
tfkl = tf.keras.layers
tfk = tf.keras
tfpl = tfp.layers
tfd = tfp.distributions

class Cnn_block(tfkl.Layer):
    def __init__(self, 
                 filters, 
                 kernel=(3,3), 
                 strides = (2,2), 
                 activation=None,
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
        self.activation = tfkl.Activation(activation)
    def call(self, x, training):
        x = self.conv2d(x)
        x = self.batchNorm(x, training)
        x = self.activation(x)
        return x
    
class downsample_block(tfkl.Layer):
    def __init__(self, filters, kernel=(3,3), name='down_block', **kwargs):
        super(downsample_block, self).__init__(name=name, **kwargs)
        self.down_cnn = Cnn_block(filters = filters, 
                                  kernel = kernel,
                                  strides = (2,2),
                                  activation = tfkl.LeakyReLU(0.2, name = name + '_leakyRelu_down'),
                                  name="{0}_Cnn_block1".format(self.name))
        self.cnn1 = Cnn_block(filters = filters, 
                              kernel = kernel,
                              strides = (1,1),
                              activation = tfkl.LeakyReLU(0.2, name = name + '_leakyRelu'),
                              name="{0}_Cnn_block2".format(self.name))
        self.cnn2 = Cnn_block(filters = filters, 
                              kernel = kernel,
                              strides = (1,1),
                              name="{0}_Cnn_block3".format(self.name))
        self.add = tfkl.Add(name=self.name + '_add')
        self.activation = tfkl.LeakyReLU(0.2, name = self.name + '_leakyReluOut') 
    def call(self, x, traning):
        x = self.down_cnn(x, traning)
        out = self.cnn2(self.cnn1(x, traning), traning)
        out = self.add([x, out])
        out = self.activation(out)
        return out
    
class upsample_block(tfkl.Layer):
    def __init__(self, filters, kernel=(3,3), name='up_block', **kwargs):
        super(upsample_block, self).__init__(name=name, **kwargs)
        
        self.convT = CnnT_block(filters = filters,
                                use_subpixel=False,
                                kernel = kernel,
                                name="{0}_CnnT_block".format(self.name))
        self.cnn1 = Cnn_block(filters = filters, 
                              kernel = kernel,
                              strides = (1,1),
                              activation = tfkl.LeakyReLU(0.2, name = name + '_leakyRelu'),
                              name="{0}_Cnn_block1".format(self.name))
        self.cnn2 = Cnn_block(filters = filters, 
                              kernel = kernel,
                              strides = (1,1),
                              name="{0}_Cnn_block2".format(self.name))
        
        self.activation = tfkl.LeakyReLU(0.2, name = self.name + '_leakyReluOut') 
        self.add = tfkl.Add(name=self.name+ '_add')
    
    def call(self, x, traning):
        x = self.convT(x, traning)
        out = self.cnn2(self.cnn1(x, traning), traning)
        out = self.add([x, out])
        out = self.activation(out)
        return out
        
class CnnT_block(tf.keras.layers.Layer):
    def __init__(self, 
                 filters,
                 factor = 2,
                 kernel=(3,3),
                 strides=(2,2),
                 use_subpixel=False,
                 momentum=0.99,
                 epsilon=0.001,
                 name='CnnT_block', **kwargs):
        super(CnnT_block, self).__init__(name=name, **kwargs)
        self.use_subpixel = use_subpixel
        if self.use_subpixel:
            self.factor = factor
            self.convT = tfkl.Conv2D(filters=filters,
                                   kernel_size=kernel,
                                   strides=(1,1),
                                   padding='same',
                                   use_bias=True,
                                   kernel_initializer='glorot_normal',
                                   bias_initializer='zeros',
                                   activation=None,
                                   name = name+'_convT')
        else:
            self.convT = tfkl.Conv2DTranspose(filters=filters,
                                       kernel_size=kernel,
                                       strides=(2,2),
                                       padding='same',
                                       use_bias=True,
                                       kernel_initializer='glorot_normal',
                                       bias_initializer='zeros',
                                       activation=None,
                                       name = name+'_convT')
            
        self.batchNorm = tfkl.BatchNormalization(momentum=momentum, epsilon=epsilon, name = name+'_bn')
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
    def call(self, x, training):
        x = self.convT(x)
        x = self.batchNorm(x,training)
        x = self.leakyReLU(x)
        if self.use_subpixel:
            return CnnT_block.subpixel_reshape(x, self.factor)
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
    def __init__(self, config, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dim_x = config.dim_x
        self.dim_y = config.dim_y
        self.filters = config.enc_filters
        
        self.down_blocks = []
        for i in range(len(self.filters)):
            self.down_blocks.append(
                downsample_block(filters=self.filters[i], kernel=config.filter_size, name='{0}_down_block{1}'.format(self.name, i)))
        
        self.flatten = tfkl.Flatten(name='{0}_Flatten'.format(self.name))
        
        self.use_dist = tfpl.IndependentNormal
        #use_dist = tfpl.MultivariateNormalTriL
        self.dense = tfkl.Dense(self.use_dist.params_size(self.dim_x), activation=None)
        
        activity_regularizer = None
        
        self.enc_q = self.use_dist(self.dim_x, activity_regularizer = activity_regularizer, name='encoder_dist')
        
    def call(self, y, training):
        ph_steps = tf.shape(y)[1]
        x = tf.reshape(y, (-1, *self.dim_y, 1)) # (b,s,h,w) -> (b*s, h, w, 1)
        
        for l in self.down_blocks:
            x = l(x, training)
        
        x = self.flatten(x)
        x = self.dense(x)
        x = tf.reshape(x, (-1, ph_steps, self.use_dist.params_size(self.dim_x)))
        q_enc = self.enc_q(x)
        
        return q_enc
    
class Decoder(tfk.Model):
    def __init__(self, config, output_channels, name='Decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.output_channels = output_channels
        self.dim_y = config.dim_y
        self.dim_x = config.dec_input_dim
        self.filters = config.dec_filters
                                           
        h = int(config.dim_y[0] / (2**(len(self.filters))))
        w = int(config.dim_y[1] / (2**(len(self.filters))))
        self.fc_block = Fc_block(h,w,self.filters[-1], name='{0}_FC_block'.format(self.name))
        self.up_blocks = []
        for i in reversed(range(len(self.filters))):
            self.up_blocks.append(
                upsample_block(filters = self.filters[i], kernel=config.filter_size, name='{0}_up_block{1}'.format(self.name, i)))
        
        self.last_cnn = tfkl.Conv2D(filters=self.output_channels,
                                   kernel_size=config.filter_size,
                                   strides=(1,1),
                                   padding='same',
                                   use_bias=True,
                                   kernel_initializer='glorot_normal',
                                   bias_initializer='zeros',
                                   activation=None,
                                   name = name+'_lastCNN')
        
        #self.dec_p = tfpl.DistributionLambda(lambda d: tfd.Independent(tfd.Normal(loc = d[0], scale = d[1])))
        decoder_dist = tfpl.IndependentNormal
        if output_channels > 1:
            self.decoder_dist = decoder_dist((*config.dim_y,output_channels), name='decoder_dist')
        else:
            self.decoder_dist = decoder_dist(config.dim_y, name='decoder_dist')

    def call(self, inputs, training):
        x = inputs
        
        ph_steps = tf.shape(x)[1]
        bs = tf.shape(x)[0]
        x = tf.reshape(x, (-1, self.dim_x)) # (b,s,latent_dim) -> (b*s, latent_dim)
        x = self.fc_block(x)
        i = 1
        for l in self.up_blocks:
            x = l(x, training)
            i += 1
        
        y_mu = self.last_cnn(x)        
        # softplus is used so a large negative number gives #0.01
        y_mu = tf.reshape(y_mu, (-1, ph_steps, np.prod(self.dim_y)*self.output_channels))
        y_sigma = tf.ones_like(y_mu, dtype='float32') * tfp.math.softplus_inverse(0.01)        

        p_dec = self.decoder_dist(tf.concat([y_mu, y_sigma], axis=-1))
        return p_dec