import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa

import numpy as np
tfkl = tf.keras.layers
tfk = tf.keras
tfpl = tfp.layers

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
        
class Subpixel_CNN(tf.keras.layers.Layer):
    def __init__(self, 
                 filters,
                 factor = 2,
                 kernel=(3,3),
                 name='subpixel_cnn', **kwargs):
        super(Subpixel_CNN, self).__init__(name=name, **kwargs)
        self.factor = factor
        self.subpixelCNN = tfkl.Conv2D(filters=filters,
                                   kernel_size=kernel,
                                   strides=(1,1),
                                   padding='same',
                                   use_bias=True,
                                   kernel_initializer='glorot_normal',
                                   bias_initializer='zeros',
                                   activation=None,
                                   name = name+'_subpixelCNN')         
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
        x = self.subpixelCNN(x)
        return Subpixel_CNN.subpixel_reshape(x, self.factor)
    
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
        self.activation_fn = config.activation
        
        self.down_blocks = []
        for i in range(len(self.filters)):
            self.down_blocks.append(
                downsample_block(filters=self.filters[i], kernel=config.filter_size, name='{0}_down_block{1}'.format(self.name, i)))
        
        self.flatten = tfkl.Flatten(name='{0}_Flatten'.format(self.name))
        
        self.mu_layer = tfkl.Dense(self.dim_x, name='{0}_Dense_mu'.format(self.name))
        activation='softplus' # IndependentNormal uses 'softplus' already
        self.sigma_layer = tfkl.Dense(self.dim_x, activation=activation, name='{0}_Dense_sigma'.format(self.name))
        
    def call(self, y, training):
        ph_steps = tf.shape(y)[1]
        x = tf.reshape(y, (-1, *self.dim_y, 1)) # (b,s,h,w) -> (b*s, h, w, 1)
        
        for l in self.down_blocks:
            x = l(x, training)
        
        x = self.flatten(x)

        mu = tf.reshape(self.mu_layer(x), (-1, ph_steps, self.dim_x))
        sigma = tf.reshape(self.sigma_layer(x), (-1, ph_steps, self.dim_x))
        # if variance is zero, log_p(x) will be NaN and gradients will be NaN, therefore adding epsilson
        sigma = tfp.math.softplus_inverse(sigma + tf.keras.backend.epsilon())

        return mu, sigma
    
class Decoder(tfk.Model):
    def __init__(self, config, output_channels, name='Decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.output_channels = output_channels
        self.dim_y = config.dim_y
        self.dim_x = config.dec_input_dim
        self.filters = config.dec_filters
                                   
        activation_y_mu = None #None For Gaussian
        
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

        self.mu_activation = tfkl.Activation(activation_y_mu)
    
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
        
        y = self.last_cnn(x)
         
        y_mu = self.mu_activation(y)           
        # softplus is used so a large negative number gived #0.01
        y_sigma = tf.ones_like(y_mu, dtype='float32') * tfp.math.softplus_inverse(0.01) 

        y_mu = tf.reshape(y_mu, (-1, ph_steps, np.prod(self.dim_y)*self.output_channels))
        y_sigma = tf.reshape(y_sigma, (-1, ph_steps, np.prod(self.dim_y)*self.output_channels))

        return y_mu, y_sigma

    
class Bspline(tf.keras.layers.Layer):
    def __init__(self, config, name='Bspline', **kwargs):
        super(Bspline,self).__init__(name=name, **kwargs)
        self.dim_x = config.dim_x
        self.dim_y = config.dim_y
        self.dist = tfpl.IndependentNormal((*self.dim_y,2))
        self.dim = 32
        self.filters = config.dec_filters
        self.output_channels = 2
        
        y_range = tf.range(start=0, limit=self.dim_y[0])/self.dim_y[0]
        x_range = tf.range(start=0, limit=self.dim_y[1])/self.dim_y[1]
        y_grid, x_grid = tf.meshgrid(y_range, x_range, indexing="ij")
        self.grid = tf.cast(tf.stack((y_grid, x_grid), -1), 'float32')
        
        
        
        h = int(self.dim / (2**3))
        w = int(self.dim / (2**3))
        self.fc_block = Fc_block(h,w,self.filters[-1], name='{0}_FC_block'.format(self.name))
        self.up_blocks = []
        for i in reversed(range(3)):
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
        self.interp = tf.keras.layers.Lambda(lambda x: self.interpolation(x))
        
    def body(i, d, p, s):
        d_new = tfa.image.interpolate_bilinear(p[...,i:i+1], s)
        return (i+1, tf.concat([d,d_new], axis=-1), p, s)
    
    def interpolation(self, inputs):
        parameters = inputs[0]
        bs = inputs[1]
        ph_steps = inputs[2]
        transform_grid = tf.repeat(self.grid[None,...], bs*ph_steps, axis=0)
        transform_grid = tf.reshape(transform_grid, (bs*ph_steps, -1, self.grid.shape[-1]))
        
        scaled_points = transform_grid * (np.array([self.dim, self.dim], dtype='float32') - 1)[None,None,...]
        
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
        return d
    
    def call(self, inputs, training):
        parameters = inputs
        bs = tf.shape(parameters)[0]
        ph_steps = parameters.shape[1]
        
        
        parameters = tf.reshape(parameters, (bs*ph_steps, -1))
        parameters = self.fc_block(parameters)
        for l in self.up_blocks:
            parameters = l(parameters, training)
        
        parameters = self.last_cnn(parameters)
        d = self.interp([parameters, bs, ph_steps])
        flow = -d*np.asarray(self.dim_y)[None, None,...]
        flow = tf.reshape(flow, (bs, ph_steps, *self.dim_y, 2))
        mu = flow
        sigma = tf.ones_like(mu, dtype='float32') * tfp.math.softplus_inverse(0.01) # softplus is used so a -4.6 approx std 0.01

        mu = tf.reshape(mu, (-1, mu.shape[1], np.prod(mu.shape[2:])))
        sigma = tf.reshape(sigma, (-1, mu.shape[1], np.prod(mu.shape[2:])))
        phi_y_x = self.dist(tf.concat([mu, sigma], axis=-1))
        
        return phi_y_x