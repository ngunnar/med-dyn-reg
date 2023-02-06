import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
tfkl = tf.keras.layers
tfk = tf.keras
tfpl = tfp.layers
tfd = tfp.distributions

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

    
class LK_encoder(tfkl.Layer):
    def __init__(self, filters, kernel, name, **kwargs):
        super(LK_encoder, self).__init__(name=name, **kwargs)
        self.cnn = tfkl.Conv2D(filters=filters,
                               kernel_size=kernel, 
                               padding='same', 
                               kernel_initializer='glorot_normal')
        self.batch_norm = tfkl.BatchNormalization(momentum=0.99, epsilon=0.001,name = name + '_bn')
        
    def call(self, x, training):
        x = self.cnn(x)
        x = self.batch_norm(x, training)
        return x

class EncoderMiniBlock(tfkl.Layer):
    def __init__(self, filters, kernel=(3,3), name='down_block', dropout_prob=0.0, **kwargs):
        super(EncoderMiniBlock, self).__init__(name=name, **kwargs)
        self.conv1 = tfkl.Conv2D(filters=filters,
                                 kernel_size=kernel,
                                 activation=tfkl.LeakyReLU(0.2, name = name + '_leakyRelu'),
                                 padding='same',
                                 kernel_initializer='glorot_normal')
        
        # LK
        self.regular_kernel = LK_encoder(filters=filters, 
                                          kernel=kernel, 
                                          name = name + '_regular_kernel')
        self.large_kernel = LK_encoder(filters=filters, 
                                          kernel=(5,5), 
                                          name = name + '_large_kernel')
        self.one_kernel = LK_encoder(filters=filters, 
                                          kernel=(1,1), 
                                          name = name + '_one_kernel')
        
        self.activation_fn = tfkl.LeakyReLU(0.2, name = name + '_leakyRelu')
        
        self.batch_normalization = tfkl.BatchNormalization(momentum=0.99, epsilon=0.001,name = name + '_bn')
        
        self.drop_out = tf.keras.layers.Dropout(dropout_prob)
        self.down_cnn = tfkl.Conv2D(filters = filters, 
                                    kernel_size=kernel, 
                                    strides = (2,2), 
                                    activation=tfkl.LeakyReLU(0.2, name = name + '_leakyRelu_down'), 
                                    padding='same', 
                                    kernel_initializer='glorot_normal')

    def call(self, x, training):
        x = self.conv1(x)
        x_rk = self.regular_kernel(x)
        x_lk = self.large_kernel(x)
        x_ok = self.one_kernel(x)
        x = x + x_rk + x_lk + x_ok
        x = self.activation_fn(x)
        
        #x = self.conv2(x)
        x = self.batch_normalization(x, training)
        down_x = self.drop_out(x)
        down_x = self.down_cnn(down_x)
        return x, down_x

class DecoderMiniBlock(tfkl.Layer):
    def __init__(self, skip_connection, filters, kernel=(3,3), name='up_block', dropout_prob=0.0, **kwargs):
        super(DecoderMiniBlock, self).__init__(name=name, **kwargs)
        self.skip_connection = skip_connection
        self.up_cnnT = tfkl.Conv2DTranspose(filters=filters, kernel_size=kernel, strides = (2,2), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.conv1 = tfkl.Conv2D(filters=filters, 
                                 kernel_size=kernel, 
                                 activation=tfkl.LeakyReLU(0.2, name = name + '_leakyRelu'), 
                                 padding='same', kernel_initializer='glorot_normal')
        self.conv2 = tfkl.Conv2D(filters=filters, 
                                 kernel_size=kernel, 
                                 activation=tfkl.LeakyReLU(0.2, name = name + '_leakyRelu'),
                                 padding='same', kernel_initializer='glorot_normal')
        
        self.batch_normalization = tfkl.BatchNormalization(momentum=0.99, epsilon=0.001,name = name + '_bn')
        self.drop_out = tf.keras.layers.Dropout(dropout_prob)
        
    def call(self, x, training):
        feat = x[0]
        skip_feat = x[1]
        up = self.up_cnnT(feat)
        if self.skip_connection:
            x = tf.concat([up, tf.reshape(skip_feat, (-1, *skip_feat.shape[2:]))], axis=-1)
        else:
            x = up
            
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_normalization(x, training)
        x = self.drop_out(x)
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
                EncoderMiniBlock(filters=self.filters[i], kernel=config.filter_size, name='{0}_down_block_{1}'.format(self.name, i)))
        
        self.flatten = tfkl.Flatten(name='{0}_Flatten'.format(self.name))
        
        self.use_dist = tfpl.IndependentNormal        
        self.dense = tfkl.Dense(self.use_dist.params_size(self.dim_x), activation=None)
        
        activity_regularizer = None
        
        self.enc_q = self.use_dist(self.dim_x, activity_regularizer = activity_regularizer, name='encoder_dist')    
             
    def call(self, y, training):
        ph_steps = tf.shape(y)[1]              
        
        x_seq_down = tf.reshape(y, (-1, *self.dim_y, 1)) #(bs, ph_steps, h, w) -> (bs*ph_steps, h, w, 1)
        x_ref_down = tf.reshape(y[:,0,...], (-1, *self.dim_y, 1)) #(bs, h, w) -> (bs, h, w, 1)       
        
        x_ref_feat = []
        for i in range(len(self.filters)):
            x_seq, x_seq_down = self.down_blocks[i](x_seq_down, training)
            x_ref, x_ref_down = self.down_blocks[i](x_ref_down, training)                        
            
            # Repeat this to handle merge with sequence in decoder                
            x_ref = x_ref[:,None,...] # (bs, w, h, c) -> (bs, 1, w, h, c)            
            x_ref = tf.repeat(x_ref, ph_steps, axis=1) # (bs, 1, w, h, c) -> (bs, ph_steps, w, h, c)            
            #x_ref = tf.reshape(x_ref, (-1, *x_ref.shape[2:])) # (bs, ph_steps, w, h, c) -> (bs*ph_steps, w, h, c)            
            
            x_ref_feat.append(x_ref)
        
        x_seq_down = self.flatten(x_seq_down)
        x_seq_down = self.dense(x_seq_down)
        x_seq_down = tf.reshape(x_seq_down, (-1, ph_steps, self.use_dist.params_size(self.dim_x)))
        q_enc = self.enc_q(x_seq_down)
        
        return q_enc, x_ref_feat
    
class Decoder(tfk.Model):
    def __init__(self, config, output_channels, name='Decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.output_channels = output_channels
        self.dim_y = config.dim_y
        #self.dim_x = config.dec_input_dim
        self.filters = config.enc_filters
        self.skip_connection= config.skip_connection
                                           
        h = int(config.dim_y[0] / (2**(len(self.filters))))
        w = int(config.dim_y[1] / (2**(len(self.filters))))
        self.fc_block = Fc_block(h,w,self.filters[-1], name='{0}_FC_block'.format(self.name))
        self.up_blocks = []
        for i in reversed(range(len(self.filters))):
            self.up_blocks.append(
                DecoderMiniBlock(skip_connection = self.skip_connection, 
                                 filters = self.filters[i], 
                                 kernel=config.filter_size, 
                                 name='{0}_up_block{1}'.format(self.name, i)))
        
        self.cnn_mean = tfkl.Conv2D(filters=self.output_channels,
                                   kernel_size=config.filter_size,
                                   strides=(1,1),
                                   padding='same',
                                   use_bias=True,
                                   kernel_initializer='glorot_normal',
                                   bias_initializer='zeros',
                                   activation=None, #TODO: add tanh or softsign as activation
                                   name = name+'_lastCNN')
        
        self.cnn_logsigma = tf.keras.layers.Conv2D(filters=self.output_channels,
                                                   kernel_size=config.filter_size, 
                                                   strides=(1,1),
                                                   padding='same',
                                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-10),
                                                   bias_initializer=tf.keras.initializers.Constant(value=-10))
        
        decoder_dist = tfpl.IndependentNormal
        if output_channels > 1:
            self.decoder_dist = decoder_dist((*config.dim_y,output_channels), 
                                             convert_to_tensor_fn=tfd.Distribution.sample,
                                             name='decoder_dist')
        else:
            self.decoder_dist = decoder_dist(config.dim_y, 
                                             convert_to_tensor_fn=tfd.Distribution.sample,
                                             name='decoder_dist')

    def call(self, inputs, training):
        x = inputs[0]
        x_feat = inputs[1]
        x_feat.reverse()
        
        ph_steps = tf.shape(x)[1]
        dim_x = tf.shape(x)[-1]
        x = tf.reshape(x, (-1, dim_x)) # (b,s,latent_dim) -> (b*s, latent_dim)
        x = self.fc_block(x)
        for i, l in enumerate(self.up_blocks):
            x = l([x, x_feat[i]], training)
        
        y_mu = self.cnn_mean(x)
        y_logsigma = self.cnn_logsigma(x)
        
        y_mu = tf.reshape(y_mu, (-1, ph_steps, np.prod(self.dim_y)*self.output_channels))
        y_logsigma = tf.reshape(y_logsigma, (-1, ph_steps, np.prod(self.dim_y)*self.output_channels))        
        
        p_dec = self.decoder_dist(tf.concat([y_mu, y_logsigma], axis=-1))
        return p_dec