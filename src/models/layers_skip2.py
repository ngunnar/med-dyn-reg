import tensorflow as tf
import tensorflow_probability as tfp

from .utils import set_name

tfkl = tf.keras.layers
tfk = tf.keras
tfpl = tfp.layers
tfd = tfp.distributions

class Fc_block(tfkl.Layer):
    def __init__(self, h, w, c, name='fc_block', dropout_prob=0.0, reg_factor=0.01, prefix=None, **kwargs):
        super(Fc_block, self).__init__(name=set_name(name, prefix), **kwargs)
        self.dense = tfkl.Dense(h*w*c, 
                                use_bias=True,
                                bias_initializer='zeros',                                
                                bias_regularizer=tfk.regularizers.l2(reg_factor),
                                kernel_initializer='glorot_uniform',
                                kernel_regularizer=tfk.regularizers.l2(reg_factor),
                                activity_regularizer=tfk.regularizers.l2(reg_factor),
                                name = set_name(name + '_dense', prefix))
        self.dropout = tf.keras.layers.Dropout(dropout_prob, name=set_name(name + '_dropout', prefix))
        self.reshape =  tfkl.Reshape((h,w,c), name=set_name(name + '_reshape'))
        
    def call(self, x):
        x = self.dense(x)
        x = self.dropout(x)
        x = self.reshape(x)
        return x

class CNN_layer(tfkl.Layer):
    def __init__(self, filters, kernel_size, name, prefix, reg_factor=0.01, strides = (1,1), cnn_method=tfkl.Conv2D, **kwargs):
        super(CNN_layer, self).__init__(name=set_name(name, prefix), **kwargs)
        self.cnn = cnn_method(filters=filters,                              
                              name = set_name(name + '_cnn', prefix),
                              strides = strides,
                              padding='same', 
                              use_bias=True,
                              bias_regularizer=tfk.regularizers.l2(reg_factor),
                              kernel_size=kernel_size, 
                              kernel_initializer='he_normal',
                              kernel_regularizer=tfk.regularizers.l2(reg_factor))
        self.bn = tfkl.BatchNormalization(momentum=0.99, epsilon=0.001,name = set_name(name + '_bn', prefix))
        self.act = tfkl.LeakyReLU(0.2, name = set_name(name + '_leakyRelu', prefix))
        
    def call(self, x, training):
        x = self.cnn(x)
        x = self.bn(x, training)
        return self.act(x)
    
class LK_encoder(tfkl.Layer):
    def __init__(self, filters, kernel_size, name, reg_factor=0.01, prefix=None, **kwargs):
        super(LK_encoder, self).__init__(name=set_name(name, prefix), **kwargs)
        self.cnn = tfkl.Conv2D(filters=filters,
                               kernel_size=kernel_size, 
                               name = set_name(name + '_cnn', prefix),
                               padding='same', 
                               use_bias=True,
                               bias_regularizer=tfk.regularizers.l2(reg_factor),
                               kernel_initializer='he_normal',
                               kernel_regularizer=tfk.regularizers.l2(reg_factor))
        self.bn = tfkl.BatchNormalization(momentum=0.99, epsilon=0.001,name = set_name(name + '_bn', prefix))
        
    def call(self, x, training):
        x = self.cnn(x)
        x = self.bn(x, training)
        return x

class EncoderMiniBlock(tfkl.Layer):
    def __init__(self, filters, reg_factor=0.01, kernel_size=(3,3), name='down_block', prefix=None, dropout_prob=0.0, **kwargs):
        super(EncoderMiniBlock, self).__init__(name=set_name(name, prefix), **kwargs)
        self.conv_1 = CNN_layer(filters=filters, kernel_size=kernel_size, reg_factor=reg_factor, name=name + '_cnn1', prefix=prefix)
        self.conv_2 = CNN_layer(filters=filters, kernel_size=kernel_size, reg_factor=reg_factor, name=name + '_cnn2', prefix=prefix)
        self.dropout = tf.keras.layers.Dropout(dropout_prob)
        self.down_cnn = tfkl.MaxPool2D(pool_size=(2,2), name = set_name(name + '_down'))

    def call(self, x, training):
        x = self.conv_1(x, training)
        x = self.conv_2(x, training)
        down_x = self.dropout(x)
        down_x = self.down_cnn(down_x)
        return x, down_x

class DecoderMiniBlock(tfkl.Layer):
    def __init__(self, skip_connection, filters, reg_factor=0.01, kernel_size=(3,3), name='up_block', prefix=None, dropout_prob=0.0, **kwargs):
        super(DecoderMiniBlock, self).__init__(name=set_name(name, prefix), **kwargs)
        self.skip_connection = skip_connection
        '''
        self.up_cnn_T = CNN_layer(filters=filters,
                                  kernel_size=kernel_size, 
                                  strides = (2,2), 
                                  name= name + '_up',
                                  prefix=prefix,
                                  cnn_method=tfkl.Conv2DTranspose                         
                                  )
        '''
        self.up_cnn_T = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        # TODO: UpSampling2D instead of Conv2DTranspose
        
        self.conv_1 = CNN_layer(filters=filters, 
                                kernel_size=kernel_size, 
                                name=name + '_cnn1',
                                reg_factor = reg_factor,
                                prefix=prefix)        

        self.conv_2 = CNN_layer(filters=filters, 
                                kernel_size=kernel_size, 
                                name=name + '_cnn2',
                                reg_factor = reg_factor,
                                prefix=prefix)        
        self.dropout = tf.keras.layers.Dropout(dropout_prob)
        
    def call(self, x, training):
        feat = x[0]
        # Upsampling
        up = self.up_cnn_T(feat)#, training)
        if self.skip_connection:
            # Concatenate skip connection
            skip_feat = x[1]
            x = tf.concat([up, tf.reshape(skip_feat, (-1, *skip_feat.shape[2:]))], axis=-1)
        else:
            x = up
            
        x = self.conv_1(x, training)
        x = self.conv_2(x, training)
        x = self.dropout(x)
        return x

class Encoder(tfk.Model):
    def __init__(self, config, name="Encoder", prefix=None, **kwargs):
        super(Encoder, self).__init__(name=set_name(name, prefix), **kwargs)
        dropout_prob = 0.0
        reg_factor = 0.05
        self.dim_x = config.dim_x
        self.dim_y = config.dim_y
        self.filters = config.enc_filters        
                
        self.down_blocks = []
        for i in range(len(self.filters)):
            self.down_blocks.append(
                EncoderMiniBlock(filters=self.filters[i], 
                                 kernel_size=config.filter_size,
                                 dropout_prob = dropout_prob,
                                 name=set_name('{0}_down_block_{1}'.format(self.name, i), prefix)))
        
        self.flatten = tfkl.Flatten(name=set_name('{0}_flatten'.format(self.name), prefix))
        
        self.use_dist = tfpl.IndependentNormal        
        self.dense = tfkl.Dense(self.use_dist.params_size(self.dim_x), 
                                kernel_regularizer=tfk.regularizers.l2(reg_factor),      
                                bias_regularizer=tfk.regularizers.l2(reg_factor),
                                activation=None,
                                name = set_name(name+'_dense', prefix))
        
        self.dense_s = tfkl.Dense(self.dim_x, 
                                  kernel_regularizer=tfk.regularizers.l2(reg_factor),      
                                  bias_regularizer=tfk.regularizers.l2(reg_factor),
                                  activation=None,
                                  name = set_name(name+'_dense_s', prefix))
        
        activity_regularizer = None
        
        self.enc_q = self.use_dist(self.dim_x, 
                                   activity_regularizer = activity_regularizer, 
                                   name=set_name('encoder_dist', prefix)) 
             
    def call(self, y, y0, training):
        length = tf.shape(y)[1]              
        
        x = tf.reshape(y, (-1, *self.dim_y, 1)) #(bs, length, h, w) -> (bs*length, h, w, 1)
        s = tf.reshape(y0, (-1, *self.dim_y, 1)) #(bs, h, w) -> (bs, h, w, 1)       
        
        s_feats = []
        for i in range(len(self.filters)):
            _, x = self.down_blocks[i](x, training)
            s_feat, s = self.down_blocks[i](s, training)                        
            
            # Repeat this to handle merge with sequence in decoder                
            s_feat = s_feat[:,None,...] # (bs, w, h, c) -> (bs, 1, w, h, c)            
            s_feat = tf.repeat(s_feat, length, axis=1) # (bs, 1, w, h, c) -> (bs, length, w, h, c)     
            
            s_feats.append(s_feat)
        
        x = self.flatten(x)
        x = self.dense(x)
        x = tf.reshape(x, (-1, length, self.use_dist.params_size(self.dim_x)))
        q_xt = self.enc_q(x)

        s = self.flatten(s)
        s = self.dense_s(s)
        
        return q_xt, s, s_feats

    @tf.function
    def simple_encode(self, y):
        x = tf.reshape(y, (-1, *self.dim_y, 1))
        x_feats = []
        for i in range(len(self.filters)):
            x_feat, x = self.down_blocks[i](x, False)
            x_feats.append(x_feat[:,None,...])
        x = self.flatten(x)
        x = self.dense(x)
        x = tf.reshape(x, (-1, 1, self.use_dist.params_size(self.dim_x)))
        q_xt = self.enc_q(x)
        return q_xt, x_feats
    
    @tf.function
    def reference_encode(self, y0, length):
        s = tf.reshape(y0, (-1, *self.dim_y, 1))
        s_feats = []
        for i in range(len(self.filters)):
            s_feat, s = self.down_blocks[i](s, False)
            s_feat = s_feat[:,None,...] # (bs, w, h, c) -> (bs, 1, w, h, c)            
            s_feat = tf.repeat(s_feat, length, axis=1) # (bs, 1, w, h, c) -> (bs, length, w, h, c)     
            
            s_feats.append(s_feat)
            
        s = self.flatten(s)
        s = self.dense_s(s)
        return s, s_feats

    
class Decoder(tfk.Model):
    def __init__(self, skip_connection, config, output_channels, name='Decoder', prefix=None, **kwargs):
        super(Decoder, self).__init__(name=set_name(name, prefix), **kwargs)
        dropout_prob = 0.25
        reg_factor = 0.05
        self.output_channels = output_channels
        self.dim_y = config.dim_y
        self.filters = config.enc_filters
        self.skip_connection= skip_connection
                                           
        h = int(config.dim_y[0] / (2**(len(self.filters))))
        w = int(config.dim_y[1] / (2**(len(self.filters))))
        self.fc_block = Fc_block(h,w,self.filters[-1], 
                                 reg_factor = reg_factor,
                                 dropout_prob=dropout_prob, 
                                 name=set_name('{0}_fc_block'.format(self.name), prefix))
        self.up_blocks = []
        for i in reversed(range(len(self.filters))):
            self.up_blocks.append(
                DecoderMiniBlock(skip_connection = self.skip_connection, 
                                 dropout_prob = 0.0,
                                 reg_factor = reg_factor,
                                 filters = self.filters[i], 
                                 kernel_size=config.filter_size, 
                                 name=set_name('{0}_up_block{1}'.format(self.name, i), prefix)))
        
        self.cnn_mean = tfkl.Conv2D(filters=self.output_channels,
                                   kernel_size=config.filter_size,
                                   strides=(1,1),
                                   padding='same',
                                   use_bias=True,
                                   kernel_initializer='glorot_normal',
                                   bias_initializer='zeros',
                                   activation=None, #TODO: add tanh or softsign as activation
                                   kernel_regularizer=tfk.regularizers.l2(reg_factor),      
                                   bias_regularizer=tfk.regularizers.l2(reg_factor),
                                   name = set_name(name+'_mean', prefix))
        
        self.cnn_logsigma = tfkl.Conv2D(filters=self.output_channels,
                                        kernel_size=config.filter_size, 
                                        strides=(1,1),
                                        padding='same',
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-10),
                                        bias_initializer=tf.keras.initializers.Constant(value=-10),
                                        kernel_regularizer=tfk.regularizers.l2(reg_factor),      
                                        bias_regularizer=tfk.regularizers.l2(reg_factor),
                                        name = set_name(name+'_sigma', prefix))
        
        decoder_dist = tfpl.IndependentNormal
        if output_channels > 1:
            self.decoder_dist = decoder_dist((*config.dim_y,output_channels), 
                                             convert_to_tensor_fn=tfd.Distribution.sample,
                                             name=set_name('decoder_dist', prefix))
        else:
            self.decoder_dist = decoder_dist(config.dim_y, 
                                             convert_to_tensor_fn=tfd.Distribution.sample,
                                             name=set_name('decoder_dist', prefix))

    def call(self, inputs, training):
        x = inputs[0]
        if self.skip_connection:
            x_feat = inputs[1]
            x_feat.reverse()
        
        length = tf.shape(x)[1]
        dim_x = tf.shape(x)[-1]
        x = tf.reshape(x, (-1, dim_x)) # (b,s,latent_dim) -> (b*s, latent_dim)
        x = self.fc_block(x)
        for i, l in enumerate(self.up_blocks):
            if self.skip_connection:
                x = l([x, x_feat[i]], training)
            else:
                x = l([x], training)
        
        y_mu = self.cnn_mean(x)
        y_logsigma = self.cnn_logsigma(x)
        
        last_channel = self.dim_y[0] * self.dim_y[1] * self.output_channels
        y_mu = tf.reshape(y_mu, (-1, length, last_channel))
        y_logsigma = tf.reshape(y_logsigma, (-1, length, last_channel))   
        
        p_dec = self.decoder_dist(tf.concat([y_mu, y_logsigma], axis=-1))
        return p_dec