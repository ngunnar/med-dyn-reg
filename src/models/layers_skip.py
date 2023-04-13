import tensorflow as tf
import tensorflow_probability as tfp

from .utils import set_name

tfkl = tf.keras.layers
tfk = tf.keras
tfpl = tfp.layers
tfd = tfp.distributions

class Fc_block(tfkl.Layer):
    def __init__(self, h, w, c, name='FC_block', prefix=None, **kwargs):
        super(Fc_block, self).__init__(name=set_name(name, prefix), **kwargs)
        self.dense = tfkl.Dense(h*w*c, 
                                use_bias=True,
                                kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                name = set_name(name + '_dense', prefix))
        self.reshape =  tfkl.Reshape((h,w,c), name=set_name(name + '_reshape'))
        
    def call(self, x):
        x = self.dense(x)
        x = self.reshape(x)
        return x

    
class LK_encoder(tfkl.Layer):
    def __init__(self, filters, kernel, name, prefix=None, **kwargs):
        super(LK_encoder, self).__init__(name=set_name(name, prefix), **kwargs)
        self.cnn = tfkl.Conv2D(filters=filters,
                               kernel_size=kernel, 
                               padding='same', 
                               kernel_initializer='glorot_normal')
        self.batch_norm = tfkl.BatchNormalization(momentum=0.99, epsilon=0.001,name = set_name(name + '_bn', prefix))
        
    def call(self, x, training):
        x = self.cnn(x)
        x = self.batch_norm(x, training)
        return x

class EncoderMiniBlock(tfkl.Layer):
    def __init__(self, filters, kernel=(3,3), name='down_block', prefix=None, dropout_prob=0.0, **kwargs):
        super(EncoderMiniBlock, self).__init__(name=set_name(name, prefix), **kwargs)
        self.conv1 = tfkl.Conv2D(filters=filters,
                                 kernel_size=kernel,
                                 activation=tfkl.LeakyReLU(0.2, name = set_name(name + '_leakyRelu', prefix)),
                                 padding='same',
                                 kernel_initializer='glorot_normal')
        
        # LK
        self.regular_kernel = LK_encoder(filters=filters, 
                                          kernel=kernel, 
                                          name = set_name(name + '_regular_kernel', prefix))
        self.large_kernel = LK_encoder(filters=filters, 
                                          kernel=(5,5), 
                                          name = set_name(name + '_large_kernel', prefix))
        self.one_kernel = LK_encoder(filters=filters, 
                                          kernel=(1,1), 
                                          name = set_name(name + '_one_kernel', prefix))
        
        self.activation_fn = tfkl.LeakyReLU(0.2, name = set_name(name + '_leakyRelu', prefix))
        
        self.batch_normalization = tfkl.BatchNormalization(momentum=0.99, epsilon=0.001,name = set_name(name + '_bn', prefix))
        
        self.drop_out = tf.keras.layers.Dropout(dropout_prob)
        self.down_cnn = tfkl.Conv2D(filters = filters, 
                                    kernel_size=kernel, 
                                    strides = (2,2), 
                                    activation=tfkl.LeakyReLU(0.2, name = set_name(name + '_leakyRelu_down', prefix)), 
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
    def __init__(self, skip_connection, filters, kernel=(3,3), name='up_block', prefix=None, dropout_prob=0.0, **kwargs):
        super(DecoderMiniBlock, self).__init__(name=set_name(name, prefix), **kwargs)
        self.skip_connection = skip_connection
        self.up_cnnT = tfkl.Conv2DTranspose(filters=filters, kernel_size=kernel, strides = (2,2), activation='relu', padding='same', kernel_initializer='HeNormal')
        self.conv1 = tfkl.Conv2D(filters=filters, 
                                 kernel_size=kernel, 
                                 activation=tfkl.LeakyReLU(0.2, name = set_name(name + '_leakyRelu1', prefix)), 
                                 padding='same', kernel_initializer='glorot_normal')
        self.conv2 = tfkl.Conv2D(filters=filters, 
                                 kernel_size=kernel, 
                                 activation=tfkl.LeakyReLU(0.2, name = set_name(name + '_leakyRelu2', prefix)),
                                 padding='same', kernel_initializer='glorot_normal')
        
        self.batch_normalization = tfkl.BatchNormalization(momentum=0.99, epsilon=0.001,name = set_name(name + '_bn', prefix))
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
    def __init__(self, config, name="Encoder", prefix=None, **kwargs):
        super(Encoder, self).__init__(name=set_name(name, prefix), **kwargs)
        self.dim_x = config.dim_x
        self.dim_y = config.dim_y
        self.filters = config.enc_filters        
                
        self.down_blocks = []
        for i in range(len(self.filters)):
            self.down_blocks.append(
                EncoderMiniBlock(filters=self.filters[i], kernel=config.filter_size, name=set_name('{0}_down_block_{1}'.format(self.name, i), prefix)))
        
        self.flatten = tfkl.Flatten(name=set_name('{0}_Flatten'.format(self.name), prefix))
        
        self.use_dist = tfpl.IndependentNormal        
        self.dense = tfkl.Dense(self.use_dist.params_size(self.dim_x), activation=None)
        
        activity_regularizer = None
        
        self.enc_q = self.use_dist(self.dim_x, activity_regularizer = activity_regularizer, name=set_name('encoder_dist', prefix)) 
             
    def call(self, y, y_ref, training):
        length = tf.shape(y)[1]              
        
        x_down = tf.reshape(y, (-1, *self.dim_y, 1)) #(bs, length, h, w) -> (bs*length, h, w, 1)
        s_down = tf.reshape(y_ref, (-1, *self.dim_y, 1)) #(bs, h, w) -> (bs, h, w, 1)       
        
        s_feats = []
        for i in range(len(self.filters)):
            _, x_down = self.down_blocks[i](x_down, training)
            s_feat, s_down = self.down_blocks[i](s_down, training)                        
            
            # Repeat this to handle merge with sequence in decoder                
            s_feat = s_feat[:,None,...] # (bs, w, h, c) -> (bs, 1, w, h, c)            
            s_feat = tf.repeat(s_feat, length, axis=1) # (bs, 1, w, h, c) -> (bs, length, w, h, c)     
            
            s_feats.append(s_feat)
        
        x_down = self.flatten(x_down)
        x_down = self.dense(x_down)
        x_down = tf.reshape(x_down, (-1, length, self.use_dist.params_size(self.dim_x)))
        q_enc = self.enc_q(x_down)

        s_down = self.flatten(s_down)
        s_down = self.dense(s_down)
        s_down = tf.reshape(s_down, (-1, self.use_dist.params_size(self.dim_x)))
        q_ref_enc = self.enc_q(s_down)
        
        return q_enc, q_ref_enc, s_feats

    @tf.function
    def simple_encode(self, y):
        x_down = tf.reshape(y, (-1, *self.dim_y, 1))
        x_feats = []
        for i in range(len(self.filters)):
            x_feat, x_down = self.down_blocks[i](x_down, False)
            x_feats.append(x_feat[:,None,...])
        x_down = self.flatten(x_down)
        x_down = self.dense(x_down)
        x_down = tf.reshape(x_down, (-1, 1, self.use_dist.params_size(self.dim_x)))
        q_enc = self.enc_q(x_down)
        return q_enc, x_feats
    
    @tf.function
    def reference_encode(self, y_ref, length):
        x_down = tf.reshape(y_ref, (-1, *self.dim_y, 1))
        x_feats = []
        for i in range(len(self.filters)):
            x_feat, x_down = self.down_blocks[i](x_down, False)
            x_feat = x_feat[:,None,...] # (bs, w, h, c) -> (bs, 1, w, h, c)            
            x_feat = tf.repeat(x_feat, length, axis=1) # (bs, 1, w, h, c) -> (bs, length, w, h, c)     
            
            x_feats.append(x_feat)
            
        x_down = self.flatten(x_down)
        x_down = self.dense(x_down)
        x_down = tf.reshape(x_down, (-1, self.use_dist.params_size(self.dim_x)))
        q_enc = self.enc_q(x_down)
        return q_enc, x_feats

    
class Decoder(tfk.Model):
    def __init__(self, config, output_channels, name='Decoder', prefix=None, **kwargs):
        super(Decoder, self).__init__(name=set_name(name, prefix), **kwargs)
        self.output_channels = output_channels
        self.dim_y = config.dim_y
        self.filters = config.enc_filters
        self.skip_connection= config.skip_connection
                                           
        h = int(config.dim_y[0] / (2**(len(self.filters))))
        w = int(config.dim_y[1] / (2**(len(self.filters))))
        self.fc_block = Fc_block(h,w,self.filters[-1], name=set_name('{0}_FC_block'.format(self.name), prefix))
        self.up_blocks = []
        for i in reversed(range(len(self.filters))):
            self.up_blocks.append(
                DecoderMiniBlock(skip_connection = self.skip_connection, 
                                 filters = self.filters[i], 
                                 kernel=config.filter_size, 
                                 name=set_name('{0}_up_block{1}'.format(self.name, i), prefix)))
        
        self.cnn_mean = tfkl.Conv2D(filters=self.output_channels,
                                   kernel_size=config.filter_size,
                                   strides=(1,1),
                                   padding='same',
                                   use_bias=True,
                                   kernel_initializer='glorot_normal',
                                   bias_initializer='zeros',
                                   activation=None, #TODO: add tanh or softsign as activation
                                   name = set_name(name+'_lastCNN', prefix))
        
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
                                             name=set_name('decoder_dist', prefix))
        else:
            self.decoder_dist = decoder_dist(config.dim_y, 
                                             convert_to_tensor_fn=tfd.Distribution.sample,
                                             name=set_name('decoder_dist', prefix))

    def call(self, inputs, training):
        x = inputs[0]
        x_feat = inputs[1]
        x_feat.reverse()
        
        length = tf.shape(x)[1]
        dim_x = tf.shape(x)[-1]
        x = tf.reshape(x, (-1, dim_x)) # (b,s,latent_dim) -> (b*s, latent_dim)
        x = self.fc_block(x)
        for i, l in enumerate(self.up_blocks):
            x = l([x, x_feat[i]], training)
        
        y_mu = self.cnn_mean(x)
        y_logsigma = self.cnn_logsigma(x)
        
        last_channel = self.dim_y[0] * self.dim_y[1] * self.output_channels
        y_mu = tf.reshape(y_mu, (-1, length, last_channel))
        y_logsigma = tf.reshape(y_logsigma, (-1, length, last_channel))   
        
        p_dec = self.decoder_dist(tf.concat([y_mu, y_logsigma], axis=-1))
        return p_dec