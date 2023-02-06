import tensorflow as tf
import tensorflow_probability as tfp
from voxelmorph.tf.layers import SpatialTransformer as SpatialTransformer
from voxelmorph.tf.layers import VecInt
import numpy as np
from .layers_skip import Encoder, Decoder
from ..losses import grad_loss

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class VRNN(tfk.Model):
    def __init__(self, config,
                 n_layers:int = 1,
                 bias=False, **kwargs):
        super(VRNN, self).__init__(self, **kwargs)
        self.config = config
        
        self.y_dim = config.dim_y        
        self.x_dim =config.dim_x        
        self.z_dim = config.dim_z
        self.n_layers = n_layers
        
        using_dist = tfpl.IndependentNormal
        
        # feature-extracting transformations (phi_y and phi_z)
        self.feat_y = Encoder(self.config)
        
        self.feat_x = tfk.Sequential([
            tfkl.Dense(self.z_dim, activation=tf.nn.relu)
        ])
        
        # encoder function (phi_enc) -> Inference
        self.enc = tfk.Sequential([
            tfkl.Dense(self.z_dim, activation = tf.nn.relu),
            tfkl.Dense(self.z_dim, activation = tf.nn.relu),
            tfkl.Dense(using_dist.params_size(self.x_dim), activation = None),
            using_dist(self.x_dim, convert_to_tensor_fn=tfd.Distribution.sample)
        ])       
        
         # prior function (phi_prior) -> Prior
        self.prior = tfk.Sequential([
            tfkl.Dense(self.z_dim, activation=tf.nn.relu),
            tfkl.Dense(self.z_dim, activation=tf.nn.relu),
            tfkl.Dense(using_dist.params_size(self.x_dim), activation = None),
            using_dist(self.x_dim, convert_to_tensor_fn=tfd.Distribution.sample)
        ])
        
        # decoder function (phi_dec) -> Generation
        self.dec = Decoder(self.config, output_channels = 2)
        
        self.output_dist = tfpl.IndependentNormal(config.dim_y, name='output_dist')
        self.y_sigma = lambda y: tf.ones_like(y, dtype='float32') * tfp.math.softplus_inverse(0.01)
        self.stn = SpatialTransformer()
        self.warp = tf.keras.layers.Lambda(lambda x: self.warping(x), name='warping')
        
        if self.config.int_steps > 0:            
            self.vecInt = VecInt(method='ss', name='s_flow_int', int_steps=self.config.int_steps)

        # recurrence function (f_theta) -> Recurrence                
        self.grus = []
        for i in range(self.n_layers):
            self.grus.append(tfkl.GRUCell(units=self.z_dim, use_bias = bias))
        
        #self.rnn = tfk.
        
    def warping(self, inputs):
        phi = inputs[0]
        y_0 = inputs[1]
        bs, ph_steps, dim_y, _, channels = phi.shape
        y_0 = tf.repeat(y_0[:,None,...], ph_steps, axis=1)
        images = tf.reshape(y_0, (-1, *(dim_y,dim_y), 1))

        flows = tf.reshape(phi, (-1, *(dim_y,dim_y), 2))
        y_pred = self.stn([images, flows])
        y_pred = tf.reshape(y_pred, (-1, ph_steps, *(dim_y,dim_y)))
        return y_pred
    
    def initialize_z(self, batch_size):
        # initialization
        def cond(i,l):
            return i < self.n_layers

        def body(i,l):
            l = l.write(i, tf.zeros((batch_size, self.z_dim)))
            return tf.add(i, 1), l

        _, z = tf.while_loop(cond, body, loop_vars = [tf.constant(0),
                                                      tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)])
        
        return z.stack()
    
    
    def diff_steps(self, phi_t):
        dim_y = phi_t.shape[2:4]
        ph_steps = phi_t.shape[1]
        phi_t = tf.reshape(phi_t, (-1, *dim_y, 2))
        phi_t= self.vecInt(phi_t)
        phi_t = tf.reshape(phi_t, (-1, ph_steps, *dim_y, 2))
        return phi_t
    
    def call(self, inputs, training): # p(x_t | z_t)
        y = inputs['input_video']        
        y_shape = tf.shape(y)
        batch_size = y_shape[0]
        seq_len = y_shape[1]
        features = y_shape[2]  #(bs, seq, *dim_y)
        ref_y = y[:,0,...]
        
        # y (bs, ph_steps, h, w)        
        feat_y_ref, hl_feat_y_ref = self.feat_y(ref_y[:,None,...], training)
        
        
        loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        loss_grads = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        
        loss_preds = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        loss_klds = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        samples_img_dist = []
        samples_flow_dist = []
        samples_z_dist = []
    
        z = self.initialize_z(batch_size)   # TODO initialize h based on ref_y?                                              
               
        # for all time steps
        for t in range(seq_len):            
            # feature extraction: y_t
            feat_y_t, _ = self.feat_y(y[:,t,...][:,None,...], training)

            # encoder: y_t, z_t -> x_t            
            enc_in = tf.concat([feat_y_t[:,0,:], z[-1]], axis=1)
            x_dist_t = self.enc(enc_in)

            # prior: z_t -> x_t (for KLD loss)
            prior_in = tf.concat([feat_y_ref[:,0,:], z[-1]], axis=1)
            prior_dist_t = self.prior(prior_in)

            # sampling and reparameterization: get a new x_t            
            x_t = x_dist_t.sample()
            # feature extraction: z_t
            feat_x_t = self.feat_x(x_t)

            # decoder: x_t -> y_t
            dec_in = [tf.concat([feat_x_t, z[-1]], axis=1)[:,None,:], hl_feat_y_ref]
            phi_dist_t = self.dec(dec_in)
            phi_t = phi_dist_t.sample()
        
            if self.config.int_steps > 0:
                phi_t = self.diff_steps(phi_t)

            y_mu = self.warp([phi_t, ref_y])
            y_mu = tf.reshape(y_mu, (-1, y_mu.shape[1], np.prod(y_mu.shape[2:4])))
            y_sigma = self.y_sigma(y_mu)
            pred_dist_t =  self.output_dist(tf.concat([y_mu, y_sigma], axis=-1))  
            
            # recurrence: y_t+1, x_t+1 -> z_t+1
            for gru in self.grus:                
                _, z = gru(inputs=tf.concat([feat_y_t[:,0,:], feat_x_t], axis=1), states = z)

            # computing the loss
            KLD = tfp.distributions.kl_divergence(x_dist_t, prior_dist_t)
            
            loss_grad = grad_loss('l2', phi_dist_t.mean())
            loss_grads = loss_grads.write(t, loss_grad)
            loss_pred = pred_dist_t.log_prob(y[:, t, :][:, None, :])[:,0]          
            loss_preds = loss_preds.write(t, loss_pred)
            loss_klds = loss_klds.write(t, KLD)
            loss = loss.write(t, -loss_pred + KLD)
                        
            samples_img_dist.append(pred_dist_t)
            samples_flow_dist.append(phi_dist_t)
            samples_z_dist.append(x_dist_t)
        
        loss = loss.stack() #(t, batch_size)
        loss = tf.transpose(loss, perm=[1,0])

        loss_klds = loss_klds.stack()
        loss_klds = tf.transpose(loss_klds, perm=[1,0])

        loss_preds = loss_preds.stack()
        loss_preds = tf.transpose(loss_preds, perm=[1,0])        
        #self.add_loss(tf.reduce_mean(tf.reduce_sum(loss, axis=1)))
        self.add_loss(tf.reduce_mean(tf.reduce_sum(-loss_preds, axis=1)))
        self.add_loss(tf.reduce_mean(tf.reduce_sum(loss_klds, axis=1)))
        self.add_metric(tf.reduce_mean(tf.reduce_sum(loss_klds, axis=1)), 'KLD ↓')
        self.add_metric(tf.reduce_mean(tf.reduce_sum(loss_preds, axis=1)), 'log p(y|x) ↑')        
        
        loss_grads = loss_grads.stack()
        loss_grads = tf.transpose(loss_grads, perm=[1,0])
        self.add_metric(tf.reduce_mean(tf.reduce_sum(loss_grads, axis=1)), 'grad flow ↓') # TODO: Add to loss
        self.add_loss(tf.reduce_mean(tf.reduce_sum(loss_grads, axis=1)))
        return samples_img_dist, samples_flow_dist, samples_z_dist

    def generate(self, y, ref_y, mask):
        training = False
        feat_y_ref, hl_feat_y_ref = self.feat_y(ref_y[:,None,...], training)
        
        batch_size = tf.shape(y)[0]
        seq_len = tf.shape(y)[1]

        samples_imgs_dist = []
        samples_flow_dist = []
        samples_x_dist = []

        # initialization
        z = self.initialize_z(batch_size)
    
        for t in range(seq_len):
            # Feature extration
            if tf.reduce_any(mask[:,t]): #unknown value
                # Prior
                prior_in = tf.concat([feat_y_ref[:,0,:], z[-1]], axis=1)
                x_dist_t = self.prior(prior_in)
            else:
                # feature extraction: y_t
                feat_y_t,_ = self.feat_y(y[:, t, :][:,None,...], training)
                # encoder: y_t, z_t -> x_t
                enc_in = tf.concat([feat_y_t[:,0,:], z[-1]], 1)
                x_dist_t = self.enc(enc_in)                

            # sampling and reparameterization: get a new x_t            
            x_t = x_dist_t.sample()

            # feature extraction: x_t
            feat_x_t = self.feat_x(x_t)

            # decoder: x_t -> y_t
            #pred_dist_t = self.dec(phi_z_t)
            dec_in = [tf.concat([feat_x_t, z[-1]], axis=1)[:,None,:], hl_feat_y_ref]
            phi_dist_t = self.dec(dec_in)
            phi_t = phi_dist_t.sample()
            if self.config.int_steps > 0:
                phi_t = self.diff_steps(phi_t)

            y_mu = self.warp([phi_t, ref_y])
            y_mu = tf.reshape(y_mu, (-1, y_mu.shape[1], np.prod(y_mu.shape[2:4])))
            y_sigma = self.y_sigma(y_mu)
            pred_dist_t =  self.output_dist(tf.concat([y_mu, y_sigma], axis=-1))
                
            if tf.reduce_any(mask[:,t]): #unknown value
                feat_y_t, _ = self.feat_y(pred_dist_t.sample(), training)
            
            # recurrence: y_t+1 -> h_t+1
            for gru in self.grus:                
                _, z = gru(inputs=tf.concat([feat_y_t[:,0,:], feat_x_t], axis=1), states = z)            

            samples_imgs_dist.append(pred_dist_t)
            samples_flow_dist.append(phi_dist_t)
            samples_x_dist.append(x_dist_t)

        return samples_imgs_dist, samples_flow_dist, samples_x_dist

    def train(self, data, epochs:int = 300, val_data = None):
        self.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001))
        self.history = self.fit(x = data, epochs = epochs, validation_data = val_data)

    def get_loss(self):
        return self.history.history