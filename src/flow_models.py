import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from voxelmorph.tf.layers import SpatialTransformer as SpatialTransformer
from voxelmorph.tf.layers import VecInt

tfk = tf.keras
tfpl = tfp.layers

from .models import VAE, KVAE

from .metrics import batch_jacobian_determinant
from .losses import grad_loss
#from .stn import SpatialTransformer


def gaussian_filter(x, sigma, filter_shape, filter_num):
    for i in range(filter_num):
        x = tfa.image.gaussian_filter2d(x, filter_shape, sigma=sigma)
    return x

def gaussian_filter_batch_seq(x, sigma, filter_shape, filter_num):
    bs = x.shape[0]
    ph = x.shape[1]
    x = tf.reshape(x, (bs*ph, x.shape[2], x.shape[3], x.shape[4]))
    for i in range(filter_num):
        x = tfa.image.gaussian_filter2d(x, filter_shape, sigma=sigma)
    x = tf.reshape(x, (bs, ph, x.shape[1], x.shape[2], x.shape[3]))
    return x

class fKVAE(KVAE):
    def __init__(self, config, name="fKVAE", output_channels=2, **kwargs):
        super(fKVAE, self).__init__(name=name, output_channels = output_channels, config=config, **kwargs)
        self.grad_flow_metric = tfk.metrics.Mean(name = 'grad flow ↓')
        self.dist = tfpl.IndependentNormal(config.dim_y)
        self.gaussian_kernel = tf.keras.layers.Lambda(lambda x: gaussian_filter_batch_seq(x, 3.0, (15,15), 1), name='gaussian_kernel')
        self.stn = SpatialTransformer()
        self.warp = tf.keras.layers.Lambda(lambda x: self.warping(x), name='warping')
        external_mask = tf.convert_to_tensor(np.load(config.ds_path + '/external_mask.npy'))
        self.external_mask = tf.repeat(external_mask[...,None], 2, axis=-1)
        if self.config.int_steps > 0:            
            self.vecInt = VecInt(method='ss', name='s_flow_int', int_steps=self.config.int_steps)
        
    
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
    
    def flow2grid(self, flow, d = 10):
        bs = flow.shape[0]
        ph_steps = flow.shape[1]
        dim_y = flow.shape[-3:-1]

        l = np.array(dim_y.as_list()) // d
        # grid image
        img_grid = np.zeros((bs, *dim_y), dtype='float32')
        img_grid[:,10::10,:] = 1.0
        img_grid[:,:,10::10] = 1.0
        return self.warp([flow, img_grid])

    
    def call(self, inputs, training):
        y = inputs[0]
        mask = inputs[1]
        #y_0 = inputs[2]
        
        # Add noise to image(s)
        #y = y + tf.random.normal(shape=tf.shape(y), mean=0.0, stddev=0.01, dtype=tf.float32)
        #y_0 = y_0 + tf.random.normal(shape=tf.shape(y_0), mean=0.0, stddev=0.01, dtype=tf.float32)
        
        p_dec, q_enc, p_smooth, p_fdec, x_sample, z_sample = self.forward(y, mask, training)
        
        metrices = self.get_loss(y, x_sample, z_sample, q_enc, p_smooth, p_dec, mask = mask, prior = self.prior)
        
        
        grad = grad_loss('l2', p_fdec.mean())
        self.grad_flow_metric.update_state(grad)
        if 'grad' in self.config.losses:
            self.add_loss(grad)
        metrices['grad']=tf.reduce_mean(grad)

        return p_dec, metrices
    
    def forward(self, y, mask, training):
        q_enc, p_smooth, p_fdec, x_sample, z_sample = super().forward(y, mask, training)
        phi_sample = p_fdec.sample() #bs, t, w, h, 2
        if self.config.int_steps > 0:
            dim_y = y.shape[2:4]
            ph_steps = y.shape[1]
            phi_sample = tf.reshape(phi_sample, (-1, *dim_y, 2))
            phi_sample= self.vecInt(phi_sample)
            phi_sample = tf.reshape(phi_sample, (-1, ph_steps, *dim_y, 2))
               
        y_mu = self.warp([phi_sample, y[:,0,...]])
        y_sigma = tf.ones_like(y_mu, dtype='float32') * tfp.math.softplus_inverse(0.01) # softplus is used so a -4.6 approx std 0.01
        y_mu = tf.reshape(y_mu, (-1, y_mu.shape[1], np.prod(y_mu.shape[2:])))
        y_sigma = tf.reshape(y_sigma, (-1, y_mu.shape[1], np.prod(y_mu.shape[2:])))
        p_dec = self.dist(tf.concat([y_mu, y_sigma], axis=-1))

        return p_dec, q_enc, p_smooth, p_fdec, x_sample, z_sample

    @tf.function
    def predict(self, inputs, use_kernel=False):
        y = inputs[0]
        mask = inputs[1]
        
        #inputs = [y + tf.random.normal(shape=tf.shape(y), mean=0.0, stddev=0.01, dtype=tf.float32),
        #          mask,
        #          y_0 + tf.random.normal(shape=tf.shape(y_0), mean=0.0, stddev=0.01, dtype=tf.float32)]
        
        phi_filt_sample, phi_pred_sample, phi_smooth_sample = self._predict(inputs)
        
        if self.config.int_steps > 0:
            dim_y = y.shape[2:4]
            ph_steps = y.shape[1]
            phi_filt_sample= tf.reshape(self.vecInt(tf.reshape(phi_filt_sample, (-1, *dim_y, 2))), (-1, ph_steps, *dim_y, 2))
            phi_pred_sample= tf.reshape(self.vecInt(tf.reshape(phi_pred_sample, (-1, *dim_y, 2))), (-1, ph_steps, *dim_y, 2))
            phi_smooth_sample= tf.reshape(self.vecInt(tf.reshape(phi_smooth_sample, (-1, *dim_y, 2))), (-1, ph_steps, *dim_y, 2))
        
        
        if use_kernel:
            phi_filt_sample = tf.reshape(phi_filt_sample, (-1, y.shape[2], y.shape[3], 2))
            phi_pred_sample = tf.reshape(phi_pred_sample, (-1, y.shape[2], y.shape[3], 2))
            phi_smooth_sample = tf.reshape(phi_smooth_sample, (-1, y.shape[2], y.shape[3], 2))
            
            phi = tf.concat([phi_filt_sample, phi_pred_sample, phi_smooth_sample], axis=0)
            
            phi = gaussian_filter(phi, 3.0, (5,5), 4)
            
            div = phi.shape[0]//3
            phi_filt_sample = phi[:div]
            phi_pred_sample = phi[div:div*2]
            phi_smooth_sample = phi[div*2:]
            
            phi_filt_sample = tf.reshape(phi_filt_sample, (y.shape[0], y.shape[1], y.shape[2], y.shape[3], 2))
            phi_pred_sample = tf.reshape(phi_pred_sample, (y.shape[0], y.shape[1], y.shape[2], y.shape[3], 2))
            phi_smooth_sample = tf.reshape(phi_smooth_sample, (y.shape[0], y.shape[1], y.shape[2], y.shape[3], 2))
        
        
        phi_filt_sample *= self.external_mask
        phi_pred_sample *= self.external_mask
        phi_smooth_sample *= self.external_mask
        y_filt = self.warp([phi_filt_sample, y[:,0,...]])
        y_pred = self.warp([phi_pred_sample, y[:,0,...]])
        y_smooth = self.warp([phi_smooth_sample, y[:,0,...]])       
        
        result = [{'name':'filt', 'data': y_filt},
                {'name':'filt_flow', 'data': phi_filt_sample},
                {'name':'pred', 'data': y_pred},
                {'name':'pred_flow', 'data': phi_pred_sample},
                {'name':'smooth', 'data': y_smooth},
                {'name':'smooth_flow', 'data': phi_smooth_sample}]

        return result
    
    @tf.function
    def get_filt(self, inputs, use_kernel=False):
        y = inputs[0]
        mask = inputs[1]
        phi_filt_sample = self._get_filt(inputs)
        
        if self.config.int_steps > 0:
            phi_filt_sample= self.vecInt(phi_filt_sample)
        
        if use_kernel:
            phi_filt_sample = tf.reshape(phi_filt_sample, (-1, y.shape[2], y.shape[3], 2))
            phi_filt_sample = gaussian_filter(phi_filt_sample, 3.0, (5,5), 4)
            phi_filt_sample = tf.reshape(phi_filt_sample, (y.shape[0], y.shape[1], y.shape[2], y.shape[3], 2))
            
        phi_filt_sample *= self.external_mask
        y_filt = self.warp([phi_filt_sample, y[:,0,...]])
        result = [{'name':'filt', 'data': y_filt},
                  {'name':'filt_flow', 'data': phi_filt_sample}]
        return result
        
    def info(self):
        y = tf.keras.layers.Input(shape=(self.config.ph_steps, *self.config.dim_y), batch_size=self.config.batch_size)
        mask = tf.keras.layers.Input(shape=(self.config.ph_steps), batch_size=self.config.batch_size)
        first_frame = tf.keras.layers.Input(shape=self.config.dim_y, batch_size=self.config.batch_size)
        inputs = [y, mask, first_frame]
        
        self.gaussian_kernel.build((1, 112,112,1))
        
        self._print_info(inputs)