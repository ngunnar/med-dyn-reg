import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from voxelmorph.tf.layers import SpatialTransformer as SpatialTransformer
from voxelmorph.tf.layers import VecInt

tfk = tf.keras
tfpl = tfp.layers
tfd = tfp.distributions

from ..losses import grad_loss

from .layers_skip import Decoder
from .kvae import KVAE

class fKVAE(KVAE):
    def __init__(self, config, name="fKVAE", **kwargs):
        super(fKVAE, self).__init__(config = config, name=name, **kwargs)
        self.decoder = Decoder(self.config, output_channels = 2)       
        
        self.output_dist = tfpl.IndependentNormal(config.dim_y, name='output_dist')
        self.y_sigma = lambda y: tf.ones_like(y, dtype='float32') * tfp.math.softplus_inverse(0.01)
        self.stn = SpatialTransformer()
        self.warp = tf.keras.layers.Lambda(lambda x: self.warping(x), name='warping')
        #external_mask = tf.convert_to_tensor(np.load(config.ds_path + '/external_mask.npy'))
        #self.external_mask = tf.repeat(external_mask[...,None], 2, axis=-1)
        if self.config.int_steps > 0:            
            self.vecInt = VecInt(method='ss', name='s_flow_int', int_steps=self.config.int_steps)
        
        self.grad_flow_metric = tfk.metrics.Mean(name = 'grad flow ↓')        

    def warping(self, inputs):
        phi = inputs[0]
        y_0 = inputs[1]
        _, ph_steps, dim_y, _, _ = phi.shape
        y_0 = tf.repeat(y_0[:,None,...], ph_steps, axis=1)
        images = tf.reshape(y_0, (-1, *(dim_y,dim_y), 1))

        flows = tf.reshape(phi, (-1, *(dim_y,dim_y), 2))
        y_pred = self.stn([images, flows])
        y_pred = tf.reshape(y_pred, (-1, ph_steps, *(dim_y,dim_y)))
        return y_pred

    def diff_steps(self, phi_sample):
        dim_y = phi_sample.shape[2:4]
        ph_steps = phi_sample.shape[1]
        phi_sample = tf.reshape(phi_sample, (-1, *dim_y, 2))
        phi_sample= self.vecInt(phi_sample)
        phi_sample = tf.reshape(phi_sample, (-1, ph_steps, *dim_y, 2))
        return phi_sample

    def call(self, inputs, training):
        y = inputs['input_video']
        mask = inputs['input_mask'] 
        
        q_enc, x_sample, p_smooth, z_sample, p_dec = self.forward(inputs, training)
        phi_sample = p_dec.sample()
        
        if self.config.int_steps > 0:
            phi_sample = self.diff_steps(phi_sample)
        
        y_mu = self.warp([phi_sample, y[:,0,...]])
        y_mu = tf.reshape(y_mu, (-1, y.shape[1], np.prod(y.shape[2:4])))
        y_sigma = self.y_sigma(y_mu)
        out_dist =  self.output_dist(tf.concat([y_mu, y_sigma], axis=-1))        

        self.set_loss(y, mask, out_dist, p_dec, q_enc, x_sample, z_sample, p_smooth)
        self.loss_metric.update_state(tf.reduce_sum(self.losses))
        return

    def eval(self, inputs):
        y = inputs['input_video']
        mask = inputs['input_mask'] 
        
        q_enc, x_ref_feat = self.encoder(y, training=False)         
        x_sample = q_enc.sample()
        
        # Latent distributions 
        latent_dist = self.lgssm.get_distribtions(x_sample, mask)
        
        # Flow distributions        
        p_dec_vae = self.decoder([x_sample, x_ref_feat], training=False)
        p_dec_smooth = self.decoder([latent_dist['smooth'].sample(), x_ref_feat], training=False)
        p_dec_filt = self.decoder([latent_dist['filt'].sample(), x_ref_feat], training=False)
        p_dec_pred = self.decoder([latent_dist['pred'].sample(), x_ref_feat], training=False)

        # Flow samples
        phi_vae = p_dec_vae.sample()
        phi_smooth = p_dec_smooth.sample()
        phi_filt = p_dec_filt.sample()
        phi_pred = p_dec_pred.sample()
        if self.config.int_steps > 0:
            phi_vae = self.diff_steps(phi_vae)
            phi_smooth = self.diff_steps(phi_smooth)
            phi_filt = self.diff_steps(phi_filt)
            phi_pred = self.diff_steps(phi_pred)

        # Image distributions and samples
        y_mu_vae = self.warp([phi_vae, y[:,0,...]])
        y_mu_vae = tf.reshape(y_mu_vae, (-1, y.shape[1], np.prod(y.shape[2:4])))
        y_sigma = self.y_sigma(y_mu_vae)
        out_dist_vae =  self.output_dist(tf.concat([y_mu_vae, y_sigma], axis=-1))
        y_vae = out_dist_vae.sample()

        y_mu_smooth = self.warp([phi_smooth, y[:,0,...]])
        y_mu_smooth = tf.reshape(y_mu_smooth, (-1, y.shape[1], np.prod(y.shape[2:4])))
        out_dist_smooth =  self.output_dist(tf.concat([y_mu_smooth, y_sigma], axis=-1))
        y_smooth = out_dist_smooth.sample()

        y_mu_filt = self.warp([phi_filt, y[:,0,...]])
        y_mu_filt = tf.reshape(y_mu_filt, (-1, y.shape[1], np.prod(y.shape[2:4])))
        out_dist_filt =  self.output_dist(tf.concat([y_mu_filt, y_sigma], axis=-1))
        y_filt = out_dist_filt.sample()

        y_mu_pred = self.warp([phi_pred, y[:,0,...]])
        y_mu_pred = tf.reshape(y_mu_pred, (-1, y.shape[1], np.prod(y.shape[2:4])))
        out_dist_pred =  self.output_dist(tf.concat([y_mu_pred, y_sigma], axis=-1))
        y_pred = out_dist_pred.sample()
        
        return {'image_data': {'vae': {'images' : y_vae, 'flows': phi_vae},
                               'smooth': {'images': y_smooth, 'flows': phi_smooth},
                               'filt': {'images': y_filt, 'flows': phi_filt},
                               'pred': {'images': y_pred, 'flows': phi_pred}},
                'latent_dist': latent_dist,
                'x_obs': x_sample}

    def set_loss(self, y, mask, out_dist, p_dec, q_enc, x_sample, z_sample, p_smooth):
        super().set_loss(y=y, 
                         mask=mask, 
                         p_dec=out_dist, 
                         q_enc=q_enc, 
                         x_sample=x_sample, 
                         z_sample=z_sample, 
                         p_smooth=p_smooth)
        
        grad = grad_loss('l2', p_dec.mean())
        self.grad_flow_metric.update_state(grad)
        if 'grad' in self.config.losses:
            self.add_loss(grad)
        return