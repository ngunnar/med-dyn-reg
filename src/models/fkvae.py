import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from voxelmorph.tf.layers import SpatialTransformer as SpatialTransformer
from voxelmorph.tf.layers import VecInt
from voxelmorph.tf.losses import Grad

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
        
        self.grad_loss = Grad(penalty='l2', )
        self.grad_flow_metric = tfk.metrics.Mean(name = 'grad flow ↓')        

    def warping(self, inputs):
        phi = inputs[0]
        y_0 = inputs[1]
        _, length, dim_y, _, _ = phi.shape
        y_0 = tf.repeat(y_0[:,None,...], length, axis=1)
        images = tf.reshape(y_0, (-1, *(dim_y,dim_y), 1))

        flows = tf.reshape(phi, (-1, *(dim_y,dim_y), 2))
        y_pred = self.stn([images, flows])
        y_pred = tf.reshape(y_pred, (-1, length, *(dim_y,dim_y)))
        return y_pred

    def diff_steps(self, phi):
        dim_y = phi.shape[2:4]
        length = phi.shape[1]
        phi = tf.reshape(phi, (-1, *dim_y, 2))
        phi= self.vecInt(phi)
        phi = tf.reshape(phi, (-1, length, *dim_y, 2))
        return phi

    def call(self, inputs, training):
        y = inputs['input_video']
        y_ref = inputs['input_ref']
        mask = inputs['input_mask'] 
        
        #q_enc, x, x_ref, p_smooth, z, p_dec = self.forward(inputs, training)
        q_enc, x, x_ref, log_pred, log_filt, log_p_1, log_smooth, ll, p_dec = self.forward(inputs, training)

        phi = p_dec.sample()
        
        if self.config.int_steps > 0:
            phi = self.diff_steps(phi)
        
        y_mu = self.warp([phi, y_ref])
        y_mu = tf.reshape(y_mu, (-1, y.shape[1], np.prod(y.shape[2:4])))
        y_sigma = self.y_sigma(y_mu)
        out_dist =  self.output_dist(tf.concat([y_mu, y_sigma], axis=-1))        

        self.set_loss(y, mask, out_dist, p_dec, q_enc, x, x_ref, log_pred, log_filt, log_p_1, log_smooth, ll)
        return

    def eval(self, inputs):
        y = inputs['input_video']
        y_ref = inputs['input_ref']
        mask = inputs['input_mask'] 
        length = y.shape[1]
        
        q_enc, q_ref_enc, x_ref_feat = self.encoder(y, y_ref, training=False)         
        x = q_enc.sample()
        x_ref = q_ref_enc.sample()
        
        # Latent distributions 
        latent_dist = self.lgssm.get_distribtions(x, x_ref, mask)
        
        # Flow distributions         
        p_dec_vae = self.dec(x, x_ref, x_ref_feat, length, False)
        #p_dec_smooth = self.dec(latent_dist['smooth'].sample(), x_ref, x_ref_feat, length, False)
        #p_dec_filt = self.dec(latent_dist['filt'].sample(), x_ref, x_ref_feat, length, False)
        #p_dec_pred = self.dec(latent_dist['pred'].sample(), x_ref, x_ref_feat, length, False)
        
        p_dec_smooth = self.dec(latent_dist['smooth'].mean(), x_ref, x_ref_feat, length, False)
        p_dec_filt = self.dec(latent_dist['filt'].mean(), x_ref, x_ref_feat, length, False)
        p_dec_pred = self.dec(latent_dist['pred'].mean(), x_ref, x_ref_feat, length, False)

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
        y_mu_vae = self.warp([phi_vae, y_ref])
        y_mu_vae = tf.reshape(y_mu_vae, (-1, y.shape[1], np.prod(y.shape[2:4])))
        y_sigma = self.y_sigma(y_mu_vae)
        out_dist_vae =  self.output_dist(tf.concat([y_mu_vae, y_sigma], axis=-1))
        y_vae = out_dist_vae.sample()

        y_mu_smooth = self.warp([phi_smooth, y_ref])
        y_mu_smooth = tf.reshape(y_mu_smooth, (-1, y.shape[1], np.prod(y.shape[2:4])))
        out_dist_smooth =  self.output_dist(tf.concat([y_mu_smooth, y_sigma], axis=-1))
        y_smooth = out_dist_smooth.sample()

        y_mu_filt = self.warp([phi_filt, y_ref])
        y_mu_filt = tf.reshape(y_mu_filt, (-1, y.shape[1], np.prod(y.shape[2:4])))
        out_dist_filt =  self.output_dist(tf.concat([y_mu_filt, y_sigma], axis=-1))
        y_filt = out_dist_filt.sample()

        y_mu_pred = self.warp([phi_pred, y_ref])
        y_mu_pred = tf.reshape(y_mu_pred, (-1, y.shape[1], np.prod(y.shape[2:4])))
        out_dist_pred =  self.output_dist(tf.concat([y_mu_pred, y_sigma], axis=-1))
        y_pred = out_dist_pred.sample()
        
        return {'image_data': {'vae': {'images' : y_vae, 'flows': phi_vae},
                               'smooth': {'images': y_smooth, 'flows': phi_smooth},
                               'filt': {'images': y_filt, 'flows': phi_filt},
                               'pred': {'images': y_pred, 'flows': phi_pred}},
                'latent_dist': latent_dist,
                'x_obs': x,
                'x_ref': x_ref,
                'x_ref_feat': x_ref_feat}

    def set_loss(self, y, mask, out_dist, p_dec, q_enc, x, x_ref, log_pred, log_filt, log_p_1, log_smooth, ll):
        super().set_loss(y=y, 
                         mask=mask, 
                         p_dec=out_dist, 
                         q_enc=q_enc, 
                         x=x, 
                         x_ref=x_ref,
                         log_pred = log_pred,
                         log_filt = log_filt,
                         log_p_1 = log_p_1,
                         log_smooth = log_smooth,
                         ll=ll)
        
        grad = self.grad_loss.loss(None, p_dec.mean())
        self.grad_flow_metric.update_state(grad)
        if 'grad' in self.config.losses:
            self.add_loss(grad)
        return

    @tf.function
    def reconstruct(self, x, y_ref, x_ref, x_ref_feat):
        phi = self.dec(x, x_ref, x_ref_feat, 1, False).sample()
        if self.config.int_steps > 0:
            phi = self.diff_steps(phi)

        y_mu = self.warp([phi, y_ref])
        y_mu = tf.reshape(y_mu, (-1, 1, np.prod(y_ref.shape[1:3])))
        y_sigma = self.y_sigma(y_mu)
        out_dist =  self.output_dist(tf.concat([y_mu, y_sigma], axis=-1))
        return out_dist
        
