import tensorflow as tf
import tensorflow_probability as tfp

from voxelmorph.tf.layers import SpatialTransformer as SpatialTransformer
from voxelmorph.tf.layers import VecInt
from voxelmorph.tf.losses import Grad

tfk = tf.keras
tfpl = tfp.layers
tfd = tfp.distributions

from .layers_skip import Decoder
from .kvae import KVAE
from .utils import set_name

class fKVAE(KVAE):
    def __init__(self, config, name="fKVAE", prefix=None, **kwargs):
        super(fKVAE, self).__init__(config = config, name=name, prefix=prefix, **kwargs)
        self.decoder = Decoder(self.config.skip_connection, self.config, output_channels = 2, prefix=prefix)        
        
        self.output_dist = tfpl.IndependentNormal(config.dim_y, name=set_name('output_dist', prefix))
        self.y_sigma = lambda y: tf.ones_like(y, dtype='float32') * tfp.math.softplus_inverse(0.01)
        self.stn = SpatialTransformer()
        self.warp = tf.keras.layers.Lambda(lambda x: self.warping(x), name=set_name('warping', prefix))
        self.w_g = tf.Variable(initial_value=1., trainable=False, dtype="float32", name=set_name("w_g", prefix))
        #external_mask = tf.convert_to_tensor(np.load(config.ds_path + '/external_mask.npy'))
        #self.external_mask = tf.repeat(external_mask[...,None], 2, axis=-1)
        if self.config.int_steps > 0:            
            self.vecInt = VecInt(method='ss', name=set_name('s_flow_int', prefix), int_steps=self.config.int_steps)
        
        self.grad_loss = Grad(penalty='l2')
        self.grad_flow_metric = tfk.metrics.Mean(name = set_name('grad flow ↓', prefix))        

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
        y, y0, mask = self.parse_inputs(inputs)
        
        q_x, x, _, _, log_pred, log_filt, log_p_1, log_smooth, ll, p_phi = self.forward(y, y0, mask, training)
        phi = p_phi.sample()
        
        if self.config.int_steps > 0:
            phi = self.diff_steps(phi)
        
        y_mu = self.warp([phi, y0])
        y_mu = tf.reshape(y_mu, (-1, y.shape[1], y.shape[2] * y.shape[3]))
        y_sigma = self.y_sigma(y_mu)
        p_y =  self.output_dist(tf.concat([y_mu, y_sigma], axis=-1))        

        self.set_loss(y, mask, p_y, p_phi, q_x, x, log_pred, log_filt, log_p_1, log_smooth, ll)
        return

    @tf.function
    def eval(self, inputs):
        y, y0, mask = self.parse_inputs(inputs)
        length = y.shape[1]
        
        q_x, s, s_feats = self.encoder(y, y0, training=False)         
        x = q_x.sample()
        
        # Latent distributions 
        p_obssmooth, p_obsfilt, p_obspred = self.lgssm.get_distribtions(x, mask)
        
        # Flow distributions         
        p_phi_vae = self.dec(x, s, s_feats, length, False)
        p_phi_smooth = self.dec(p_obssmooth.mean(), s, s_feats, length, False)
        p_phi_filt = self.dec(p_obsfilt.mean(), s, s_feats, length, False)
        p_phi_pred = self.dec(p_obspred.mean(), s, s_feats, length, False)

        # Flow samples
        phi_vae = p_phi_vae.sample()
        phi_smooth = p_phi_smooth.sample()
        phi_filt = p_phi_filt.sample()
        phi_pred = p_phi_pred.sample()
        if self.config.int_steps > 0:
            phi_vae = self.diff_steps(phi_vae)
            phi_smooth = self.diff_steps(phi_smooth)
            phi_filt = self.diff_steps(phi_filt)
            phi_pred = self.diff_steps(phi_pred)

        # Image distributions and samples
        last_channel = y.shape[2] * y.shape[3]
        y_mu_vae = self.warp([phi_vae, y0])
        y_mu_vae = tf.reshape(y_mu_vae, (-1, y.shape[1], last_channel))
        y_sigma = self.y_sigma(y_mu_vae)
        p_y_vae =  self.output_dist(tf.concat([y_mu_vae, y_sigma], axis=-1))
        y_vae = p_y_vae.sample()

        y_mu_smooth = self.warp([phi_smooth, y0])
        y_mu_smooth = tf.reshape(y_mu_smooth, (-1, y.shape[1], last_channel))
        p_y_smooth =  self.output_dist(tf.concat([y_mu_smooth, y_sigma], axis=-1))
        y_smooth = p_y_smooth.sample()

        y_mu_filt = self.warp([phi_filt, y0])
        y_mu_filt = tf.reshape(y_mu_filt, (-1, y.shape[1], last_channel))
        p_y_filt =  self.output_dist(tf.concat([y_mu_filt, y_sigma], axis=-1))
        y_filt = p_y_filt.sample()

        y_mu_pred = self.warp([phi_pred, y0])
        y_mu_pred = tf.reshape(y_mu_pred, (-1, y.shape[1], last_channel))
        p_y_pred =  self.output_dist(tf.concat([y_mu_pred, y_sigma], axis=-1))
        y_pred = p_y_pred.sample()
        
        return {'image_data': {'vae': {'images' : y_vae, 'flows': phi_vae},
                               'smooth': {'images': y_smooth, 'flows': phi_smooth},
                               'filt': {'images': y_filt, 'flows': phi_filt},
                               'pred': {'images': y_pred, 'flows': phi_pred}},
                'latent_dist': {'smooth': p_obssmooth, 'filt': p_obsfilt, 'pred': p_obspred},
                'x_obs': x,
                's': s,
                's_feat': s_feats}

    def set_loss(self, y, mask, p_y, p_phi, q_x, x, log_pred, log_filt, log_p_1, log_smooth, ll):
        super().set_loss(y=y,
                         mask=mask, 
                         p_y=p_y,
                         q_x=q_x, 
                         x=x, 
                         log_pred = log_pred,
                         log_filt = log_filt,
                         log_p_1 = log_p_1,
                         log_smooth = log_smooth,
                         ll=ll)
        
        grad = self.grad_loss.loss(None, p_phi.mean())
        self.grad_flow_metric.update_state(grad)
        if 'grad' in self.config.losses:
            #self.add_loss(self.w_g * grad)
            self.add_loss((tf.reduce_mean(self.w_g * grad)))
        return

    @tf.function
    def reconstruct(self, x, y0, s, s_feat):
        phi = self.dec(x, s, s_feat, 1, False).sample()
        if self.config.int_steps > 0:
            phi = self.diff_steps(phi)

        y_mu = self.warp([phi, y0])
        y_mu = tf.reshape(y_mu, (-1, 1, y0.shape[1]*y0.shape[2]))
        y_sigma = self.y_sigma(y_mu)
        p_y =  self.output_dist(tf.concat([y_mu, y_sigma], axis=-1))
        return p_y
        
