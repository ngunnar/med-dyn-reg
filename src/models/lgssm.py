import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm

tfk = tf.keras
tfpl = tfp.layers


def get_cholesky(A):
    L = tfp.experimental.distributions.marginal_fns.retrying_cholesky(A, jitter=None, max_iters=5, name='retrying_cholesky')
    return L[0]

class LGSSM(tfk.Model):
    def __init__(self, config, name='LGSSM', **kwargs):
        super(LGSSM, self).__init__(name=name, **kwargs)
        self.dim_z = config.dim_z
        self.dim_x = config.dim_x
        self.config = config
        
        ## Parameters
        A_init = tf.random_normal_initializer()
        self.A = tf.Variable(initial_value=A_init(shape=(config.dim_z,config.dim_z)),
                             trainable=config.trainable_A, 
                             dtype="float32", 
                             name="A")
        
        C_init = tf.random_normal_initializer()
        self.C = tf.Variable(initial_value=C_init(shape=(config.dim_x, config.dim_z)), 
                             trainable=config.trainable_C, 
                             dtype="float32", 
                             name="C")
        # w ~ N(0,Q)
        init_Q_np = np.ones(config.dim_z).astype('float32') * config.noise_transition
            
        init_Q = tf.constant_initializer(init_Q_np)
        self.Q = tf.Variable(initial_value=init_Q(shape=init_Q_np.shape), trainable=config.trainable_Q, dtype='float32', name="Q")
        self.transition_noise = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(config.dim_z, dtype='float32'), 
                                                                        scale_diag=self.Q) #FULLY_REPARAMETERIZED by default
        # v ~ N(0,R)
        init_R_np = np.ones(config.dim_x).astype('float32') * config.noise_emission
        init_R = tf.constant_initializer(init_R_np)
        self.R = tf.Variable(initial_value=init_R(shape=init_R_np.shape), trainable=config.trainable_R, dtype='float32', name="R" )
        #FULLY_REPARAMETERIZED by default
        self.observation_noise = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(config.dim_x, dtype='float32'),
                                                                         scale_diag=self.R) 
        # Trainable        
        self.trainable_params = []
        self.trainable_params.append(self.A) if config.trainable_A else None
        self.trainable_params.append(self.C) if config.trainable_C else None
        self.trainable_params.append(self.Q) if config.trainable_Q else None
        self.trainable_params.append(self.R) if config.trainable_R else None
                
        self.length = config.length
        self.initial_state_dist = tfpl.IndependentNormal
        self.dense1 = tf.keras.layers.Dense(self.dim_x, activation="relu", name="init_1")
        self.dense2 = tf.keras.layers.Dense(self.dim_x, activation="relu", name="init_2")
        self.dense_mu = tf.keras.layers.Dense(self.dim_z, name="init_mu_out")
        self.dense_scale = tf.keras.layers.Dense(self.dim_z, activation='softplus', name="init_scale_out")
		
		
        self.initial_prior = tfp.distributions.MultivariateNormalDiag(scale_diag=tf.ones([config.dim_z]))
        self.length = config.length
    
    @property
    def lgssm(self):
        return tfp.distributions.LinearGaussianStateSpaceModel(num_timesteps = self.length,
                                                               transition_matrix = self.A, 
                                                               transition_noise = self.transition_noise, 
                                                               observation_matrix = self.C,
                                                               observation_noise = self.observation_noise, 
                                                               initial_state_prior = self.initial_prior, 
                                                               initial_step=0,
                                                               validate_args=False, 
                                                               allow_nan_stats=True,
                                                               name='LinearGaussianStateSpaceModel')
	
    #@tf.function
    def initial_prior_network(self, x, x_ref):
        x = self.dense1(x)
        x = self.dense2(x)
        
        mu = self.dense_mu(x)
        sigma = self.dense_scale(x)
		
        self.initial_prior = tfp.distributions.MultivariateNormalDiag(loc = mu, scale_diag = sigma)    
  
    def call(self, inputs):
        x = inputs[0]
        x_ref = inputs[1]
        mask = inputs[2]

        # Initialize state-space model
        self.initial_prior_network(x[:,0,:], x_ref)
        self.length = x.shape[1]
        #model = self.get_LGSSM(x[:,0,:], x_ref, x.shape[1])
        
        # Run the smoother and draw sample from i
        # p(z[t]|x[:T])  
        mu_s, P_s = self.lgssm.posterior_marginals(x, mask = mask)
        p_smooth = tfp.distributions.MultivariateNormalTriL(mu_s, get_cholesky(P_s))
        z = p_smooth.sample()

        # Get p(z[t+1] | x[:t]) and p(z[t] | x[:t])
        ll, mu_f, P_f, mu_p, P_p, _, _ = self.lgssm.forward_filter(x, mask=mask)

        # p(z[t+1] | x[:t]) -> p(z[t] | z[t-1]) for t = 2,...T, z = [z_1, z_2, ..., z_T]
        p_pred = tfp.distributions.MultivariateNormalTriL(mu_p[:,:-1,:], get_cholesky(P_p[:,:-1,:]))
        
        # p(z[t] | x[:t]) -> p(x[t] | x[:t])
        mu_obsfilt, P_obsfilt = self.lgssm.latents_to_observations(mu_f, P_f)
        p_obs_filt = tfp.distributions.MultivariateNormalTriL(mu_obsfilt, get_cholesky(P_obsfilt))
        
        log_pred, log_filt, log_p_1, log_smooth = self._get_loss(x, z, p_smooth, p_obs_filt, p_pred)
        return log_pred, log_filt, log_p_1, log_smooth, ll
    
    def _get_loss(self, x, z, p_smooth, p_obs_filt, p_pred):
        """
        Get log probability densitity functions for the kalman filter
        ```
        z_t ~ N(μ_{t|T}, ∑_{t|T}) for t = 1,...,T
        log p(z_t|z_{t-1}) = log N(z_t | A z_{t-1}, R) = log N(z_t - Az_{t-1} | 0, R) for t = 2,...,T 
        log p(x_t|z_t) = log N (x_t | Cz_t, Q) = log N(x_t - Cz_t | 0, Q) for t = 1,...,T
        log p(z_1) = log N(z_1 | μ_0, ∑_0)
        log p(z_t|x_{1:T}) = log N(z_t | μ_{t|T}, ∑_{t|T}) for t = 1,...,T
        ```
        
        Args:
            x_: encoder sample
            z: smooth sample
            p_smooth: smooth distribution p(z_t | z_{1:T})
            
        Returns:
            Prediction : log p(z[t] | z[:t-1]) for t = 2,...,T
            Filtering : log p(x[t] | z[t]) for t = 1,..., T
            Initial : log p(z[1] | x_M)
            Smoothing : log p(z[t] | x[1:T]) for t = 1,..., T
            Log-likelihood: log p(x[t] | x[:t-1])
        """
        
        # log p(z[t] | x[1:T])
        log_smooth = p_smooth.log_prob(z)
        
        # log p(z[1] | x_ref), TODO: is this z_1 or z_0?
        z_1 = z[:, 0, :]
        log_p_1 = self.lgssm.initial_state_prior.log_prob(z_1)
        
        # log p(z[t] | z[t-1]) for t = 2,...T, z = [z_1, z_2, ..., z_T]
        log_pred = p_pred.log_prob(z[:,1:,:])
        
        # log p(x[t] | x[:t]) for t = 1,...,T
        log_filt = p_obs_filt.log_prob(x)
        
        return log_pred, log_filt, log_p_1, log_smooth

    def get_obs_distributions(self, x, mask):        
        steps = x.shape[1]
        #Smooth
        p_smooth, p_obssmooth = self.get_smooth_dist(x, mask, steps = steps)

        # Filter        
        p_filt, p_obsfilt, p_obspred = self.get_filter_dist(x, mask, steps = steps, get_pred=True)   

        return {"smooth_mean": p_obssmooth.mean(),
                "smooth_cov": p_obssmooth.covariance(),
                "filt_mean": p_obsfilt.mean(),
                "filt_cov": p_obsfilt.covariance(),
                "pred_mean": p_obspred.mean(), 
                "pred_cov": p_obspred.covariance(),
                "x": x}
        
    def get_distribtions(self, x, x_ref, mask):
        self.initial_prior_network(x[:,0,:], x_ref)
        self.length = x.shape[1]
        
        #Smooth
        mu, P = self.lgssm.posterior_marginals(x, mask = mask)
        p_smooth = tfp.distributions.MultivariateNormalTriL(loc=mu,
                                                            scale_tril=get_cholesky(P))
        mu, P = self.lgssm.latents_to_observations(mu, P)
        p_obssmooth = tfp.distributions.MultivariateNormalTriL(loc=mu, 
                                                               scale_tril=get_cholesky(P))
        		
		
        #_, p_obssmooth = self.get_smooth_dist(x, x_ref, mask, steps = steps)

        # Filter        
        #model = self.get_LGSSM(x[:,0,:], x_ref, steps)
        _, mu_f, P_f, _, _, mu_obsp, P_obsp = self.lgssm.forward_filter(x, mask=mask)
        # Obs filt dist
        mu_obsfilt, P_obsfilt = self.lgssm.latents_to_observations(mu_f, P_f)
        p_obsfilt = tfp.distributions.MultivariateNormalTriL(mu_obsfilt, get_cholesky(P_obsfilt))
        
		# Obs pred dist p(x[t] | x[:t-1])
        p_obspred = tfp.distributions.MultivariateNormalTriL(mu_obsp, get_cholesky(P_obsp))
        
        
        return {"smooth": p_obssmooth,
                "filt": p_obsfilt,
                "pred": p_obspred}

    def get_smooth_dist(self, x, x_ref, mask, steps = None):
        self.initial_prior_network(x[:,0,:], x_ref)
        self.length = x.shape[1]
        #model = self.get_LGSSM(x[:,0,:], x_ref, steps)
        mu, P = self.lgssm.posterior_marginals(x, mask = mask)
        p_smooth = tfp.distributions.MultivariateNormalTriL(loc=mu,
                                                            scale_tril=get_cholesky(P))
        mu, P = self.lgssm.latents_to_observations(mu, P)
        p_obssmooth = tfp.distributions.MultivariateNormalTriL(loc=mu, 
                                                               scale_tril=get_cholesky(P))
        return p_smooth, p_obssmooth

    def get_filter_dist(self, x, x_ref, mask, get_pred=False, steps = None):
        self.initial_prior_network(x[:,0,:], x_ref)
        self.length = x.shape[1]
        _, mu_f, P_f, _, _, mu_obsp, P_obsp = self.lgssm.forward_filter(x, mask=mask)
        # Filt dist
        p_filt = tfp.distributions.MultivariateNormalTriL(mu_f, get_cholesky(P_f))
        # Obs filt dist
        mu_obsfilt, P_obsfilt = self.lgssm.latents_to_observations(mu_f, P_f)
        p_obsfilt = tfp.distributions.MultivariateNormalTriL(mu_obsfilt, get_cholesky(P_obsfilt))
        
        if get_pred:
            # Obs pred dist p(x[t] | x[:t-1])
            p_obspred = tfp.distributions.MultivariateNormalTriL(mu_obsp, get_cholesky(P_obsp))
        
            return p_filt, p_obsfilt, p_obspred
        
        return p_filt, p_obsfilt
        
class FineTunedLGSSM(LGSSM):
    def __init__(self, config, loss_metric, name='LGSSM', **kwargs):
        super(FineTunedLGSSM, self).__init__(config, name=name, **kwargs)  
        self.loss_metric = loss_metric

    def call(self, inputs):
        x = inputs['x']
        x_ref = inputs['x_ref']
        mask = inputs['mask']
        
        self.initial_prior_network(x[:,0,:], x_ref)
        self.length = x.shape[1]

        #m = self.get_LGSSM(x[:,0,:], x_ref, steps)
        ll, mu_f, P_f, mu_p, P_p, _, _ = self.lgssm.forward_filter(x, mask=mask)
        mu_obsfilt, P_obsfilt = self.lgssm.latents_to_observations(mu_f, P_f)
        p_obsfilt = tfp.distributions.MultivariateNormalTriL(mu_obsfilt, get_cholesky(P_obsfilt))

        loss = -tf.reduce_sum(ll, axis=1)
        self.add_loss(loss)
        return p_obsfilt

        '''
        mu_s, P_s = m.posterior_marginals(x, mask = mask)
        p_smooth = tfp.distributions.MultivariateNormalTriL(mu_s, get_cholesky(P_s))
        z = p_smooth.sample()
        log_pred, log_filt, log_p_1, log_smooth, ll = self.get_loss(x, x_ref, z, p_smooth, mask)
        log_p_xz = tf.reduce_sum(log_filt, axis=1) + log_p_1 + tf.reduce_sum(log_pred, axis=1)
        log_p_z_x = tf.reduce_sum(log_smooth, axis=1)
        
        if self.loss_metric == 'll':
            loss = -tf.reduce_sum(ll, axis=1)
        else:            
            loss = -(log_p_xz - log_p_z_x)
        self.add_loss(loss)
        self.add_metric(log_p_xz, name='log p(z[t],x[t]) ↑', aggregation="mean")
        self.add_metric(log_p_z_x, name='log p(z[t]|x[t]) ↓', aggregation="mean")
        self.add_metric(tf.reduce_sum(ll, axis=1), name='log p(x[t]|x[:t-1]) ↑', aggregation="mean")
        #self.log_pzx_metric.update_state(log_p_xz)
        #self.log_pz_x_metric.update_state(log_p_z_x)
        #self.log_px_x_metric.update_state(tf.reduce_sum(ll, axis=1))

        return ll
        '''
    
    def train(self, data, learning_rate = 0.001, epochs:int = 300, val_data = None, callbacks=None):
        self.compile(optimizer = tf.keras.optimizers.Adam(learning_rate))
        self.history = self.fit(x = data, epochs = epochs, validation_data = val_data, callbacks = callbacks)


class OnlineTraining(LGSSM):
    def __init__(self, weights, lr, config, name='LGSSM', **kwargs):
        super(OnlineTraining, self).__init__(config, name=name, **kwargs)

        self.optimizer = tf.keras.optimizers.Adam(lr)

        self.set_weights(weights)

        self.dense1.trainable = False
        self.dense2.trainable = False
        self.dense_mu.trainable = False
        self.dense_scale.trainable = False
        #self.R = tf.constant(self.R)
        #self.Q = tf.constant(self.Q)
        #self.C = tf.constant(self.C)
    
    def call(self, inputs):
        x = inputs['x']
        mask = inputs['mask']
        
        ll, mu_f, P_f, mu_p, P_p, _, _ = self.lgssm.forward_filter(x, mask=mask)
        
        p_filt = tfp.distributions.MultivariateNormalTriL(mu_f, get_cholesky(P_f))
        
        mu_obsfilt, P_obsfilt = self.lgssm.latents_to_observations(mu_f, P_f)
        p_obsfilt = tfp.distributions.MultivariateNormalTriL(mu_obsfilt, get_cholesky(P_obsfilt))

        loss = -tf.reduce_mean(ll, axis=1)
        self.add_loss(loss)
        return p_filt, p_obsfilt

    @tf.function(
    autograph=False,
    input_signature=[
        {'x' : tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
        'mask': tf.TensorSpec(shape=[None,None], dtype=tf.bool)}
        ],
    )
    def predict(self, inputs):
        x = inputs['x']
        mask = inputs['mask']
        
        ll, mu_f, P_f, _, _, _, _ = self.lgssm.forward_filter(x, mask=mask)
        p_filt = tfp.distributions.MultivariateNormalTriL(mu_f, get_cholesky(P_f))

        mu_obsfilt, P_obsfilt = self.lgssm.latents_to_observations(mu_f, P_f)
        p_obsfilt = tfp.distributions.MultivariateNormalTriL(mu_obsfilt, get_cholesky(P_obsfilt))
        loss = -tf.reduce_mean(ll, axis=1)
        return p_filt, p_obsfilt, loss

    @tf.function(
    autograph=False,
    input_signature=[
        {'x' : tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
        'mask': tf.TensorSpec(shape=[None,None], dtype=tf.bool)}
        ],
    )
    def train_step(self, data):
        with tf.GradientTape() as tape:
            p_filt, p_obsfilt = self(data, training=True)
            loss_value = self.losses
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))    
        return p_filt, p_obsfilt, loss_value