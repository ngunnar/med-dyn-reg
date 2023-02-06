import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm

tfk = tf.keras
tfpl = tfp.layers


def get_cholesky(A):
    L = tfp.experimental.distributions.marginal_fns.retrying_cholesky(A, jitter=None, max_iters=5, name='retrying_cholesky')
    return L[0]


def isPD(B, name):
    """Returns true when input is positive-definite, via Cholesky"""
    S = True
    A = tf.linalg.cholesky(B)
    if tf.reduce_any(tf.math.is_nan(A)):
        S = False
        tqdm.write("{0}, {1}: {2}".format(name, "Cholesky failed", "trying with retrying_cholesky..."))   
    return S, A

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
                
        self.ph_steps = config.ph_steps
        self.initial_state_dist = tfpl.IndependentNormal
        self.dense1 = tf.keras.layers.Dense(self.dim_x, activation="relu", name="init_1")
        self.dense2 = tf.keras.layers.Dense(self.dim_x, activation="relu", name="init_2")
        self.dense_mu = tf.keras.layers.Dense(self.dim_z, name="init_mu_out")
        self.dense_scale = tf.keras.layers.Dense(self.dim_z, activation='softplus', name="init_scale_out")
    
    @tf.function
    def initial_prior_network(self, x_M):
        x = self.dense1(x_M)
        x = self.dense2(x)
        
        mu = self.dense_mu(x)
        sigma = self.dense_scale(x)
        return mu, sigma
    
    def get_LGSSM(self, x_M, steps = None):
        if steps is None:
            steps = self.ph_steps
        mu, sigma = self.initial_prior_network(x_M)
        sigma += tf.keras.backend.epsilon()
        
        init_dist = tfp.distributions.MultivariateNormalDiag(loc = mu, scale_diag = sigma)
        return tfp.distributions.LinearGaussianStateSpaceModel(num_timesteps = steps,
                                                               transition_matrix = self.A, 
                                                               transition_noise = self.transition_noise, 
                                                               observation_matrix = self.C,
                                                               observation_noise = self.observation_noise, 
                                                               initial_state_prior = init_dist, 
                                                               initial_step=0,
                                                               validate_args=False, 
                                                               allow_nan_stats=True,
                                                               name='LinearGaussianStateSpaceModel')
        
    def call(self, inputs):
        x = inputs[0]
        mask = inputs[1]
        model = self.get_LGSSM(x[:,0,...])
        mu_s, P_s = model.posterior_marginals(x, mask = mask)
        p_smooth = tfp.distributions.MultivariateNormalTriL(mu_s, get_cholesky(P_s))
        
        return p_smooth
    
    def get_obs_distributions(self, x_sample, mask):        
        steps = x_sample.shape[1]
        #Smooth
        p_smooth, p_obssmooth = self.get_smooth_dist(x_sample, mask, steps = steps)

        # Filter        
        p_filt, p_obsfilt, p_pred, p_obspred = self.get_filter_dist(x_sample, mask, steps = steps, get_pred=True)   

        return {"smooth_mean": p_obssmooth.mean(),
                "smooth_cov": p_obssmooth.covariance(),
                "filt_mean": p_obsfilt.mean(),
                "filt_cov": p_obsfilt.covariance(),
                "pred_mean": p_obspred.mean(), 
                "pred_cov": p_obspred.covariance(),
                "x": x_sample}
        
    def get_distribtions(self, x_sample, mask):
        steps = x_sample.shape[1]
        #Smooth
        _, p_obssmooth = self.get_smooth_dist(x_sample, mask, steps = steps)

        # Filter        
        _, p_obsfilt, _, p_obspred = self.get_filter_dist(x_sample, mask, steps = steps, get_pred=True)   

        return {"smooth": p_obssmooth,
                "filt": p_obsfilt,
                "pred": p_obspred}

    def get_smooth_dist(self, x, mask, steps = None):
        model = self.get_LGSSM(x[:,0,...], steps)
        mu, P = model.posterior_marginals(x, mask = mask)
        p_smooth = tfp.distributions.MultivariateNormalTriL(loc=mu,
                                                            scale_tril=get_cholesky(P))
        mu, P = model.latents_to_observations(mu, P)
        p_obssmooth = tfp.distributions.MultivariateNormalTriL(loc=mu, 
                                                               scale_tril=get_cholesky(P))
        return p_smooth, p_obssmooth

    def get_filter_dist(self, x, mask, get_pred=False, steps = None):
        model = self.get_LGSSM(x[:,0,...], steps)
        _, mu_f, P_f, mu_p, P_p, mu_obsp, P_obsp = model.forward_filter(x, mask=mask)
        # Filt dist
        p_filt = tfp.distributions.MultivariateNormalTriL(mu_f, get_cholesky(P_f))
        # Obs filt dist
        mu_obsfilt, P_obsfilt = model.latents_to_observations(mu_f, P_f)
        p_obsfilt = tfp.distributions.MultivariateNormalTriL(mu_obsfilt, get_cholesky(P_obsfilt))
        
        if get_pred:
            # Pred dist
            p_pred = tfp.distributions.MultivariateNormalTriL(mu_p, get_cholesky(P_p))
            # Obs pred dist
            p_obspred = tfp.distributions.MultivariateNormalTriL(mu_obsp, get_cholesky(P_obsp))
        
            return p_filt, p_obsfilt, p_pred, p_obspred
        
        return p_filt, p_obsfilt
    
    def get_loss(self, x_sample, z_sample, p_smooth, mask):
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
            x_sample: encoder sample
            z_sample: smooth sample
            p_smooth: smooth distribution p(z_t | z_{1:T})
            
        Returns:
            Prediction : log p(z[t] | z[:t-1]) for t = 2,...,T
            Filtering : log p(x[t] | z[t]) for t = 1,..., T
            Initial : log p(z[1] | x_M)
            Smoothing : log p(z[t] | x[1:T]) for t = 1,..., T
            Log-likelihood: log p(x[t] | x[:t-1])
        """
        model = self.get_LGSSM(x_sample[:,0,...], x_sample.shape[1])
        
        # log p(z[t] | x[1:T])
        log_smooth = p_smooth.log_prob(z_sample)
        
        # log p(z[1] | x_M)
        z_1 = z_sample[:, 0, :]
        log_p_1 = model.initial_state_prior.log_prob(z_1)
            
        ll, mu_f, P_f, mu_p, P_p, _, _ = model.forward_filter(x_sample, mask=mask)
        
        # log p(z[t] | z[:t-1])
        p_pred = tfp.distributions.MultivariateNormalTriL(mu_p, get_cholesky(P_p))
        log_pred = p_pred.log_prob(z_sample)
        
        # log p(x[t] | z[t])
        mu_obsfilt, P_obsfilt = model.latents_to_observations(mu_f, P_f)
        p_filt = tfp.distributions.MultivariateNormalTriL(mu_obsfilt, get_cholesky(P_obsfilt))
        log_filt = p_filt.log_prob(x_sample)
        
        return log_pred, log_filt, log_p_1, log_smooth, ll
        
class FineTunedLGSSM(LGSSM):
    def __init__(self, config, name='LGSSM', **kwargs):
        super(FineTunedLGSSM, self).__init__(config, name=name, **kwargs)    
    
    def call(self, inputs):
        x = inputs['x']
        mask = inputs['mask']
        steps = x.shape[1]
        m = self.get_LGSSM(x[:,0,...], steps)

        '''
        mu_s, P_s = m.posterior_marginals(x, mask = mask)
        p_smooth = tfp.distributions.MultivariateNormalTriL(mu_s, get_cholesky(P_s))
        z_sample = p_smooth.sample()

        log_pred, log_filt, log_p_1, log_smooth, ll = self.get_loss(x, z_sample, p_smooth, mask)

        # log p(x, z)
        log_p_xz = tf.reduce_sum(log_filt, axis=1) + log_p_1 + tf.reduce_sum(log_pred[:,1:], axis=1)

        # log p(z|x)
        log_p_z_x = tf.reduce_sum(log_smooth, axis=1)

        #
        loss = -(log_p_xz - log_p_z_x)
        '''
        ll, _, _, _, _, _, _ = m.forward_filter(x, mask=mask)
        loss = -tf.reduce_mean(ll)        
        self.add_loss(loss)
        
        return ll
    
    def train(self, data, learning_rate = 0.001, epochs:int = 300, val_data = None, callbacks=None):
        self.compile(optimizer = tf.keras.optimizers.Adam(learning_rate))
        self.history = self.fit(x = data, epochs = epochs, validation_data = val_data, callbacks = callbacks)