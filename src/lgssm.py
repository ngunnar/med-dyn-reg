import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm
import inspect

tfk = tf.keras

tfpl = tfp.layers

# tfp.distributions.LinearGaussianStateSpaceModel.posterior_marginals can return non positive covariance matrix (round error?) 
def get_cholesky(A, training):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    L = tfp.experimental.distributions.marginal_fns.retrying_cholesky(A, jitter=None, max_iters=5, name='retrying_cholesky')
    #if training:
    #    is_finite = tf.reduce_any(tf.math.is_nan(L[0]))
    #    check_op = tf.Assert(is_finite, ["get_cholesky failed:", L[0]])
    
    
    '''
    is_pd, A_cholesky = isPD(A, name)
    if is_pd:
        return A_cholesky
    L = tfp.experimental.distributions.marginal_fns.retrying_cholesky(A, jitter=None, max_iters=5, name='retrying_cholesky')
    if tf.reduce_any(tf.math.is_nan(L[0])):
        tqdm.write("{0}, {1}: no further tries.".format(name, "retrying_cholesky"))
        tf.Assert(False, ["get_cholesky failed"])
        #tf.debugging.assert_all_finite(L[0], "get_cholesky failed", name="get_cholesky")
    
    '''
    return L[0]
    '''
    is_pd, A_cholesky = isPD(A)
    if is_pd:
        return A_cholesky
    
    B = (A + tf.transpose(A, perm=[0,1,3,2])) / 2
    s, u, V = tf.linalg.svd(B)
    V = tf.transpose(V, perm=[0,1,3,2])
    S = tf.linalg.diag(s)
    H = tf.matmul(V, tf.matmul(S, V), transpose_a=True)
    A2 = (B + H) / 2

    A3 = (A2 + tf.transpose(A2, perm=[0,1,3,2])) / 2
    
    is_pd, A_cholesky = isPD(A3)
    if is_pd:
        tqdm.write("0 {0}".format(name))
        return A_cholesky

    #spacing = tf.spacing(tf.norm(A))
    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = tf.eye(A.shape[-1])
    k = 1
    while True:
        tqdm.write("{0} {1}".format(k, name))
        #mineig = np.min(np.real(la.eigvals(A3)))
        is_pd, A_cholesky = isPD(A3)
        if is_pd:
            return A_cholesky
        
        mineig = tf.math.reduce_min(tf.math.real(tf.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3
    '''


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
        '''        
        # mu_0
        init_mu_np = np.random.randn(config.dim_z).astype(np.float32)         
        init_mu = tf.constant_initializer(init_mu_np)
        self.mu = tf.Variable(initial_value=init_mu(shape=init_mu_np.shape), trainable=config.trainable_mu, dtype='float32', name="mu")
        

        # Sigma_0
        if not config.sigma_full:
            init_sigma_np = np.ones(config.dim_z).astype(np.float32) * config.init_cov
        else:    
            init_sigma_np = np.linalg.cholesky(config.init_cov * np.eye(config.dim_z, dtype='float32'))
        init_sigma = tf.constant_initializer(init_sigma_np)
        self.Sigma = tf.Variable(initial_value=init_sigma(shape=init_sigma_np.shape), trainable=config.trainable_sigma, dtype='float32', name="sigma")
        
        # z_1 ~ N(mu_0, Sigma_0)
        #FULLY_REPARAMETERIZED by default
        if not config.sigma_full:
            self.initial_state_prior = tfp.distributions.MultivariateNormalDiag(loc = self.mu, 
                                                                       scale_diag = self.Sigma) 
        else:
            self.initial_state_prior = tfp.distributions.MultivariateNormalTriL(loc = self.mu,
                                                                           scale_tril = self.Sigma)
        '''
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
        
        #self.trainable_params.append(self.mu) if config.trainable_mu else None
        #self.trainable_params.append(self.Sigma) if config.trainable_sigma else None
        
        '''
        self.LGSSM = tfp.distributions.LinearGaussianStateSpaceModel(num_timesteps = config.ph_steps,
                                                                             transition_matrix = self.A, 
                                                                             transition_noise = self.transition_noise, 
                                                                             observation_matrix = self.C,
                                                                             observation_noise = self.observation_noise, 
                                                                             initial_state_prior = self.initial_state_prior, 
                                                                             initial_step=0,
                                                                             validate_args=False, 
                                                                             allow_nan_stats=True,
                                                                             name='LinearGaussianStateSpaceModel')
        '''
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
        
    def call(self, inputs, training):
        x = inputs[0]
        mask = inputs[1]
        model = self.get_LGSSM(x[:,0,...])
        mu_s, P_s = model.posterior_marginals(x, mask = mask)
        p_smooth = tfp.distributions.MultivariateNormalTriL(mu_s, get_cholesky(P_s, training))
        
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
        
    
    def get_smooth_dist(self, x, mask, training=False, steps = None):
        model = self.get_LGSSM(x[:,0,...], steps)
        mu, P = model.posterior_marginals(x, mask = mask)
        p_smooth = tfp.distributions.MultivariateNormalTriL(loc=mu,
                                                            scale_tril=get_cholesky(P, training))
        mu, P = model.latents_to_observations(mu, P)
        p_obssmooth = tfp.distributions.MultivariateNormalTriL(loc=mu, 
                                                               scale_tril=get_cholesky(P, training))
        return p_smooth, p_obssmooth

    def get_filter_dist(self, x, mask, training=False, get_pred=False, steps = None):
        model = self.get_LGSSM(x[:,0,...], steps)
        _, mu_f, P_f, mu_p, P_p, mu_obsp, P_obsp = model.forward_filter(x, mask=mask)
        # Filt dist
        p_filt = tfp.distributions.MultivariateNormalTriL(mu_f, get_cholesky(P_f, training))
        # Obs filt dist
        mu_obsfilt, P_obsfilt = model.latents_to_observations(mu_f, P_f)
        p_obsfilt = tfp.distributions.MultivariateNormalTriL(mu_obsfilt, get_cholesky(P_obsfilt, training))
        
        if get_pred:
            # Pred dist
            p_pred = tfp.distributions.MultivariateNormalTriL(mu_p, get_cholesky(P_p, training))
            # Obs pred dist
            p_obspred = tfp.distributions.MultivariateNormalTriL(mu_obsp, get_cholesky(P_obsp, training))
        
            return p_filt, p_obsfilt, p_pred, p_obspred
        
        return p_filt, p_obsfilt
    
    def get_loss(self, x_sample, z_sample, p_smooth):
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
            log_prob_z_z : log p(z_t | z_{t-1}) for t = 2,...,T
            log_prob_x_z : log p(x_t | z_t) for t = 1,..., T
            log_prob_0 : log p(z_1 | x_M)
            log_prob_z_x : log p(z_t | x_{1:T}) for t = 1,..., T
        """
                
        ## log p(z_t | z_{t-1}) for t = 2,...,T
        # log p(z_t | z_{t-1}) = log N(z_t | Az_{t-1}, Q) = log N(z_t - Az_{t-1}| 0, Q) = log N(z_Az|0, Q)
        model = self.get_LGSSM(x_sample[:,0,...])
        log_prob_x = model.log_prob(x_sample)
        
        A = model.transition_matrix
        transition_noise = model.transition_noise        
        Az_t = tf.matmul(A, tf.expand_dims(z_sample[:,:-1, :], axis=3))[...,0] # Az_1, ..., Az_{T-1}
        z_t = z_sample[:, 1:, :] # z_2, ..., z_T
        z_Az = z_t - Az_t
        log_prob_z_z1 = transition_noise.log_prob(z_Az)
        
        #log_p_pred = p_pred.log_prob(z_sample)[:,1:] # t = 2,...,T
        
        # obs filt dist
        ## log p(x_t | z_t) for all t = 1,...,T
        # log N(x_t | Cz_t, R)
        C = model.observation_matrix
        observation_noise = model.observation_noise
        Cz_t = tf.matmul(C, tf.expand_dims(z_sample, axis=3))[...,0]
        x_Cz_t = x_sample - Cz_t
        log_prob_x_z = observation_noise.log_prob(x_Cz_t)
        
        # smooth dist
        ## log p(z_t | x_T) for t=1,...,T
        log_p_smooth = p_smooth.log_prob(z_sample)
        
        ## log p(z_1) = log p(z_1 | z_0)
        z_1 = z_sample[:, 0, :]
        log_p_1 = model.initial_state_prior.log_prob(z_1)
        
        return log_prob_z_z1, log_prob_x_z, log_p_1, log_p_smooth, log_prob_x
       
        '''
        # Sample from smoothing distribution
        A = self.LGSSM.transition_matrix
        C = self.LGSSM.observation_matrix
        transition_noise = self.LGSSM.transition_noise
        observation_noise = self.LGSSM.observation_noise
        
        #z_tilde = latent_posterior_sample
        z_tilde = p_zt_xT.sample()
        
        ## log p(z_t | x_T) for t=1,...,T
        log_prob_z_x = p_zt_xT.log_prob(z_tilde)
        
        ## log p(x_t | z_t) for all t = 1,...,T
        # log N(x_t | Cz_t, R) -> log N(x_t - Cz_t|0, R) = log N(x_Cz_t | 0, R)
        Cz_t = tf.matmul(C, tf.expand_dims(z_tilde, axis=3))[...,0]
        x_Cz_t = x - Cz_t
        log_prob_x_z = observation_noise.log_prob(x_Cz_t)
        
        ## log p(z_1) = log p(z_1 | z_0)
        z_0 = z_tilde[:, 0, :]
        log_prob_0 = self.LGSSM.initial_state_prior.log_prob(z_0)
        
        ## log p(z_t | z_{t-1}) for t = 2,...,T
        # log p(z_t | z_{t-1}) = log N(z_t | Az_{t-1}, Q) = log N(z_t - Az_{t-1}| 0, Q) = log N(z_Az|0, Q)
        Az_t = tf.matmul(A, tf.expand_dims(z_tilde[:,:-1, :], axis=3))[...,0] # Az_1, ..., Az_{T-1}
        z_t = z_tilde[:, 1:, :] # z_2, ..., z_T
        z_Az = z_t - Az_t
        log_prob_z_z = transition_noise.log_prob(z_Az)
        
        return log_prob_z_z, log_prob_x_z, log_prob_0, log_prob_z_x
        '''
