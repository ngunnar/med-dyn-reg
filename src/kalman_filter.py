import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm
import inspect

tfk = tf.keras

# tfp.distributions.LinearGaussianStateSpaceModel.posterior_marginals can return non positive covariance matrix (round error?) 
def get_cholesky(A, name):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

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


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        A = tf.linalg.cholesky(B)
        return True, A
    except:
        return False, None

class KalmanFilter(tfk.Model):
    def __init__(self, config, name='kalman_filter', **kwargs):
        super(KalmanFilter, self).__init__(name=name, **kwargs)
        self.dim_z = config.dim_z
        self.dim_x = config.dim_x
        
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
        self.trainable_params.append(self.mu) if config.trainable_mu else None
        self.trainable_params.append(self.Sigma) if config.trainable_sigma else None
        
        self.kalman_filter = tfp.distributions.LinearGaussianStateSpaceModel(num_timesteps = config.ph_steps,
                                                                             transition_matrix = self.A, 
                                                                             transition_noise = self.transition_noise, 
                                                                             observation_matrix = self.C,
                                                                             observation_noise = self.observation_noise, 
                                                                             initial_state_prior = self.initial_state_prior, 
                                                                             initial_step=0,
                                                                             validate_args=False, 
                                                                             allow_nan_stats=True,
                                                                             name='LinearGaussianStateSpaceModel')
        
    def call(self, inputs):
        x = inputs[0]
        mask = inputs[1]
        
        mu_smooth, Sigma_smooth = self.kalman_filter.posterior_marginals(x, mask = mask)
        p_zt_xT = tfp.distributions.MultivariateNormalTriL(mu_smooth, get_cholesky(Sigma_smooth, inspect.stack()[0][3]))
        return p_zt_xT
    
    def get_loss(self, x, p_zt_xT):
        """
        Get log probability densitity functions for the kalman filter
        ```
        z_t ~ N(μ_{t|T}, ∑_{t|T}) for t = 1,...,T
        log p(z_t|z_{t-1}) = log N(z_t | C z_{t-1}, R) = log N(z_t - Cz_{t-1} | 0, R) for t = 2,...,T 
        log p(x_t|z_t) = log N (x_t | Az_t, Q) = log N(x_t - Az_t | 0, Q) for t = 1,...,T
        log p(z_1) = log N(z_1 | μ_0, ∑_0)
        log p(z_t|x_{1:T}) = log N(z_t | μ_{t|T}, ∑_{t|T}) for t = 1,...,T
        ```
        
        Args:
            x: smooth sample
            mu_smooth: smooth mean
            Sigma_smooth: smooth covariance
            
        Returns:
            log_prob_z_z : log p(z_t | z_{t-1}) for t = 1,..., T
            log_prob_x_z : log p(x_t | z_t) for t = 2,..., T
            log_prob_0 : log p(z_1)
            log_prob_z_x : log p(z_t | x_{1:T}) for t = 1,..., T
        """
        # Sample from smoothing distribution
        A = self.kalman_filter.transition_matrix
        C = self.kalman_filter.observation_matrix
        transition_noise = self.kalman_filter.transition_noise
        observation_noise = self.kalman_filter.observation_noise
        
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
        log_prob_0 = self.kalman_filter.initial_state_prior.log_prob(z_0)
        
        ## log p(z_t | z_{t-1}) for t = 2,...,T
        # log p(z_t | z_{t-1}) = log N(z_t | Az_{t-1}, Q) = log N(z_t - Az_{t-1}| 0, Q) = log N(z_Az|0, Q)
        Az_t = tf.matmul(A, tf.expand_dims(z_tilde[:,:-1, :], axis=3))[...,0] # Az_1, ..., Az_{T-1}
        z_t = z_tilde[:, 1:, :] # z_2, ..., z_T
        z_Az = z_t - Az_t
        log_prob_z_z = transition_noise.log_prob(z_Az)
        
        return log_prob_z_z, log_prob_x_z, log_prob_0, log_prob_z_x

    def get_smooth_dist(self, x, mask):
        mu_smooth, Sigma_smooth = self.kalman_filter.posterior_marginals(x, mask = mask)
        x_mu_smooth, x_cov_smooth = self.kalman_filter.latents_to_observations(mu_smooth, Sigma_smooth)
        smooth_dist = tfp.distributions.MultivariateNormalTriL(loc=x_mu_smooth, 
                                                               scale_tril=get_cholesky(x_cov_smooth, inspect.stack()[0][3]))
        return smooth_dist

    def get_filter_dist(self, x, mask, get_pred=False):
        kalman_data = self.kalman_filter.forward_filter(x, mask=mask)
        _, mu_filt, Sigma_filt, mu_pred, Sigma_pred, x_mu_filt, x_covs_filt = kalman_data
        filt_dist = tfp.distributions.MultivariateNormalTriL(loc=x_mu_filt,
                                                             scale_tril=get_cholesky(x_covs_filt, inspect.stack()[0][3]))

        if get_pred:
            x_mu_filt_pred, x_covs_filt_pred = self.kalman_filter.latents_to_observations(mu_pred, Sigma_pred)
            filt_pred_dist = tfp.distributions.MultivariateNormalTriL(loc=x_mu_filt_pred, 
                                                                      scale_tril=get_cholesky(x_covs_filt_pred, inspect.stack()[0][3]))
            return filt_dist, filt_pred_dist

        return filt_dist
    
    def get_params(self):
        A = self.kalman_filter.transition_matrix
        A_eig = tf.linalg.eig(A)[0]
        Q = self.kalman_filter.transition_noise.stddev()
        C = self.kalman_filter.observation_matrix
        R = self.kalman_filter.observation_noise.stddev()
        mu_0 = self.kalman_filter.initial_state_prior.mean()
        sigma_0 = self.kalman_filter.initial_state_prior.covariance()
        return A, A_eig, Q, C, R, mu_0, sigma_0
    
    def sample(self, mu_smooth, Sigma_smooth, init_fixed_steps, n_steps, deterministic=True):
        A = self.kalman_filter.get_transition_matrix_for_timestep
        R = self.kalman_filter.get_transition_noise_for_timestep
        C = self.kalman_filter.get_observation_matrix_for_timestep
        Q = self.kalman_filter.get_observation_noise_for_timestep
        
        bs = tf.shape(mu_smooth)[0]
        z = mu_smooth[:,0]
        z = tf.expand_dims(z,2)
       
        x_samples = list()
        z_samples = list()
        for n in range(n_steps):
            # Output for the current time step
            if not deterministic:
                x = C(n).matmul(z) + Q(n).sample()[...,None]
            else:
                x = C(n).matmul(z) + Q(n).mean()[...,None]
            x = tf.squeeze(x,2)
            
            z_samples.append(tf.squeeze(z,2))
            x_samples.append(x)
            
            if (n+1) >= init_fixed_steps:
                if not deterministic:
                    z = A(n).matmul(z) + R(n).sample()[...,None]
                else:
                    z = A(n).matmul(z) + R(n).mean()[...,None]
            else:
                z = mu_smooth[:, n+1]
                z = tf.expand_dims(z,2)
         
        x_samples = tf.stack(x_samples, 1)
        z_samples = tf.stack(z_samples, 1)
        return x_samples, z_samples
