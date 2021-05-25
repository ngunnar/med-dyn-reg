import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


# tfp.distributions.LinearGaussianStateSpaceModel.posterior_marginals can return non positive covariance matrix (round error?) 
def get_cholesky(A):
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
        print("test")
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
    I = np.eye(A.shape[0])
    k = 1
    while True:
        print(k)
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

class KalmanFilter(tf.keras.layers.Layer):
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
            initial_state_prior = tfp.distributions.MultivariateNormalDiag(loc = self.mu, 
                                                                       scale_diag = self.Sigma) 
        else:
            initial_state_prior = tfp.distributions.MultivariateNormalTriL(loc = self.mu,
                                                                           scale_tril = self.Sigma)
        
        # w ~ N(0,Q)
        if True:
            init_Q_np = np.ones(config.dim_z).astype('float32') * config.noise_transition
        else:    
            init_Q_np = np.eye(config.dim_z).astype('float32') * config.noise_transition
            
        init_Q = tf.constant_initializer(init_Q_np)
        self.Q = tf.Variable(initial_value=init_Q(shape=init_Q_np.shape), trainable=config.trainable_Q, dtype='float32', name="Q")
        if True:
            transition_noise = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(config.dim_z, dtype='float32'), 
                                                                        scale_diag=self.Q) #FULLY_REPARAMETERIZED by default
        else:
            transition_noise = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(config.dim_z, dtype='float32'), 
                                                                                  scale_tril=self.Q) #FULLY_REPARAMETERIZED by default
        # v ~ N(0,R)
        if True:
            #init_R_np = np.random.rand(config.dim_x).astype(np.float32)
            init_R_np = np.ones(config.dim_x).astype('float32') * config.noise_emission
        else:    
            init_R_np = np.eye(config.dim_x).astype('float32') * config.noise_emission
        init_R = tf.constant_initializer(init_R_np)
        self.R = tf.Variable(initial_value=init_R(shape=init_R_np.shape), trainable=config.trainable_R, dtype='float32', name="R" )
        #FULLY_REPARAMETERIZED by default
        if True:
            observation_noise = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(config.dim_x, dtype='float32'),
                                                                         scale_diag=self.R) 
        else:
            observation_noise = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(config.dim_x, dtype='float32'),
                                                                                   scale_tril=self.R)
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
                                                                             transition_noise = transition_noise, 
                                                                             observation_matrix = self.C,
                                                                             observation_noise = observation_noise, 
                                                                             initial_state_prior = initial_state_prior, 
                                                                             initial_step=0,
                                                                             validate_args=False, 
                                                                             allow_nan_stats=True,
                                                                             name='LinearGaussianStateSpaceModel')
        
    def call(self, inputs):
        x = inputs[0]
        mask = inputs[1]
        
        mu_smooth, Sigma_smooth = self.kalman_filter.posterior_marginals(x, mask = mask)
        p_zt_xT = tfp.distributions.MultivariateNormalTriL(mu_smooth, get_cholesky(Sigma_smooth))
        return p_zt_xT
    
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
