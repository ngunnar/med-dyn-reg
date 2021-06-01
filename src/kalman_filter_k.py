import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from .kalman_filter import KalmanFilter

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

class AlphaNetwork(tf.keras.layers.Layer):
    def __init__(self, dim_RNN_alpha, k, name='alpha_rnn', **kwargs):
        super(AlphaNetwork, self).__init__(name=name, **kwargs)
        self.lstm = tf.keras.layers.LSTM(dim_RNN_alpha, return_sequences=True, name='LSTM')
        self.linear = tf.keras.layers.Dense(k, name='Linear')
        self.softmax = tf.keras.layers.Softmax(name='Softmax')
    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.linear(x)
        return self.softmax(x)


class KalmanFilterK(KalmanFilter):
    def __init__(self, config, name='kalman_filter_k', **kwargs):
        super(KalmanFilterK, self).__init__(name=name, config=config, **kwargs)
        self.dim_z = config.dim_z
        self.dim_x = config.dim_x

        k = 3
        self.alpha_network = AlphaNetwork(50, k)
        self.alpha = tf.keras.Input(name='alpha', shape=(config.ph_steps, k, ), dtype=tf.dtypes.float32)#tf.Variable(dtype='float32', name='alpha')
        ## Parameters        
        A_init = tf.random_normal_initializer()
        self.A = tf.Variable(initial_value=A_init(shape=(k, config.dim_z,config.dim_z)), 
                             trainable=config.trainable_A, 
                             dtype="float32", 
                             name="A")
        
        C_init = tf.random_normal_initializer()
        self.C = tf.Variable(initial_value=C_init(shape=(k, config.dim_x, config.dim_z)), 
                             trainable=config.trainable_C, 
                             dtype="float32", 
                             name="C")
        
        self.kalman_filter = tfp.distributions.LinearGaussianStateSpaceModel(num_timesteps = config.ph_steps,
                                                                             transition_matrix = self.transition_matrix, 
                                                                             transition_noise = self.transition_noise, 
                                                                             observation_matrix = self.observation_matrix,
                                                                             observation_noise = self.observation_noise, 
                                                                             initial_state_prior = self.initial_state_prior, 
                                                                             initial_step=0,
                                                                             validate_args=False, 
                                                                             allow_nan_stats=True,
                                                                             name='LinearGaussianStateSpaceModel')
    
    def observation_matrix(self, t):
        C = tf.reshape(tf.matmul(tf.gather(self.alpha, t, axis=1), tf.reshape(self.C, (-1, self.dim_x*self.dim_z))), (-1, self.dim_x, self.dim_z)) # Or alpha GLOBAL
        return tf.linalg.LinearOperatorFullMatrix(C)

    def transition_matrix(self, t):
        A =  tf.reshape(tf.matmul(tf.gather(self.alpha, t, axis=1), tf.reshape(self.A, (-1, self.dim_z*self.dim_z))), (-1, self.dim_z, self.dim_z)) # Or alpha GLOBAL
        return tf.linalg.LinearOperatorFullMatrix(A)

    def call(self, inputs):
        x = inputs[0]
        mask = inputs[1]
        self.alpha = self.alpha_network(x) # TODO tf.multiply((1-mask), y)) + tf.multiply(mask, y_pred)
        return super(KalmanFilterK, self).call([x, mask])
    
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
        A = self.transition_matrix
        C = self.observation_matrix
        transition_noise = self.kalman_filter.transition_noise
        observation_noise = self.kalman_filter.observation_noise
        
        #z_tilde = latent_posterior_sample
        z_tilde = p_zt_xT.sample()
        
        ## log p(z_t | x_T) for t=1,...,T
        log_prob_z_x = p_zt_xT.log_prob(z_tilde)
        
        ## log p(x_t | z_t) for all t = 1,...,T
        # log N(x_t | Cz_t, R) -> log N(x_t - Cz_t|0, R) = log N(x_Cz_t | 0, R)
        #Cz_t = tf.matmul(C, tf.expand_dims(z_tilde, axis=3))[...,0]
        Cz_t = tf.reshape(tf.matmul(C(np.arange(z_tilde.shape[1])), tf.reshape(z_tilde,(-1,z_tilde.shape[-1], 1))), (z_tilde.shape[0], -1, x.shape[-1]))
        x_Cz_t = x - Cz_t
        log_prob_x_z = observation_noise.log_prob(x_Cz_t)
        
        ## log p(z_1) = log p(z_1 | z_0)
        z_0 = z_tilde[:, 0, :]
        log_prob_0 = self.kalman_filter.initial_state_prior.log_prob(z_0)
        
        ## log p(z_t | z_{t-1}) for t = 2,...,T
        # log p(z_t | z_{t-1}) = log N(z_t | Az_{t-1}, Q) = log N(z_t - Az_{t-1}| 0, Q) = log N(z_Az|0, Q)
        #Az_t = tf.matmul(A, tf.expand_dims(z_tilde[:,:-1, :], axis=3))[...,0] # Az_1, ..., Az_{T-1}
        Az_t = tf.reshape(tf.matmul(A(np.arange(z_tilde.shape[1]-1)), tf.reshape(z_tilde[:,:-1, :], (-1, z_tilde.shape[-1], 1))),(z_tilde.shape[0],-1, z_tilde.shape[-1])) #[...,0]
        z_t = z_tilde[:, 1:, :] # z_2, ..., z_T
        z_Az = z_t - Az_t
        log_prob_z_z = transition_noise.log_prob(z_Az)
        
        return log_prob_z_z, log_prob_x_z, log_prob_0, log_prob_z_x
    
    def get_smooth_dist(self, x, mask):
        self.alpha = self.alpha_network(x) # TODO tf.multiply((1-mask), y)) + tf.multiply(mask, y_pred)
        return super(KalmanFilterK, self).get_smooth_dist(x, mask)

    def get_filter_dist(self, x, mask, get_pred=False):
        self.alpha = self.alpha_network(x) # TODO tf.multiply((1-mask), y)) + tf.multiply(mask, y_pred)
        return super(KalmanFilterK, self).get_filter_dist(x, mask, get_pred)

    def get_params(self):
        A = self.A
        A_eig = tf.linalg.eig(A)[0]
        Q = self.kalman_filter.transition_noise.stddev()
        C = self.C
        R = self.kalman_filter.observation_noise.stddev()
        mu_0 = self.kalman_filter.initial_state_prior.mean()
        sigma_0 = self.kalman_filter.initial_state_prior.covariance()
        return A, A_eig, Q, C, R, mu_0, sigma_0