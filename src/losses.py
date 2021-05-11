import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from .utils import get_cholesky

def log_normal_pdf(sample, mean, logvar, mask, num_el, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    log_prob = tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
    log_prob = tf.multiply(mask, log_prob)
    log_prob = tf.truediv(tf.reduce_sum(log_prob), num_el)
    return log_prob

def elbo_kl(x, x_mu, x_logvar, mask, num_el):
    log_prob_x = log_normal_pdf(x, 0., 0., mask, num_el)
    log_prob_x_y = log_normal_pdf(x, x_mu, x_logvar, mask, num_el)
    return log_prob_x - log_prob_x_y

def elbo_reconstruction(config, y_true, y_hat, y_mean, y_logvar, mask, num_el):
    log_prob_y_x = tf.reduce_sum(-.5 * ((y_true - y_mean) ** 2. * tf.exp(-y_logvar) + y_logvar), axis=1)        
    log_prob_y_x = tf.multiply(mask, log_prob_y_x)
    log_prob_y_x = tf.truediv(tf.reduce_sum(log_prob_y_x), num_el)
    return log_prob_y_x

def log_p_kalman(x, mu_smooth, Sigma_smooth, kalman_filter):
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
         kalman_filter: kalman filter
         
     Returns:
         log_prob_z_z : log p(z_t | z_{t-1}) for t = 1,..., T
         log_prob_x_z : log p(x_t | z_t) for t = 2,..., T
         log_prob_0 : log p(z_1)
         log_prob_z_x : log p(z_t | x_{1:T}) for t = 1,..., T
    """
    # Sample from smoothing distribution
    A = kalman_filter.transition_matrix
    #A = kalman_filter.get_transition_matrix_for_timestep
    C = kalman_filter.observation_matrix
    #C = kalman_filter.get_observation_matrix_for_timestep
    transition_noise = kalman_filter.transition_noise
    observation_noise = kalman_filter.observation_noise

    mvn_smooth = tfp.distributions.MultivariateNormalTriL(mu_smooth, get_cholesky(Sigma_smooth))
    #mvn_smooth = tfp.distributions.MultivariateNormalFullCovariance(mu_smooth, Sigma_smooth)
    
    #z_tilde = latent_posterior_sample
    z_tilde = mvn_smooth.sample()
    
    ## log p(z_t | x_T) for t=1,...,T
    log_prob_z_x = mvn_smooth.log_prob(z_tilde)
    
    ## log p(x_t | z_t) for all t = 1,...,T
    # log N(x_t | Cz_t, R) -> log N(x_t - Cz_t|0, R) = log N(x_Cz_t | 0, R)
    Cz_t = tf.matmul(C, tf.expand_dims(z_tilde, axis=3))[...,0]
    x_Cz_t = x - Cz_t
    log_prob_x_z = observation_noise.log_prob(x_Cz_t)
    
    ## log p(z_1) = log p(z_1 | z_0)
    z_0 = z_tilde[:, 0, :]
    log_prob_0 = kalman_filter.initial_state_prior.log_prob(z_0)
    
    ## log p(z_t | z_{t-1}) for t = 2,...,T
    # log p(z_t | z_{t-1}) = log N(z_t | Az_{t-1}, Q) = log N(z_t - Az_{t-1}| 0, Q) = log N(z_Az|0, Q)
    Az_t = tf.matmul(A, tf.expand_dims(z_tilde[:,:-1, :], axis=3))[...,0] # Az_1, ..., Az_{T-1}
    z_t = z_tilde[:, 1:, :] # z_2, ..., z_T
    z_Az = z_t - Az_t
    log_prob_z_z = transition_noise.log_prob(z_Az)
    
    return log_prob_z_z, log_prob_x_z, log_prob_0, log_prob_z_x

def elbo_kalman(config, x, mask_flat, num_el, mu_smooth, Sigma_smooth, kalman_filter):
    # Sample from smoothing distribution
    A = kalman_filter.transition_matrix
    #A = kalman_filter.get_transition_matrix_for_timestep
    C = kalman_filter.observation_matrix
    #C = kalman_filter.get_observation_matrix_for_timestep
    transition_noise = kalman_filter.transition_noise
    observation_noise = kalman_filter.observation_noise

    mvn_smooth = tfp.distributions.MultivariateNormalTriL(mu_smooth, get_cholesky(Sigma_smooth))
    z_tilde = mvn_smooth.sample()
    #z_tilde = latent_posterior_sample
    ## log p(x_t | z_t) for all t = 1,...,T
    # log N(x_t | Cz_t, R) -> log N(x_t - Cz_t|0, R) = log N(x_Cz_t | 0, R)
    Cz_t = tf.reshape(tf.matmul(C, tf.expand_dims(z_tilde, axis=3)), [-1, config.dim_x]) #
    #Cz_t = tf.stack([C(t).matmul(z_tilde[:,t,:][...,None]) for t in range(z_tilde.shape[1])], axis=1)
    #Cz_t = tf.reshape(Cz_t, [-1, config.dim_x])
    x_Cz_t = tf.reshape(x, [-1, config.dim_x]) - Cz_t
    log_prob_x_z = observation_noise.log_prob(x_Cz_t)
    log_prob_x_z = tf.multiply(mask_flat, log_prob_x_z)
    
    ## log p(z_1) = log p(z_1 | z_0)
    z_0 = z_tilde[:, 0, :]
    log_prob_0 = kalman_filter.initial_state_prior.log_prob(z_0)
    
    ## log p(z_t | z_{t-1}) for t = 2,...,T
    # log p(z_t | z_{t-1}) = log N(z_t | Az_{t-1}, Q) = log N(z_t - Az_{t-1}| 0, Q) = log N(z_Az|0, Q)
    Az_t = tf.reshape(tf.matmul(A, tf.expand_dims(z_tilde[:,:-1, :], axis=3)), [-1, config.dim_z]) # Az_1, ..., Az_{T-1}
    z_t = tf.reshape(z_tilde[:, 1:, :], [-1, config.dim_z]) # z_2, ..., z_T
    z_Az = z_t - Az_t
    log_prob_z_z = transition_noise.log_prob(z_Az)
    
    ## - log p(z_t | x_T) p(z_t) for t=1,...,T
    log_prob_z_x = - mvn_smooth.log_prob(z_tilde)
    
    log_probs = [tf.truediv(tf.reduce_sum(log_prob_z_z), num_el),
                 tf.truediv(tf.reduce_sum(log_prob_x_z), num_el),
                 tf.truediv(tf.reduce_sum(log_prob_0), num_el),
                 tf.truediv(tf.reduce_sum(log_prob_z_x), num_el)]
    kf_elbo = tf.reduce_sum(log_probs)
    
    return kf_elbo, log_probs, z_tilde 

def loss_function_vae(config, y_true, mask, y_hat, y_mu, y_logvar, x_vae, x_mu, x_logvar):
    flat_mask = tf.cast(tf.reshape(mask == False, (-1, )), dtype='float32')
    num_el = tf.reduce_sum(flat_mask) # ()
    
    ## Reconstruction loss ##
    elbo_recon = elbo_reconstruction(config, tf.reshape(y_true, (-1, np.prod(config.dim_y))), y_hat, y_mu, y_logvar, flat_mask, num_el)
    
    ## D KL loss ##
    kl = elbo_kl(tf.reshape(x_vae, (-1, config.dim_x)), x_mu, x_logvar, flat_mask, num_el)
    loss_sum = - elbo_recon - kl
    return loss_sum, -elbo_recon, -kl


def loss_function(config, y_true, mask, y_hat, y_mu, y_logvar,
                  x_vae, x_mu, x_logvar,
                  x_seq, mu_smooth, Sigma_smooth, 
                  kalman_filter):
    flat_mask = tf.cast(tf.reshape(mask == False, (-1, )), dtype='float32')
    num_el = tf.reduce_sum(flat_mask) # ()
    
    ## Reconstruction loss ##
    elbo_recon = elbo_reconstruction(config, tf.reshape(y_true, (-1, np.prod(config.dim_y))), y_hat, y_mu, y_logvar, flat_mask, num_el)
    
    ## D KL loss ##
    kl = elbo_kl(tf.reshape(x_vae, (-1, config.dim_x)), x_mu, x_logvar, flat_mask, num_el)
    
    ## Kalman loss ##
    kf_elbo, _, _ = elbo_kalman(config, x_seq, flat_mask, num_el, mu_smooth, Sigma_smooth, kalman_filter)
    loss_sum = -elbo_recon - kl - kf_elbo
    return loss_sum, -elbo_recon, -kl, -kf_elbo

def _diffs(y):
    vol_shape = y.get_shape().as_list()[1:-1]
    ndims = len(vol_shape)

    df = [None] * ndims
    for i in range(ndims):
        d = i + 1
        # permute dimensions to put the ith dimension first
        r = [d, *range(d), *range(d + 1, ndims + 2)]
        y = tf.keras.backend.permute_dimensions(y, r)
        dfi = y[1:, ...] - y[:-1, ...]
        
        # permute back
        # note: this might not be necessary for this loss specifically,
        # since the results are just summed over anyway.
        r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
        df[i] = tf.keras.backend.permute_dimensions(dfi, r)
    
    return df

def grad_loss(penalty, y_pred):
    if penalty == 'l1':
        df = [tf.reduce_mean(tf.abs(f)) for f in _diffs(y_pred)]
    else:
        assert penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % penalty
        df = [tf.reduce_mean(f * f) for f in _diffs(y_pred)]
    return tf.add_n(df) / len(df)