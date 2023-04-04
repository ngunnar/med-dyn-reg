import tensorflow as tf

#@tf.function
def get_log_dist(dist, y, mask_ones):
    log_dist = dist.log_prob(y)    
    log_dist = tf.multiply(log_dist, mask_ones)
    log_dist = tf.reduce_sum(log_dist, axis=1)
    return log_dist

#@tf.function
def ssim_calculation(y, y_pred):
    pred_imgs = tf.reshape(y_pred, (-1, y_pred.shape[2], y_pred.shape[3], 1))
    true_imgs = tf.reshape(y, (-1, y.shape[2], y.shape[3], 1))
    ssim = tf.image.ssim(pred_imgs, true_imgs, max_val=tf.reduce_max(true_imgs), filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    ssim = tf.reshape(ssim, (-1, y.shape[1]))
    return ssim

def set_name(name, prefix=None):
    return '_'.join(filter(None, (name, prefix)))
