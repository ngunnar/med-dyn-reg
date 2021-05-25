import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

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