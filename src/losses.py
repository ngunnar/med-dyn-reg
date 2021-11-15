import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import tensorflow.keras.backend as K

def _diffs(y):
    org_shape = y.get_shape()
    y = tf.reshape(y, (y.shape[0]*y.shape[1], y.shape[2], y.shape[3], y.shape[4]))
    vol_shape = y.get_shape().as_list()[1:-1]
    ndims = len(vol_shape)

    df = [None] * ndims
    for i in range(ndims):
        d = i + 1
        # permute dimensions to put the ith dimension first
        r = [d, *range(d), *range(d + 1, ndims + 2)]
        yp = K.permute_dimensions(y, r)
        dfi = yp[1:, ...] - yp[:-1, ...]

        # permute back
        # note: this might not be necessary for this loss specifically,
        # since the results are just summed over anyway.
        r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
        out = K.permute_dimensions(dfi, r)
        out = tf.reshape(out, (org_shape[0], org_shape[1], out.shape[1], out.shape[2], out.shape[3]))
        df[i] = out
        #print("H", df[i].shape)

    return df

def grad_loss(penalty, y_pred):
    """
    returns Tensor of size [bs]
    """
    #y_pred = tf.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2], y_pred.shape[3], 2))
    difs = _diffs(y_pred)
    
    if penalty == 'l1':
        dif = [tf.abs(f) for f in difs]
    else:
        assert penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
        dif = [f * f for f in difs]

    #print("D", [d.shape for d in dif])
    df = [tf.reduce_sum(K.batch_flatten(f), axis=-1) for f in dif]
    #print("Df", [d.shape for d in df])
    grad = tf.add_n(df) / len(df)
    return grad



import tensorflow.keras.backend as K

class Grad:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            yp = K.permute_dimensions(y, r)
            dfi = yp[1:, ...] - yp[:-1, ...]

            if self.vox_weight is not None:
                w = K.permute_dimensions(self.vox_weight, r)
                # TODO: Need to add square root, since for non-0/1 weights this is bad.
                dfi = w[1:, ...] * dfi

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):
        """
        returns Tensor of size [bs]
        """

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        print(len(df))
        grad = tf.add_n(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad