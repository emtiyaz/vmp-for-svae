import tensorflow as tf
import numpy as np

from helpers.tf_utils import logdet


def _logprob_full_scale(y, mu, sigma, v, name='student_t_logprob'):
    """
    Computes Student-t log-probability with full scale matrix sigma
    Args:
        y: data tensor; shape = N, K, S, D
        mu: means; shape = K, D
        sigma: scale matrices; shape = K, D, D
        v: degree of freedom; shape = K
        name: operation name

    Returns:
        log S(y|mu, sigma, v)
    """
    with tf.name_scope(name):

        N, K, S, D = y.get_shape().as_list()
        assert mu.get_shape() == (K, D)
        assert sigma.get_shape() == (K, D, D)
        assert v.get_shape() == K
        mu = tf.tile(tf.expand_dims(tf.expand_dims(mu, axis=0), axis=2), (N, 1, S, 1))
        sigma = tf.tile(tf.expand_dims(tf.expand_dims(sigma, axis=0), axis=2), (N, 1, S, 1, 1))
        v = tf.expand_dims(tf.expand_dims(v, axis=0), axis=2)


        err = tf.expand_dims(y - mu, axis=-1)  # shape = N, K, S, D, 1
        mahalanobis_sqr_d = tf.einsum('nksdi,nksdi->nks', err, tf.matrix_solve(sigma, err))  # shape = N, K, S

        logprob = tf.lgamma(0.5 * (v + D)) - tf.lgamma(0.5 * v)            # shape = 1, K, 1
        logprob -= 0.5 * D * tf.log(tf.constant(np.pi, name='pi') * v)     # shape = 1, K, 1
        logprob -= 0.5 * logdet(sigma)                                     # shape = N, K, S
        logprob -= 0.5 * (v + D) * (tf.log1p(mahalanobis_sqr_d/v))         # important to use log1p for num. stability

        return tf.identity(logprob, name='logprob')


def logprob_smm_mixture(y, mu, sigma, v, log_pi, name='student_t_logprob'):
    with tf.name_scope(name):

        N, D = y.get_shape().as_list()
        K, D_ = mu.get_shape().as_list()
        assert D_ == D                                                   # shape = N, D
        y = tf.tile(tf.expand_dims(tf.expand_dims(y, axis=1), axis=2), (1, K, 1, 1))  # shape = N, K, 1, D

        logprob = _logprob_full_scale(y, mu, sigma, v)
        logprob = tf.reshape(logprob, (N, K))                                         # shape = N, K

        # multiply by mixture weights
        logprob += tf.expand_dims(log_pi, axis=0)

        return tf.identity(logprob, name='logprob')


def log_probability_per_samp(y, mu, sigma, v, name='student_t_logprob_per_samp'):
    with tf.name_scope(name):
        return _logprob_full_scale(y, mu, sigma, v, name='logprob_per_samp')
