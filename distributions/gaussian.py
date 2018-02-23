from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from helpers.tf_utils import logdet


def standard_to_natural(mu, sigma, name='gauss_to_nat'):
    with tf.name_scope(name):
        # if len(mu.get_shape()) != 2 or len(sigma.get_shape()) != 3:
        #     raise NotImplementedError("standard_to_natural is not implemented for this case.")
        eta_2 = -0.5 * tf.matrix_inverse(sigma)  # shape = (nb_components, latent_dim, latent_dim)
        eta_1 = -2 * tf.matmul(eta_2, tf.expand_dims(mu, axis=-1))  # shape = (nb_components, latent_dim)
        eta_1 = tf.reshape(eta_1, mu.get_shape())

        return tf.tuple((eta_1, eta_2), name='nat_params')


def natural_to_standard(eta1, eta2, name='gauss_to_stndrd'):
    with tf.name_scope(name):
        sigma = tf.matrix_inverse(-2 * eta2)
        mu = tf.matmul(sigma, tf.expand_dims(eta1, axis=2))
        mu = tf.reshape(mu, eta1.get_shape())
        return tf.tuple((mu, sigma), name='stndrd_params')


def log_probability_nat(x, eta1, eta2, weights=None):
    """
    Computes log N(x|eta1, eta2)
    Args:
        x:
        eta1:
        eta2:
        weights:

    Returns:
        Normalized log-probability
    """

    # log guassian (log pdf)
    N, D = x.get_shape().as_list()

    # distinguish between 1 and k>1 components...
    if len(eta1.get_shape()) != 3:
        shape = eta1.get_shape()
        raise AssertionError("eta1 must be of shape (N,K,D). Its shape is %s." % str(shape))

    with tf.name_scope('gaussian_logprob'):
        logprob = tf.einsum('nd,nkd->nk', x, eta1)
        logprob += tf.einsum('nkd,nd->nk', tf.einsum('nd,nkde->nke', x, eta2), x)
        logprob -= D/2. * tf.log(2. * np.pi)

        # add dimension for further computations
        eta1 = tf.expand_dims(eta1, axis=3)
        logprob += 1./4 * tf.einsum('nkdi,nkdi->nk', tf.matrix_solve(eta2, eta1, name='mu'), eta1)

        # logprob += 0.5 * tf.log(tf.matrix_determinant(-2. * eta2 + 1e-20 * tf.eye(D)))  # todo this is nummerically NOT stable!!
        logprob += 0.5 * logdet(-2. * eta2 + 1e-20 * tf.eye(D))

        if weights is not None:
            logprob += tf.expand_dims(tf.log(weights), axis=0, name='component_weighting')

        # log sum exp trick
        with tf.name_scope('log_sum_exp'):
            max_logprob = tf.reduce_max(logprob, axis=1, keep_dims=True)
            normalizer = tf.add(max_logprob, tf.log(tf.reduce_sum(tf.exp(logprob - max_logprob), axis=1, keep_dims=True)))

        return tf.subtract(logprob, normalizer, name='normalized_logprob')


def log_probability_nat_per_samp(x_samps, eta1, eta2):
    """
        Args:
            x_samps: matrix of shape (minibatch_size, nb_components, nb_samps, latent_dims)
            eta1: 1st natural parameter for Gaussian distr; shape: (size_minibatch, nb_components, latent_dim)
            eta2: 2nd natural parameter for Gaussian distr; shape: (size_minibatch, nb_components, latent_dim, latent_dim)

        Returns:
            1/S sum^S_{s=1} log N(x^(s)|eta1, eta2) of shape (N, K, S)
        """
    # same as above, but x consists of S samples for K components: x.shape = (N, K, S, D)
    # todo: merge with above function (above is the same but normalised)
    N, K, S, D = x_samps.get_shape().as_list()
    assert eta1.get_shape() == (N, K, D)
    assert eta2.get_shape() == (N, K, D, D)

    with tf.name_scope('log_prob_4d'):

        # -1\2 (sigma^(-1) * x) * x + sigma^(-1)*mu*x
        log_normal = tf.einsum('nksd,nksd->nks',
                               tf.einsum('nkij,nksj->nksi', eta2, x_samps),
                               x_samps)
        log_normal += tf.einsum('nki,nksi->nks', eta1, x_samps)

        # 1/4 (-2 * sigma * (sigma^(-1) * mu)) sigma^(-1) * mu = -1/2 mu sigma^(-1) mu; shape = N, K, 1
        log_normal += 1.0 / 4 * tf.einsum('nkdi,nkd->nki', tf.matrix_solve(eta2, tf.expand_dims(eta1, axis=-1)), eta1)
        log_normal -= D/2. * tf.constant(np.log(2 * np.pi), dtype=tf.float32, name='log2pi')

        # + 1/2 log |sigma^(-1)|
        log_normal += 1.0 / 2 * tf.expand_dims(logdet(-2.0 * eta2 + 1e-20 * tf.eye(D)), axis=2)

        return log_normal

