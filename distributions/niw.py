from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def expected_values(niw_standard_params):
    beta, m, C, v = niw_standard_params
    with tf.name_scope('niw_expectation'):
        exp_m = tf.identity(m, 'expected_mean')
        C_inv = tf.matrix_inverse(C)
        C_inv_sym = tf.divide(tf.add(C_inv, tf.matrix_transpose(C_inv)), 2., name='C_inv_symmetrised')
        exp_C = tf.matrix_inverse(
            tf.multiply(C_inv_sym, tf.expand_dims(tf.expand_dims(v, 1), 2), name='expected_precision'),
            name='expected_covariance')
        return exp_m, exp_C


def standard_to_natural(beta, m, C, v):
    with tf.name_scope('niw_to_nat'):
        K, D = m.get_shape()
        assert beta.get_shape() == (K,)
        D = int(D)

        b = tf.expand_dims(beta, -1) * m
        A = C + _outer(b, m)
        v_hat = v + D + 2

        return A, b, beta, v_hat


def natural_to_standard(A, b, beta, v_hat):
    with tf.name_scope('niw_to_stndrd'):
        m = tf.divide(b, tf.expand_dims(beta, -1))

        K, D = m.get_shape()
        assert beta.get_shape() == (K,)
        D = int(D)

        C = A - _outer(b, m)
        v = v_hat - D - 2
        return beta, m, C, v


def _outer(a, b):
    a_ = tf.expand_dims(a, axis=-1)
    b_ = tf.expand_dims(b, axis=-2)
    return tf.multiply(a_, b_, name='outer')
