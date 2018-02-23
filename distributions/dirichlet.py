from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def expected_log_pi(dir_standard_param):
    with tf.name_scope('dirichlet_expectation'):
        return tf.subtract(tf.digamma(dir_standard_param),
                           tf.digamma(tf.reduce_sum(dir_standard_param, axis=-1, keep_dims=True)),
                           name='expected_mixing_coeffs')


def standard_to_natural(alpha):
    with tf.name_scope('dir_to_nat'):
        return alpha - 1


def natural_to_standard(alpha_nat):
    with tf.name_scope('dir_to_stndrd'):
        return alpha_nat + 1