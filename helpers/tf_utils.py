from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def variable_on_device(name, shape, initializer, trainable=True, dtype=tf.float32, device='/gpu:0'):
    """Helper to create a Variable stored on specified device.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
      trainable: indicates whether variable is trainable
      dtype: variable's type
      device: indicates where variable should be stored
    Returns:
      Variable Tensor
    """
    with tf.device(device):
        var = tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable, dtype=dtype)
    return var


def logdet(A, name='logdet'):
    """
    Numerically stable implementation of log(det(A)) for symmetric positive definite matrices
    Source: https://github.com/tensorflow/tensorflow/issues/367#issuecomment-176857495
    Args:
        A: positive definite matrix of shape ..., D, D
        name: tf name scope

    Returns:
        log(det(A))
    """
    with tf.name_scope(name):
        # return tf.log(tf.matrix_determinant(A))
        return tf.multiply(
            2.,
            tf.reduce_sum(
                tf.log(
                    tf.matrix_diag_part(
                        tf.cholesky(A)
                    )
                ),
                axis=-1
            ),
            name='logdet'
        )


def average_gradients(tower_grads):
    """
    Calculate average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been
       averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks as follows:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
