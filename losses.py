from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def weighted_mse(y_true, y_pred, r_nk_pred, name='mse'):
    """
    Args:
        y_true: True values; shape = N, D
        y_pred: S samples of predicted values for each component k; shape = N, K, S, D
        r_nk_pred: responsibility of component k for point n; shape = N, K

    Returns:
        SUM^K 1/N r_nk * SUM^N 1/S SUM^S 1/D * SUM^D (y_nd - y_hat_ndk)^2
        Is equal to sklearn.mean_square_error**0.5 for K=1 and S=1
    """
    with tf.name_scope(name):
        N, K, S, D = y_pred.get_shape().as_list()
        assert y_true.get_shape() == (N, D)
        assert r_nk_pred.get_shape() == (N, K)

        # mean square error (square euclidean distance averaged over samples); shape = N, K
        mse = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(tf.expand_dims(tf.expand_dims(y_true, 1), 2) - y_pred, name='square_err'),
                # shape = N, K, S, D
                axis=3, name='sample_mse'),  # sum over dimensions (= square euclidean dist); shape = N, K, S
            axis=2, name='cmpnt_dpnt_mse')  # mean over samples; shape = N, K

        # weight mse with predicted component responsibilities
        return tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(mse, r_nk_pred, name='weighted_cmpnt_dpnt_mse'),
                axis=1, name='dpnt_mse'),  # sum over (weighted) compontents; shape = N
            name='mse')  # mean over data points; shape = 1


def bernoulli_logprob(y_true_bin, logits, log_weights=None, missing_data_mask=None, name='bernoulli_logprob'):
    with tf.name_scope(name):
        if log_weights is None:
            N, S, D = logits.get_shape().as_list()
        else:
            N, K, S, D = logits.get_shape().as_list()
            assert log_weights.get_shape() == (N, K) or log_weights.get_shape() == (N, K, S)
            if log_weights.get_shape() == (N, K):
                log_weights = tf.expand_dims(log_weights, 2)
        assert y_true_bin.get_shape() == (N, D)

        # add dimensions for S (and K)
        y_true_bin = tf.expand_dims(y_true_bin, axis=1)
        if log_weights is not None:
            y_true_bin = tf.expand_dims(y_true_bin, axis=1)

        pixel_logprobs = -tf.log(1. + tf.exp(tf.multiply(-logits, y_true_bin)))

        # set log p(y_nd = ŷ_nd) = 0 if y_nd is observed.
        if missing_data_mask is not None:
            missing_data_mask = tf.expand_dims(tf.to_float(missing_data_mask), axis=1)
            if log_weights is not None:
                missing_data_mask = tf.expand_dims(missing_data_mask, axis=1)

            pixel_logprobs = tf.multiply(pixel_logprobs, missing_data_mask)

        # p(y_n) = prod^D p(y_nd)
        logprobs = tf.reduce_sum(pixel_logprobs, axis=-1, name='log_p_y_x_z')

        # integrate out z: log SUM^K p(y, z=k)
        if log_weights is not None:
            logprobs = tf.reduce_logsumexp(tf.add(logprobs, log_weights), axis=1, name='log_p_y_x')

        # integrate out x: log(1/S SUM^S p(y_n|x*, z))
        img_logprobs = tf.subtract(tf.reduce_logsumexp(logprobs, axis=-1),
                                   tf.constant(S, dtype=tf.float32, name='nb_samples'),
                                   name='log_p_y')

        # average over batch: return mean reconstruction loglike
        return tf.reduce_mean(img_logprobs, name='log_p_yn')


def diagonal_gaussian_logprob(y_true, mean, var, log_weights, mask=None, name='gauss_logprob'):
    """

    Args:
        y_true:
        mean: predicted reconstruction means; shape = N, K, S, D
        var: predicted reconstruction variances (diagonal); shape = N, K, S, D
        log_weights: predicted component responsibilities: shape = (N, K) or shape = (N, K, S)
        name:
    Returns:
        log p(y) = log 1/S SUM^S N(y|mu(x), var(x)); z ~ q(y|x, phi)
    """
    with tf.name_scope(name):
        N, K, S, D = mean.get_shape().as_list()
        assert var.get_shape() == mean.get_shape()
        assert y_true.get_shape() == (N, D)
        assert log_weights.get_shape() == (N, K) or log_weights.get_shape() == (N, K, S)
        if mask is not None:
            assert mask.get_shape() == (N, D)

        # add dimensions s.t. y_true.shape = N, 1, 1, D
        y_true = tf.expand_dims(tf.expand_dims(y_true, 1), 2)

        # compute log(p(y_n|x_n*, z_n))
        with tf.name_scope('p_y_given_x_z'):
            log_p_y_given_x_z = tf.divide(tf.square(y_true - mean), var)
            log_p_y_given_x_z += tf.log(var)
            log_p_y_given_x_z += tf.constant(np.log(2 * np.pi), dtype=tf.float32, name='log2pi')
            log_p_y_given_x_z *= -0.5

            # keep only imputed logliks: for 'observed values', set log p(y_nd = ŷ_nd) = 0
            if mask is not None:
                with tf.name_scope('mask_missing_data'):
                    log_p_y_given_x_z = tf.multiply(
                        tf.to_float(tf.expand_dims(tf.expand_dims(mask, 1), 2)),
                        log_p_y_given_x_z,
                        name='missing_data')

            # sum over dimensions and add log_weights; shape = N, K, S
            if len(log_weights.get_shape()) == 2:
                log_weights = tf.expand_dims(log_weights, 2)
            log_p_y_given_x_z = tf.add(tf.reduce_sum(log_p_y_given_x_z, axis=3), log_weights, name='log_p_y_given_x_z')

        # integrate out x: log(1/S SUM^S p(y|x*, z)) = log(SUM^S p(y|x*, z)) - log(S)
        with tf.name_scope('int_out_x'):
            max_y_given_x_z = tf.reduce_max(log_p_y_given_x_z, axis=2, keep_dims=True)
            with tf.name_scope('log_sum_exp'):
                log_p_y_given_z = tf.add(
                    tf.log(tf.reduce_sum(tf.exp(log_p_y_given_x_z - max_y_given_x_z), axis=2)),
                    tf.reshape(max_y_given_x_z, (N, K)))
            # divide by sample size (= subract log(S))
            log_p_y_given_z -= tf.log(tf.constant(S, dtype=tf.float32), name='log_p_y_given_z')  # shape = (N, K)

        # integrate out z (sum over weighted components):
        with tf.name_scope('int_out_z'):
            max_y_given_z = tf.reduce_max(log_p_y_given_z, axis=1, keep_dims=True)
            with tf.name_scope('log_sum_exp'):
                p_y = tf.add(
                    tf.log(tf.reduce_sum(tf.exp(log_p_y_given_z - max_y_given_z), axis=1)),
                    max_y_given_z)  # shape = (N,)

        # average over batch: return mean reconstruction loglike
        return tf.reduce_mean(p_y, name='log_p_yn')


def imputation_mse(y_true, y_pred, r_nk_pred, missing_data_mask, name='imp_mse'):
    with tf.name_scope(name):
        N, K, S, D = y_pred.get_shape().as_list()
        assert y_true.get_shape() == (N, D)
        assert r_nk_pred.get_shape() == (N, K)
        assert missing_data_mask.get_shape() == (N, D)

        # set observed values to 0 in ground truth and predictions
        y_true_missing = tf.multiply(y_true, tf.to_float(missing_data_mask))
        y_pred_missing = tf.multiply(
            tf.to_float(tf.expand_dims(tf.expand_dims(missing_data_mask, 1), 2)),  # shape N, 1, 1, D
            y_pred,  # shape N, K, S, D
            name='missing_data')

        square_err = tf.reduce_mean(
            tf.square(tf.expand_dims(tf.expand_dims(y_true_missing, 1), 2) - y_pred_missing, name='square_err'),  # shape = N, K, S, D
            axis=2, name='sample_avg_se')  # mean over samples; shape = N, K, D

        # use responsibilities to normalize over components
        w_square_err = tf.multiply(square_err, tf.expand_dims(r_nk_pred, 2), name='weighted_sample_avg_se')

        return tf.divide(tf.reduce_sum(w_square_err), tf.constant(N, dtype=tf.float32), name='imputation_mse')


def imputation_losses(y_true, missing_data_mask, imputation_method, nb_samples_pert=100, nb_samples_rec=100, seed=0,
                      decoder_type='standard', name='imputation_losses'):
    """
    Args:
        y_true: true data
        missing_data_mask: masks missing values
        imputation_method: function to impute missing values
        nb_samples_pert: number of perturbed samples should be created per data point
        nb_samples_rec: number of times a perturbed sample should be imputed
        seed: random seed
        decoder_type: 'bernoulli' or 'standard'
        name: tensorflow name scope

    Returns:
       imputation performance
    """
    with tf.name_scope(name):
        mse = tf.constant(0, dtype=tf.float32)

        # for MSE computation, we compare to binary data in {-1, 1}
        if decoder_type == 'bernoulli':
            y_true_01 = tf.where(tf.equal(y_true, -1),
                                 tf.zeros_like(y_true, dtype=tf.float32),
                                 tf.ones_like(y_true, dtype=tf.float32))
        else:
            y_true_01 = y_true

        # init lists for loglike computation
        collected_means = []
        collected_vars = []
        log_weights = []

        for s in range(nb_samples_pert):
            # sample missing data (fill missing values with random noise)
            y_perturbed = perturb_data(y_true, missing_data_mask, seed)

            # use trained variables
            tf.get_variable_scope().reuse_variables()
            y_k_rec_mean, y_k_rec_var, log_r_nk = imputation_method(y_perturbed)

            # set all values to zero except the ones we want to evaluate
            missing_data_pred = tf.multiply(
                tf.to_float(tf.expand_dims(tf.expand_dims(missing_data_mask, 1), 2)),
                y_k_rec_mean,
                name='missing_data')

            # compute MSE over missing values
            mse += imputation_mse(y_true_01, missing_data_pred, tf.exp(log_r_nk), missing_data_mask,
                                  name='mse_sample')

            # collect predicted means, vars and weights (shapes of means and vars: N, K, S, D; log_weights: N, K, S)
            collected_means.append(y_k_rec_mean)
            collected_vars.append(y_k_rec_var)
            log_weights.append(tf.tile(tf.expand_dims(log_r_nk, 2), (1, 1, nb_samples_rec)))

        # average sample-MSEs
        expected_mse = tf.divide(mse, tf.constant(nb_samples_pert, dtype=tf.float32, name='nb_samps'), name='expected_mse')

        # collect predicted means, vars/logits and responsibilities
        means = tf.concat(collected_means, axis=2, name='collect_means')  # shape = N, K, (nb_samps_pert*nb_samps_rec)
        vars_or_logits = tf.concat(collected_vars, axis=2, name='collect_vars_or_logits')  # same shape
        log_weights = tf.concat(log_weights, axis=2, name='collected_weights')  # same shape

        # compute loglikelihood
        if decoder_type == 'bernoulli':
            loglike = bernoulli_logprob(y_true, vars_or_logits, log_weights, missing_data_mask, name='gauss_logprob')
        else:
            loglike = diagonal_gaussian_logprob(y_true, means, vars_or_logits, log_weights, mask=missing_data_mask,
                                                name='gauss_logprob')

        return expected_mse, loglike


def generate_missing_data_mask(y, noise_ratio=0.3, mask_type='random', seed=0, name='make_mask'):
    # creates a random (but constant) mask indicating missing values
    with tf.name_scope(name):
        N, D = y.get_shape().as_list()
        mask_size = N * D
        mask = np.zeros(mask_size, dtype=np.bool)

        if mask_type == 'random':
            nb_noise_samps = int(mask_size * noise_ratio)

            # Set random elements to true
            npr = np.random.RandomState(seed)
            missing_entries = npr.choice(np.arange(mask_size), size=nb_noise_samps, replace=False)
            mask[missing_entries] = True

        else:
            # image is considered to be square
            side = np.sqrt(D)
            assert side.is_integer()
            side = int(side)
            half_side = side//2
            mask = mask.reshape(N, side, side)
            if mask_type == 'quarter':
                # remove lower left
                mask[:, half_side:side, :half_side] = True

            elif mask_type == 'lower_half':
                mask[:, half_side:side, :side] = True

            elif mask_type == 'left_half':
                mask[:, :side, :half_side] = True
            else:
                raise NotImplementedError("The mask type '%s' does not exist." % mask_type)

        return tf.constant(mask.reshape((N, D)), name='missing_data_mask')


def perturb_data(y, missing_data_mask, seed, decoder_type='standard', name='perturb_data'):
    """
    Replaces random elements in the input matrix by random noise.
    Args:
        y:
        mask: boolean array indicating which elements in y should be replaced by a random value.
        seed:
    Returns:
        Perturbed input tensor
    """
    with tf.name_scope(name):

        # replace some random elements by random noise.
        if decoder_type == 'standard':
            noise_missing = tf.random_normal(mean=0, stddev=1., seed=seed, shape=missing_data_mask.get_shape())
        elif decoder_type == 'bernoulli':
            probs = 0.5 * tf.ones_like(missing_data_mask, dtype=tf.float32)
            noise_missing = tf.distributions.Bernoulli(probs=probs).sample(seed=seed)
            noise_missing = tf.where(tf.equal(noise_missing, 0),
                                     -tf.ones_like(noise_missing, dtype=tf.float32),
                                     tf.ones_like(noise_missing, dtype=tf.float32))
        else:
            raise NotImplementedError
        noise_missing = tf.multiply(tf.to_float(missing_data_mask), noise_missing, name='missing_data')

        # remove missing data
        y_remaining = tf.multiply(tf.to_float(tf.logical_not(missing_data_mask)), y, name='remaining_data')

        return tf.add(y_remaining, noise_missing, 'perturbed_data')


def purity(r_nk, labels, eps=1e-10, name='purity'):
    """
    A measure of the extent to which a cluster contains objects of a single class.
    From https://www-users.cs.umn.edu/~kumar/dmbook/ch8.pdf
    Args:
        r_nk: component responsibilities per data point
        labels: ground truth; binary matrix of shape N, C
        name:

    Returns:
        entropy (0 if perfect clustering) and purity (1 if perfect clustering)
    """
    with tf.name_scope(name):
        N, K = r_nk.get_shape().as_list()
        N, C = labels.get_shape().as_list()

        eps = tf.constant(eps, name='epsilon', dtype=tf.float32)

        r_nk_tiled = tf.tile(tf.expand_dims(r_nk, 2), (1, 1, C))      # shape = N, K, C
        labels_tiled = tf.tile(tf.expand_dims(labels, 1), (1, K, 1))  # shape = N, K, C

        # for all samples in a class c, sum their probabilities of being in cluster k
        N_kc = tf.reduce_sum(tf.multiply(r_nk_tiled, labels_tiled), axis=0, name='clust_lbl_scores')  # shape = N, C

        # sum the weights for each cluster
        N_k = tf.reduce_sum(r_nk, axis=0)  # shape = N, C

        # probability that element of class C is in cluster K: p(x \in C and x \in K)
        p_kc = tf.divide(N_kc, tf.tile(tf.expand_dims(N_k + eps, 1), (1, C)), name='p_kc')

        cluster_entropy = -tf.reduce_sum(tf.multiply(p_kc, tf.log(p_kc + eps)), axis=1, name='entropy_k')  # shape = K
        entropy = tf.reduce_sum(tf.multiply(tf.divide(N_k, N), cluster_entropy), name='entropy_tot')

        cluster_purity = tf.reduce_max(p_kc, axis=1, name='purity_k')
        purity = tf.reduce_sum(tf.multiply(tf.divide(N_k, N), cluster_purity), name='purity')

        return entropy, purity
