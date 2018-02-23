from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from distributions import dirichlet, gaussian, niw, student_t
from models import vae, gmm, smm
import re

from helpers.tf_utils import variable_on_device


def e_step(phi_enc, phi_gmm, nb_samples, seed=0, name="e_step"):
    """

    Args:
        phi_enc: encoded data; Gaussian natural parameters
        phi_gmm: paramters of recognition GMM (eta1_phi2, eta2_phi2, pi_phi2)
        nb_samples: number of times to sample from q(x|z, y)
        seed: random seed
        name: tensorflow name scope

    Returns:

    """
    with tf.name_scope(name):
        eta1_phi1, eta2_phi1_diag = phi_enc
        eta2_phi1 = tf.matrix_diag(eta2_phi1_diag, name='diagonalize')

        # get gaussian natparams and dirichlet natparam for recognition GMM
        eta1_phi2, eta2_phi2, pi_phi2 = unpack_recognition_gmm(phi_gmm)

        # compute log q(z|y, phi)
        log_z_given_y_phi, dbg = compute_log_z_given_y(eta1_phi1, eta2_phi1, eta1_phi2, eta2_phi2, pi_phi2)

        # compute parameters phi_tilde (corresponds to mu_tilde and sigma_tilde in the paper)
        # eta1_phi_tilde.shape = (N, K, D, 1); eta2_phi_tilde.shape = (N, K, D, D)
        with tf.name_scope('combine_phi'):
            eta1_phi_tilde = tf.expand_dims(tf.expand_dims(eta1_phi1, axis=1) + tf.expand_dims(eta1_phi2, axis=0), axis=-1, name='combine_eta1')
            eta2_phi_tilde = tf.add(tf.expand_dims(eta2_phi1, axis=1), tf.expand_dims(eta2_phi2, axis=0), name='combine_eta2')
            phi_tilde = tf.tuple((eta1_phi_tilde, eta2_phi_tilde), name='phi_tilde')

        # sample x for each of the K components (x_samps.shape = size_minibatch, nb_components, nb_samples, latent_dim)
        x_k_samples = sample_x_per_comp(eta1_phi_tilde, eta2_phi_tilde, nb_samples, seed)

        return x_k_samples, log_z_given_y_phi, phi_tilde, dbg


def compute_log_z_given_y(eta1_phi1, eta2_phi1, eta1_phi2, eta2_phi2, pi_phi2, name='log_q_z_given_y_phi'):
    """

    Args:
        eta1_phi1: encoder output; shape = N, K, L
        eta2_phi1: encoder output; shape = N, K, L, L
        eta1_phi2: GMM-EM parameter; shape = K, L
        eta2_phi2: GMM-EM parameter; shape = K, L, L
        name: tensorflow name scope

    Returns:
        log q(z|y, phi)
    """
    with tf.name_scope(name):
        N, L = eta1_phi1.get_shape().as_list()
        assert eta2_phi1.get_shape() == (N, L, L)
        K, L2 = eta1_phi2.get_shape().as_list()
        assert L2 == L
        assert eta2_phi2.get_shape() == (K, L, L)

        # combine eta2_phi1 and eta2_phi2
        eta2_phi_tilde = tf.add(tf.expand_dims(eta2_phi1, axis=1), tf.expand_dims(eta2_phi2, axis=0))

        # w_eta2 = -0.5 * inv(sigma_phi1 + sigma_phi2)
        solved = tf.matrix_solve(eta2_phi_tilde, tf.tile(tf.expand_dims(eta2_phi2, axis=0), [N, 1, 1, 1]))
        w_eta2 = tf.einsum('nju,nkui->nkij', eta2_phi1, solved)

        # for nummerical stability...
        w_eta2 = tf.divide(w_eta2 + tf.matrix_transpose(w_eta2), 2., name='symmetrised')

        # w_eta1 = inv(sigma_phi1 + sigma_phi2) * mu_phi2
        w_eta1 = tf.einsum('nuj,nkuv->nkj',
                           eta2_phi1,
                           tf.matrix_solve(eta2_phi_tilde,
                                           tf.tile(tf.expand_dims(tf.expand_dims(eta1_phi2, axis=0), axis=-1),
                                                   [N, 1, 1, 1]))  # shape inside solve= N, K, D, 1
                           )  # w_eta1.shape = N, K, D

        # compute means
        mu_phi1, _ = gaussian.natural_to_standard(eta1_phi1, eta2_phi1)

        # compute log_z_given_y_phi
        return gaussian.log_probability_nat(mu_phi1, w_eta1, w_eta2, pi_phi2), (w_eta1, w_eta2)


def sample_x_per_comp(eta1, eta2, nb_samples, seed=0):
    """
    Args:
        eta1: 1st Gaussian natural parameter, shape = N, K, L, 1
        eta2: 2nd Gaussian natural parameter, shape = N, K, L, L
        nb_samples: nb of samples to generate for each of the K components
        seed: random seed

    Returns:
        x ~ N(x|eta1[k], eta2[k]), nb_samples times for each of the K components.
    """
    with tf.name_scope('sample_x_k'):
        inv_sigma = -2 * eta2
        N, K, _, D = eta2.get_shape()

        # cholesky decomposition and adding noise (raw_noise is of dimension (DxB), where B is the size of MC samples)
        L = tf.cholesky(inv_sigma)
        # sample_shape = (D, nb_samples)
        sample_shape = (int(N), int(K), int(D), nb_samples)
        raw_noise = tf.random_normal(sample_shape, mean=0., stddev=1., seed=seed)
        noise = tf.matrix_solve(tf.matrix_transpose(L), raw_noise)

        # reparam-trick-sampling: x_samps = mu_tilde + noise: shape = N, K, S, D
        x_k_samps = tf.transpose(tf.matrix_solve(inv_sigma, eta1) + noise, [0, 1, 3, 2], name='samples')
        return x_k_samps


def subsample_x(x_k_samples, log_q_z_given_y, seed=0):
    """
    Given S samples for each of the K components for N datapoints (x_k_samples) and q(z_n=k|y), subsample S samples for
    each data point
    Args:
        x_k_samples: sample matrix of shape (N, K, S, L)
        log_q_z_given_y: probability q(z_n=k|y_n, phi)
        seed: random seed
    Returns:
        x_samples: a sample matrix of shape (N, S, L)
    """
    with tf.name_scope('subsample_x'):
        N, K, S, L = x_k_samples.get_shape().as_list()

        # prepare indices for N and S dimension
        # tf can't tile int32 tensors on the GPU. Therefore, tile it as float and convert to int afterwards
        n_idx = tf.to_int32(tf.tile(tf.reshape(tf.range(N, dtype=tf.float32), (-1, 1)), multiples=[1, S]))
        s_idx = tf.to_int32(tf.tile(tf.reshape(tf.range(S, dtype=tf.float32), (1, -1)), multiples=[N, 1]))

        # sample S times z ~ q(z|y, phi) for each N.
        z_samps = tf.multinomial(logits=log_q_z_given_y, num_samples=S, seed=seed, name='z_samples')
        z_samps = tf.cast(z_samps, dtype=tf.int32)

        # tensor of shape (N, S, 3), containing indices of all chosen samples
        choices = tf.concat([tf.expand_dims(n_idx, 2),
                             tf.expand_dims(z_samps, 2),
                             tf.expand_dims(s_idx, 2)],
                            axis=2)

        return tf.gather_nd(x_k_samples, choices, name='x_samples')


def m_step(gmm_prior, x_samples, r_nk):
    """
    Args:
        gmm_prior: Dirichlet+NiW prior for Gaussian mixture model
        x_samples: samples of shape (N, S, L)
        r_nk: responsibilities of shape (N, K)

    Returns:
        Dirichlet+NiW parameters obtained by executing Bishop's M-step in the VEM algorithm for GMMs
    """

    with tf.name_scope('m_step'):
        # execute GMM-EM m-step
        beta_0, m_0, C_0, v_0 = niw.natural_to_standard(*gmm_prior[1:])
        alpha_0 = dirichlet.natural_to_standard(gmm_prior[0])

        alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k = gmm.m_step(x_samples, r_nk, alpha_0, beta_0, m_0, C_0, v_0,
                                                              name='gmm_m_step')

        A, b, beta, v_hat = niw.standard_to_natural(beta_k, m_k, C_k, v_k)
        alpha = dirichlet.standard_to_natural(alpha_k)

        return tf.tuple([alpha, A, b, beta, v_hat], name='theta_star')


def m_step_smm(smm_prior, r_nk):
    """
    Args:
        smm_prior: Dirichlet prior for Student-t mixture model
        r_nk: responsibilities

    Returns:
        Dirichlet parameter obtained by executing Archambeau and Verleysen's M-step in their VEM algorithm for SMMs
    """

    with tf.name_scope('m_step'):
        # execute SMM-EM m-step to update multinomial
        alpha_0 = dirichlet.natural_to_standard(smm_prior[0])

        N_k = smm.update_Nk(r_nk)
        alpha_k = smm.update_alphak(alpha_0, N_k)
        alpha = dirichlet.standard_to_natural(alpha_k)
        return tf.identity(alpha, name='alpha_star')


def compute_elbo(y, reconstructions, theta, phi_tilde, x_k_samps, log_z_given_y_phi, decoder_type):
    # ELBO for latent GMM
    with tf.name_scope('elbo'):
        # unpack phi_gmm and compute expected theta
        with tf.name_scope('expct_theta_to_nat'):
            beta_k, m_k, C_k, v_k = niw.natural_to_standard(*theta[1:])
            mu, sigma = niw.expected_values((beta_k, m_k, C_k, v_k))
            eta1_theta, eta2_theta = gaussian.standard_to_natural(mu, sigma)
            alpha_k = dirichlet.natural_to_standard(theta[0])
            expected_log_pi_theta = dirichlet.expected_log_pi(alpha_k)

            # do not backpropagate through GMM
            with tf.name_scope('block_backprop'):
                eta1_theta = tf.stop_gradient(eta1_theta)
                eta2_theta = tf.stop_gradient(eta2_theta)
                expected_log_pi_theta = tf.stop_gradient(expected_log_pi_theta)

        r_nk = tf.exp(log_z_given_y_phi)

        # compute negative reconstruction error; sum over minibatch (use VAE function)
        means, out_2 = reconstructions  # out_2 is either gaussian variances or bernoulli logits.
        if decoder_type == 'standard':
            neg_reconstruction_error = vae.expected_diagonal_gaussian_loglike(y, means, out_2, weights=r_nk)
        elif decoder_type == 'bernoulli':
            neg_reconstruction_error = vae.expected_bernoulli_loglike(y, out_2, r_nk=r_nk)
        else:
            raise NotImplementedError

        # compute E[log q_phi(x,z=k|y)]
        eta1_phi_tilde, eta2_phi_tilde = phi_tilde
        N, K, L, _ = eta2_phi_tilde.get_shape().as_list()
        eta1_phi_tilde = tf.reshape(eta1_phi_tilde, (N, K, L))

        N, K, S, L = x_k_samps.get_shape().as_list()

        with tf.name_scope('compute_regularizer'):
            with tf.name_scope('log_numerator'):
                log_N_x_given_phi = gaussian.log_probability_nat_per_samp(x_k_samps, eta1_phi_tilde, eta2_phi_tilde)
                log_numerator = log_N_x_given_phi + tf.expand_dims(log_z_given_y_phi, axis=2)

            with tf.name_scope('log_denominator'):
                log_N_x_given_theta = gaussian.log_probability_nat_per_samp(x_k_samps,
                                                                            tf.tile(tf.expand_dims(eta1_theta, axis=0), [N, 1, 1]),
                                                                            tf.tile(tf.expand_dims(eta2_theta, axis=0), [N, 1, 1, 1]))
                log_denominator = log_N_x_given_theta + tf.expand_dims(tf.expand_dims(expected_log_pi_theta, axis=0), axis=2)

            regularizer_term = tf.reduce_mean(
                tf.reduce_sum(
                    tf.reduce_sum(
                        tf.multiply(tf.expand_dims(r_nk, axis=2),
                                    log_numerator - log_denominator),
                        axis=1),  # weighted average over components
                    axis=0)  # sum over minibatch
            )  # mean over samples

        elbo = tf.subtract(neg_reconstruction_error, regularizer_term, name='elbo')

        with tf.name_scope('elbo_summaries'):
            details = tf.tuple((neg_reconstruction_error,
                                tf.reduce_sum(tf.multiply(r_nk, tf.reduce_mean(log_numerator, -1))),
                                tf.reduce_sum(tf.multiply(r_nk, tf.reduce_mean(log_denominator, -1))),
                                regularizer_term), name='debug')

        return elbo, details


def compute_elbo_smm(y, reconstructions, theta, phi_tilde, x_k_samps, log_z_given_y_phi, decoder_type):
    # ELBO for latent SMM
    with tf.name_scope('elbo'):
        # unpack phi_gmm and compute expected theta
        mu_theta, sigma_theta = unpack_smm(theta[1:3])
        alpha_k = dirichlet.natural_to_standard(theta[0])
        expected_log_pi_theta = dirichlet.expected_log_pi(alpha_k)
        dof = theta[3]  # Student-t degrees of freedom

        # make sure that gradient is not propagated through stochastic Dirichlet parameter
        with tf.name_scope('block_backprop'):
            expected_log_pi_theta = tf.stop_gradient(expected_log_pi_theta)
            dof = tf.stop_gradient(dof)

        r_nk = tf.exp(log_z_given_y_phi)

        # compute negative reconstruction error; sum over minibatch (use VAE function)
        means, out_2 = reconstructions  # out_2 is either gaussian variances or bernoulli logits.
        if decoder_type == 'standard':
            neg_reconstruction_error = vae.expected_diagonal_gaussian_loglike(y, means, out_2, weights=r_nk)
        elif decoder_type == 'bernoulli':
            neg_reconstruction_error = vae.expected_bernoulli_loglike(y, out_2, r_nk=r_nk)
        else:
            raise NotImplementedError

        eta1_phi_tilde, eta2_phi_tilde = phi_tilde
        N, K, L, _ = eta2_phi_tilde.get_shape().as_list()
        eta1_phi_tilde = tf.reshape(eta1_phi_tilde, (N, K, L))

        with tf.name_scope('compute_regularizer'):
            # compute E[log q_phi(x,z=k|y)]
            with tf.name_scope('log_numerator'):
                log_N_x_given_phi = gaussian.log_probability_nat_per_samp(x_k_samps, eta1_phi_tilde, eta2_phi_tilde)
                log_numerator = log_N_x_given_phi + tf.expand_dims(log_z_given_y_phi, axis=2)

            with tf.name_scope('log_denominator'):
                # compute E[log p_theta(x,z=k)]
                log_N_x_given_theta = student_t.log_probability_per_samp(x_k_samps, mu_theta, sigma_theta, dof)
                log_denominator = log_N_x_given_theta + tf.expand_dims(tf.expand_dims(expected_log_pi_theta, axis=0), axis=2)

            regularizer_term = tf.reduce_mean(
                tf.reduce_sum(
                    tf.reduce_sum(
                        tf.multiply(tf.expand_dims(r_nk, axis=2),
                                    log_numerator - log_denominator),
                        axis=1),  # weighted average over components
                    axis=0)  # sum over data points
            )  # mean over samples

        elbo = tf.subtract(neg_reconstruction_error, regularizer_term, name='elbo')

        with tf.name_scope('elbo_summaries'):
            details = tf.tuple((neg_reconstruction_error,
                                tf.reduce_sum(tf.multiply(r_nk, tf.reduce_mean(log_numerator, -1))),
                                tf.reduce_sum(tf.multiply(r_nk, tf.reduce_mean(log_denominator, -1))),
                                regularizer_term), name='debug')

        return elbo, details

# todo delete
def unpack_recognition_gmm_debug(phi_gmm, name='unpack_phi2'):

    with tf.name_scope(name):
        eta1, L_k_raw, pi_k_raw = phi_gmm

        # make sure that L is a valid Cholesky decomposition and compute covariance
        with tf.name_scope('compute_prec'):
            L_k = tf.contrib.linalg.LinearOperatorTriL(L_k_raw, name='to_triL').to_dense()
            L_k = tf.matrix_set_diag(L_k, tf.nn.softplus(tf.matrix_diag_part(L_k), name='softplus_diag'), name='L')
            P = tf.matmul(L_k, tf.matrix_transpose(L_k), name='precision')

        eta2 = tf.multiply(tf.constant(-0.5, dtype=tf.float32), P)
        pi_k = tf.nn.softmax(pi_k_raw)

        return eta1, eta2, pi_k, L_k_raw


def unpack_recognition_gmm(phi_gmm, name='unpack_phi2'):

    with tf.name_scope(name):
        eta1, L_k_raw, pi_k_raw = phi_gmm

        # make sure that L is a valid Cholesky decomposition and compute precision
        with tf.name_scope('compute_prec'):
            L_k = tf.contrib.linalg.LinearOperatorTriL(L_k_raw, name='to_triL').to_dense()
            L_k = tf.matrix_set_diag(L_k, tf.nn.softplus(tf.matrix_diag_part(L_k), name='softplus_diag'), name='L')
            P = tf.matmul(L_k, tf.matrix_transpose(L_k), name='precision')

        eta2 = tf.multiply(tf.constant(-0.5, dtype=tf.float32), P)

        # make sure that log_pi_k are valid mixture coefficients
        pi_k = tf.nn.softmax(pi_k_raw)

        return tf.tuple((eta1, eta2, pi_k), name='phi_gmm_unpacked')


def unpack_smm(theta_smm, name='unpack_theta_smm'):
    # extract point-estimates for Student-t mixture components

    with tf.name_scope(name):
        mu, L_k_raw = theta_smm

        # make sure that L is a valid Cholesky decomposition and compute scaling matrix
        with tf.name_scope('compute_prec'):
            L_k = tf.contrib.linalg.LinearOperatorTriL(L_k_raw, name='to_triL').to_dense()
            L_k = tf.matrix_set_diag(L_k, tf.nn.softplus(tf.matrix_diag_part(L_k), name='softplus_diag'), name='L')
            Sigma = tf.matmul(L_k, tf.matrix_transpose(L_k), name='precision')

        return tf.tuple((mu, Sigma), name='theta_smm_unpacked')


def update_gmm_params(current_gmm_params, gmm_params_star, step_size, name='cvi_update_theta'):
    """
    Computes convex combination between current and updated parameters.
    Args:
        current_gmm_params: current parameters
        gmm_params_star: parameters received by GMM-EM algorithm
        step_size: step size for convex combination
        name:

    Returns:
    """
    updates = []
    with tf.name_scope(name):
        # print('===============')
        for i, (curr_param, param_star) in enumerate(zip(current_gmm_params, gmm_params_star)):
            # use param name for defining update name; proper naming simplifies debugging ;-)
            name_re = re.search(r'(?<=\/)[^\:]*', curr_param.name)
            param_name = name_re.group(0) if name_re is not None else 'param_%d' % i

            # generate list containing updates for each parameter in theta
            updates.append(
                tf.identity(
                    tf.assign(curr_param,
                              tf.add(((1 - step_size) * curr_param), (step_size * param_star), name='convex_combination'),
                              name='update_%s' % param_name)
                )
            )
        return tf.group(*updates, name='update_step')


def predict(y, phi_gmm, encoder_layers, decoder_layers, seed=0):
    """
    Args:
        y: data to cluster and reconstruct
        phi_gmm: latent phi param
        encoder_layers: encoder NN architecture
        decoder_layers: encoder NN architecture
        seed: random seed

    Returns:
        reconstructed y and most probable cluster allocation
    """
    with tf.name_scope('prediction'):
        # encode (reusing current encoder parameters)
        nb_samples = 1
        phi_enc = vae.make_encoder(y, layerspecs=encoder_layers)

        # predict cluster allocation and sample latent variables (e-step)
        x_k_samples, log_r_nk, _, _ = e_step(phi_enc, phi_gmm, nb_samples, name="svae_e_step_predict", seed=seed)
        x_samples = subsample_x(x_k_samples, log_r_nk, seed)[:, 0, :]

        # decode (reusing current decoder parameters)
        y_mean, _ = vae.make_decoder(x_samples, layerspecs=decoder_layers)

        return tf.tuple((y_mean, tf.argmax(log_r_nk, axis=1)), name='prediction')


def init_mm_params(nb_components, latent_dims, alpha_scale=.1, beta_scale=1e-5, v_init=10., m_scale=1., C_scale=10.,
                   seed=0, as_variables=True, trainable=False, device='/gpu:0', name='gmm'):

    with tf.name_scope('gmm_initialization'):
        alpha_init = alpha_scale * tf.ones((nb_components,))
        beta_init = beta_scale * tf.ones((nb_components,))
        v_init = tf.tile([float(latent_dims + v_init)], [nb_components])
        means_init = m_scale * tf.random_uniform((nb_components, latent_dims), minval=-1, maxval=1, seed=seed)
        covariance_init = C_scale * tf.tile(tf.expand_dims(tf.eye(latent_dims), axis=0), [nb_components, 1, 1])

        # transform to natural parameters
        A, b, beta, v_hat = niw.standard_to_natural(beta_init, means_init, covariance_init, v_init)
        alpha = dirichlet.standard_to_natural(alpha_init)

        # init variable
        if as_variables:
            with tf.variable_scope(name):
                alpha = variable_on_device('alpha_k', shape=None, initializer=alpha, trainable=trainable, device=device)
                A = variable_on_device('beta_k', shape=None, initializer=A, trainable=trainable, device=device)
                b = variable_on_device('m_k', shape=None, initializer=b, trainable=trainable, device=device)
                beta = variable_on_device('C_k', shape=None, initializer=beta, trainable=trainable, device=device)
                v_hat = variable_on_device('v_k', shape=None, initializer=v_hat, trainable=trainable, device=device)

        params = alpha, A, b, beta, v_hat

        return params


def init_mm(nb_components, latent_dims, seed=0, param_device='/gpu:0', name='init_mm', theta_as_variable=True):
    with tf.name_scope(name):
        # prior parameters are always tf.constant.
        theta_prior = init_mm_params(nb_components, latent_dims, alpha_scale=0.05 / nb_components, beta_scale=0.5,
                                     m_scale=0, C_scale=latent_dims + 0.5, v_init=latent_dims + 0.5,seed=seed,
                                     as_variables=False, name='theta_prior', trainable=False, device=param_device)

        theta = init_mm_params(nb_components, latent_dims, alpha_scale=1., beta_scale=1., m_scale=5.,
                               C_scale=2 * (latent_dims), v_init=latent_dims + 1., seed=seed, name='theta',
                               as_variables=theta_as_variable, trainable=False)
        return theta_prior, theta


def make_loc_scale_variables(theta, param_device='/gpu:0', name='copy_m_v'):
    # create location/scale variables for point estimations
    with tf.name_scope(name):
        theta_copied = niw.natural_to_standard(tf.identity(theta[1]), tf.identity(theta[2]),
                                               tf.identity(theta[3]), tf.identity(theta[4]))
        mu_k_init, sigma_k = niw.expected_values(theta_copied)
        L_k_init = tf.cholesky(sigma_k)

        mu_k = variable_on_device('mu_k', shape=None, initializer=mu_k_init, trainable=True, device=param_device)
        L_k = variable_on_device('L_k', shape=None, initializer=L_k_init, trainable=True, device=param_device)

        return mu_k, L_k


def init_recognition_params(theta, nb_components, seed=0, param_device='/gpu:0', var_scope='phi_gmm'):
    # make parameters for PGM part of recognition network
    with tf.name_scope('init_' + var_scope):
        pi_k_init = tf.nn.softmax(tf.random_normal(shape=(nb_components,), mean=0.0, stddev=1., seed=seed))

        with tf.variable_scope(var_scope):
            mu_k, L_k = make_loc_scale_variables(theta, param_device)
            pi_k = variable_on_device('log_pi_k', shape=None, initializer=pi_k_init, trainable=True, device=param_device)
            return mu_k, L_k, pi_k


def inference(y, phi_gmm, encoder_layers, decoder_layers, nb_samples=10, stddev_init_nn=0.01, seed=0, name='inference',
              param_device='/gpu:0'):
    with tf.name_scope(name):

        # Use VAE encoder
        x_given_y_phi = vae.make_encoder(y, layerspecs=encoder_layers, stddev_init=stddev_init_nn,
                                         param_device=param_device, seed=seed)

        # execute E-step (update/sample local variables)
        x_k_samples, log_z_given_y_phi, phi_tilde, w_eta_12 = e_step(x_given_y_phi, phi_gmm, nb_samples, seed=seed)

        # compute reconstruction
        y_reconstruction = vae.make_decoder(x_k_samples, layerspecs=decoder_layers, stddev_init=stddev_init_nn,
                                            param_device=param_device, seed=seed)

        x_samples = subsample_x(x_k_samples, log_z_given_y_phi, seed)[:, 0, :]

        return y_reconstruction, x_given_y_phi, x_k_samples, x_samples, log_z_given_y_phi, phi_gmm, phi_tilde


def identity_transform(input, nb_components, nb_samples, type='standard', name='debug_nn'):
    # debugging: freeze neural net: output Gaussian natparams corresponding to mu_n=x_n and sigma_n = I * nn_var
    with tf.name_scope(name):
        nn_var = 1e-1
        mu = input
        sigma = tf.constant(np.repeat(np.array([[nn_var, 0], [0, nn_var]])[None, :, :], mu.shape[0], axis=0), dtype=tf.float32)

        if type == 'natparam':
            eta1, eta2 = gaussian.standard_to_natural(mu, sigma)
            eta2 = tf.matrix_diag_part(eta2)
            return eta1, eta2
        else:
            sigma = tf.matrix_diag_part(sigma)
            if sigma.get_shape() != input.get_shape():
                sigma = tf.tile(tf.expand_dims(tf.expand_dims(sigma, axis=1), axis=1), [1, nb_components, nb_samples, 1])
            return mu, sigma