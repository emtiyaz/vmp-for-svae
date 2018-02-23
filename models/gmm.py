from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data import make_minibatch
from distributions import dirichlet, niw
from helpers.logging_utils import generate_log_id
from losses import weighted_mse, generate_missing_data_mask, imputation_mse   # , gaussian_logprob
from models import svae
from helpers.scheduling import create_schedule
from visualisation.visualise_gmm import plot_clusters, plt


def update_Nk(r_nk):
    # Bishop eq 10.51
    return tf.reduce_sum(r_nk, axis=0, name='Nk_new')


def update_xk(x, r_nk, N_k):
    # Bishop eq 10.52; output shape = (K, D)
    with tf.name_scope('update_xk'):
        x_k = tf.einsum('nk,nd->kd', r_nk, x)
        x_k_normed = tf.divide(x_k, tf.expand_dims(N_k, axis=1))
        # remove nan values (if N_k == 0)
        return tf.where(tf.is_nan(x_k_normed), x_k, x_k_normed, name='xk_new')


def update_Sk(x, r_nk, N_k, x_k):
    # Bishop eq 10.53
    with tf.name_scope('update_S_k'):
        x_xk = tf.expand_dims(x, axis=1) - tf.expand_dims(x_k, axis=0)
        S = tf.einsum('nk,nkde->kde', r_nk, tf.einsum('nkd,nke->nkde', x_xk, x_xk))
        S_normed = tf.divide(S, tf.expand_dims(tf.expand_dims(N_k, axis=1), axis=2))
        # remove nan values (if N_k == 0)
        return tf.where(tf.is_nan(S_normed), S, S_normed, name='S_new')


def update_alphak(alpha_0, N_k):
    # Bishop eq 10.58
    return tf.add(alpha_0, N_k, name='new_alphak')


def update_betak(beta_0, N_k):
    # Bishop eq 10.60
    return tf.add(beta_0, N_k, name='new_betak')


def update_mk(beta_0, m_0, N_k, x_k, beta_k):
    # Bishop eq 10.61
    with tf.name_scope('update_m'):
        if len(beta_0.get_shape()) == 1:
            beta_0 = tf.reshape(beta_0, (-1, 1))

        Nk_xk = tf.multiply(tf.expand_dims(N_k, axis=1), x_k)
        beta0_m0 = np.multiply(beta_0, m_0)
        return tf.divide(beta0_m0 + Nk_xk, tf.expand_dims(beta_k, axis=1), name='m_new')


def update_Ck(C_0, x_k, N_k, m_0, beta_0, beta_k, S_k):
    # Bishop eq 10.62
    with tf.name_scope('update_C_k'):
        C = C_0 + tf.multiply(tf.expand_dims(tf.expand_dims(N_k, axis=1), axis=2), S_k)
        Q0 = x_k - m_0
        q = tf.einsum('kd,ke->kde', Q0, Q0)
        return tf.add(C, tf.einsum('k,kde->kde', np.divide(np.multiply(beta_0, N_k), beta_k), q), name='W_new')


def update_vk(v_0, N_k):
    # Bishop eq 10.63
    return tf.identity(v_0 + N_k + 1, name='vk_new')


def compute_expct_mahalanobis_dist(x, beta_k, m_k, P_k, v_k):
    # Bishop eq 10.64
    # output shape: (N, K)
    with tf.name_scope('compute_mhlnbs_dist'):
        _, D = x.get_shape().as_list()

        dist = tf.expand_dims(x, axis=1) - tf.expand_dims(m_k, axis=0)  # shape=(N, K, D)
        m = tf.einsum('k,nk->nk', v_k,
                      tf.einsum('nkd,nkd->nk', dist,
                                tf.einsum('kde,nke->nkd', P_k, dist)))
        return tf.add(m, tf.reshape(tf.divide(D, beta_k), (1, -1)), name='expct_mhlnbs_dist')   # shape=(1, K)


def compute_dev_missing_data(x, beta_k, m_k, P_k, v_k, missing_data_mask):
    # Bishop eq 10.64; ignoring missing data
    # output shape: (N, K)
    with tf.name_scope('compute_mean'):
        _, D = x.get_shape()

        d_beta = tf.reshape(tf.divide(int(D), beta_k), (1, -1))   # shape=(1, K)
        x_mk = tf.expand_dims(x, axis=1) - tf.expand_dims(m_k, axis=0)  # shape=(N, K, D)

        # exclude missing data: set 'missing' values to zero
        av_data_mask = tf.expand_dims(tf.to_float(tf.logical_not(missing_data_mask)), axis=1, name='available_data_mask')
        x_mk = tf.multiply(x_mk, av_data_mask, name='available_data')

        m = tf.einsum('k,nk->nk', v_k,
                      tf.einsum('nkd,nkd->nk', x_mk,
                                tf.einsum('kde,nke->nkd', P_k, x_mk)))

        return tf.add(d_beta, m, name='lambda')


def compute_expct_log_det_prec(v_k, P_k):
    # Bishop eq 10.65
    with tf.name_scope('compute_expct_log_det_prec'):
        det_P = tf.matrix_determinant(P_k)
        log_det_P = tf.where(det_P > 1e-20, tf.log(det_P), tf.zeros_like(det_P, dtype=tf.float32))
        # log_det_W = tf.log()  # shape=(5,)

        K, D, _ = P_k.get_shape()
        D = tf.constant(int(D), dtype=tf.float32)
        D_log_2 = D * tf.log(tf.constant(2., dtype=tf.float32))

        i = tf.expand_dims(tf.range(D, dtype=tf.float32), axis=0)
        sum_digamma = tf.reduce_sum(tf.digamma(0.5 * (tf.expand_dims(v_k, axis=1) + 1. + i)), axis=1)

        return tf.identity(sum_digamma + D_log_2 + log_det_P, name='log_det_prec')


def compute_log_pi(alpha_k):
    # Bishop eq 10.66
    with tf.name_scope('compute_log_pi'):
        alpha_hat = tf.reduce_sum(alpha_k)
        return tf.subtract(tf.digamma(alpha_k), tf.digamma(alpha_hat), name='log_pi')


def compute_rnk(expct_log_pi, expct_log_det_cov, expct_dev):
    # Bishop eq 10.49
    with tf.name_scope('compute_rnk'):
        log_rho_nk = expct_log_pi + 0.5 * expct_log_det_cov - 0.5 * expct_dev

        # for numerical stability: subtract largest log p(z=k) for each k
        rho_nk_save = tf.exp(log_rho_nk - tf.reshape(tf.reduce_max(log_rho_nk, axis=1), (-1, 1)))

        # normalize
        rho_n_sum = tf.reduce_sum(rho_nk_save, axis=1)  # shape = (N,)
        return tf.divide(rho_nk_save, tf.expand_dims(rho_n_sum, axis=1), name='r_nk')


def e_step(x, alpha_k, beta_k, m_k, P_k, v_k, name='e_step'):
    """
    Variational E-update: update local parameters
    Args:
        x: data
        alpha_k: Dirichlet parameter
        beta_k: NW param, variance of mean
        m_k: NW param, mean
        P_k: NW param, precision
        v_k: NW param, degrees of freedom

    Returns:
        responsibilities and mixture coefficients
    """
    with tf.name_scope(name):
        expct_dev = compute_expct_mahalanobis_dist(x, beta_k, m_k, P_k, v_k)  # Bishop eq 10.64
        expct_log_det_cov = compute_expct_log_det_prec(v_k, P_k)              # Bishop eq 10.65
        expct_log_pi = compute_log_pi(alpha_k)                                # Bishop eq 10.66
        r_nk = compute_rnk(expct_log_pi, expct_log_det_cov, expct_dev)        # Bishop eq 10.49

        return r_nk, tf.exp(expct_log_pi)


def e_step_missing_data(x, alpha_k, beta_k, m_k, P_k, v_k, missing_data_mask, name='e_step_imp'):
    """
    Variational E-update: update local parameters ignoring missing data.
    Args:
        x: data
        alpha_k: Dirichlet parameter
        beta_k: NW param; variance of mean
        m_k: NW param; mean
        P_k: NW param: precision
        v_k: NW param: degrees of freedom
        missing_data_mask: binary matrix of shape (N, D) indicating missing values

    Returns:
        responsibilities and mixture coefficients
    """
    with tf.name_scope(name):
        expct_dev = compute_dev_missing_data(x, beta_k, m_k, P_k, v_k, missing_data_mask)
        expct_log_det_cov = compute_expct_log_det_prec(v_k, P_k)
        expct_log_pi = compute_log_pi(alpha_k)
        r_nk = compute_rnk(expct_log_pi, expct_log_det_cov, expct_dev)

        return r_nk, tf.exp(expct_log_pi)


def m_step(x, r_nk, alpha_0, beta_0, m_0, C_0, v_0, name='m_step'):
    """
    Variational M-update: Update global parameters
    Args:
        x: data
        r_nk: responsibilities
        alpha_0: prior Dirichlet parameters
        beta_0: prior NiW; controls variance of mean
        m_0: prior of mean
        C_0: prior Covariance
        v_0: prior degrees of freedom

    Returns:
        posterior parameters as well as data statistics
    """
    with tf.name_scope(name):
        N_k = update_Nk(r_nk)                                     # Bishop eq 10.51
        x_k = update_xk(x, r_nk, N_k)                             # Bishop eq 10.52
        S_k = update_Sk(x, r_nk, N_k, x_k)                        # Bishop eq 10.53

        alpha_k = update_alphak(alpha_0, N_k)                     # Bishop eq 10.58
        beta_k = update_betak(beta_0, N_k)                        # Bishop eq 10.60
        m_k = update_mk(beta_0, m_0, N_k, x_k, beta_k)            # Bishop eq 10.61
        C_k = update_Ck(C_0, x_k, N_k, m_0, beta_0, beta_k, S_k)  # Bishop eq 10.62
        v_k = update_vk(v_0, N_k)                                 # Bishop eq 10.63

        return alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k


def inference(x, K, seed, name='inference'):
    """

    Args:
        x: data; shape = N, D
        K: number of components
        seed: random seed
        name:

    Returns:

    """
    with tf.name_scope(name):
        N, D = x.get_shape().as_list()

        with tf.name_scope('init_responsibilities'):
            r_nk = tf.Variable(
                tf.contrib.distributions.Dirichlet(tf.ones(K)).sample(N, seed=seed),
                dtype=tf.float32,
                name='r_nk')

        with tf.name_scope('init_prior'):
            alpha, A, b, beta, v_hat = svae.init_mm_params(K, D, alpha_scale=0.05 / K, beta_scale=0.5, m_scale=0,
                                                           C_scale=D + 0.5,
                                                           v_init=D + 0.5, seed=seed, name='prior', trainable=False)
            beta_0, m_0, C_0, v_0 = niw.natural_to_standard(A, b, beta, v_hat)
            alpha_0 = dirichlet.natural_to_standard(alpha)

        with tf.name_scope('em_algorithm'):
            alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k = m_step(x, r_nk, alpha_0, beta_0, m_0, C_0, v_0)
            P_k = tf.matrix_inverse(C_k)
            r_nk_new, pi = e_step(x, alpha_k, beta_k, m_k, P_k, v_k)

            step = r_nk.assign(r_nk_new)

            theta = tf.tuple((alpha_k, beta_k, m_k, C_k, v_k), name='theta')

            log_r_nk = tf.log(r_nk_new)

        return step, log_r_nk, theta, (x_k, S_k, pi)


if __name__ == '__main__':
    path_dataset = '../datasets'
    ratio_tr = 0.7
    ratio_val = None
    missing_data_ratio = 0.1

    nb_iters = 500
    measurement_freq = 10

    K = 10  # nb components

    seed = 0
    seed_data = 0

    log_dir = '../pinwheel_new'

    schedule = create_schedule({
        'method': 'gmm',
        'dataset': 'pinwheel',
        'K': 8,
        'seed': range(10)
    })
    #
    # schedule = create_schedule({
    #     'method': 'gmm',
    #     'dataset': 'auto',
    #     'K': 10,
    #     'seed': 10
    # })

    schedule = create_schedule({
        'method': 'gmm',
        'dataset': 'aggregation',
        'K': 10,
        'seed': 0
    })

    ####################################################################################################################

    for config_id, config in enumerate(schedule):
        K = config['K']
        seed = config['seed']
        dataset = config['dataset']

        print("Experiment %d with config\n%s\n" % (config_id, str(config)))

        # reset Tensorflow graph
        with tf.Graph().as_default():
            # set graph-level seed
            tf.set_random_seed(config['seed'])

            x, lbl, x_te, lbl_te = make_minibatch(config['dataset'], ratio_tr=ratio_tr, ratio_val=ratio_val,
                                                  path_datadir=path_dataset, size_minibatch=-1, nb_towers=1,
                                                  nb_threads=2, seed_split=seed_data, seed_minibatch=seed_data,
                                                  dtype=tf.float32)

            N, D = x.get_shape().as_list()
            N_te, _ = x_te.get_shape().as_list()

            update, log_r_nk, theta, (x_k, S_k, pi) = inference(x, K, seed)
            r_nk = tf.exp(log_r_nk)

            x_rec_means = tf.tile(tf.expand_dims(tf.expand_dims(x_k, 0), 2), (N, 1, 1, 1))  # shape = N, K, 1, D
            x_rec_vars = tf.expand_dims(tf.tile(tf.expand_dims(S_k, 0), (N, 1, 1, 1)), 2)  # shape = N, K, 1, D, D

            mse_tr = weighted_mse(x, x_rec_means, r_nk)
            # loli_tr, _ = gaussian_logprob(x, x_rec_means, x_rec_vars, tf.log(r_nk + 1e-8))
            tf.summary.scalar('mse_tr', mse_tr)
            # tf.summary.scalar('loli_tr', loli_tr)

            with tf.name_scope('test_performance/perf_measures'):
                # use trained theta to predict component responsibilities for test data
                r_nk_te, _ = e_step(x_te, *theta)

                # prepare 'reconstructions'
                x_rec_means = tf.tile(tf.expand_dims(tf.expand_dims(x_k, 0), 2), (N_te, 1, 1, 1))  # shape = N, K, 1, D
                x_rec_vars = tf.expand_dims(tf.tile(tf.expand_dims(S_k, 0), (N_te, 1, 1, 1)), 2)  # shape = N, K, 1, D, D

                mse_te = weighted_mse(x_te, x_rec_means, r_nk_te)
                # loli_te, _ = gaussian_logprob(x_te, x_rec_means, x_rec_vars, tf.log(r_nk_te + 1e-8))
                tf.summary.scalar('mse_te', mse_te)
                # tf.summary.scalar('loli_te', loli_te)

            with tf.name_scope('test_imputation'):
                missing_data_mask = generate_missing_data_mask(x_te, noise_ratio=missing_data_ratio, seed=seed)
                r_nk_imp, _ = e_step_missing_data(x_te, *theta, missing_data_mask)

                # prepare 'reconstructions'
                x_imp_means = tf.tile(tf.expand_dims(tf.expand_dims(x_k, 0), 2), (N_te, 1, 1, 1))  # shape = N, K, 1, D
                x_imp_vars = tf.expand_dims(tf.tile(tf.expand_dims(S_k, 0), (N_te, 1, 1, 1)), 2)  # shape = N, K, 1, D, D

                mse_imp = imputation_mse(x_te, x_imp_means, r_nk_imp, missing_data_mask, name='comp_imp_mse')
                tf.summary.scalar('imp_mse', mse_imp)


            # create session, init variables and start input queue threads
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # prepare plotting
            x_np = sess.run(x)
            plt.ion()
            fig, ax = plt.subplots()

            merged = tf.summary.merge_all()
            log_id = generate_log_id(config)
            summary_writer = tf.summary.FileWriter(log_dir + '/' + log_id, graph=tf.get_default_graph())
            model_saver = tf.train.Saver()

            for i in range(nb_iters):
                # compute means, covariances, responsibilities and mixing coefficients
                _, resps, mixing_coeff, centers, covs, summaries = sess.run([update, r_nk, pi, x_k, S_k, merged])

                if i % measurement_freq == 0 or i == nb_iters - 1:
                    summary_writer.add_summary(summaries, global_step=i)
                    ax.clear()
                    plot_clusters(x_np, centers, covs, resps, mixing_coeff, ax=ax)
                    plt.pause(0.001)
            model_saver.save(sess, log_dir + '/' + log_id + '/checkpoint', global_step=nb_iters)
            plt.savefig(log_dir + '/' + log_id + '.png')
            plt.close()

            summary_writer.flush()
            summary_writer.close()
