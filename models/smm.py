from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data import make_minibatch
from distributions import dirichlet, niw
from helpers.logging_utils import generate_log_id
from losses import weighted_mse
from helpers.scheduling import create_schedule
from visualisation.visualise_gmm import plot_clusters, plt
from helpers.tf_utils import logdet
from models import svae

"""
Bayesian Mixture of Student-t distributions, according to:
  Robust Bayesian Clustering
  C. Archambeau and M. Verleysen
  Neural Networks, 20:129-138, 2007. Elsevier.
"""

def update_Nk(r_nk):
    # N * pi_bar_k; eq 34
    return tf.reduce_sum(r_nk, axis=0, name='Nk_new')


def update_Wk(ru_nk):
    # N * omega_bar_k; eq 35
    return tf.reduce_sum(ru_nk, axis=0, name='Wk_new')


def update_xk(x, ru_nk, W_k, eps=1e-20):
    # eq. 32
    with tf.name_scope('update_xk'):
        x_k = tf.einsum('nk,nd->kd', ru_nk, x)
        eps = tf.constant(eps, dtype=tf.float32, name='eps')
        return tf.divide(x_k, tf.expand_dims(W_k, axis=1) + eps, name='xk_new')


def update_Sk(x, ru_nk, W_k, x_k, eps=1e-20):
    # eq. 33
    with tf.name_scope('update_Sk'):
        err = tf.expand_dims(x, axis=1) - tf.expand_dims(x_k, axis=0)  # shape = N, K, D
        sqerr = tf.einsum('nkd,nke->nkde', err, err)
        S_k = tf.einsum('nk,nkde->kde', ru_nk, sqerr)
        eps = tf.constant(eps, dtype=tf.float32, name='eps')
        return tf.divide(S_k, tf.expand_dims(tf.expand_dims(W_k, axis=1), axis=2) + eps, name='Sk_new')


def update_alphak(alpha_0, N_k):
    # eq. 27
    return tf.add(alpha_0, N_k, name='new_alphak')


def update_betak(beta_0, W_k):
    # eq. 28
    return tf.add(beta_0, W_k, name='new_betak')


def update_mk(beta_0, m_0, W_k, x_k, beta_k):
    # eq. 29
    with tf.name_scope('update_m'):
        if len(beta_0.get_shape()) == 1:
            beta_0 = tf.reshape(beta_0, (-1, 1))

        Nk_xk = tf.multiply(tf.expand_dims(W_k, axis=1), x_k)
        beta0_m0 = np.multiply(beta_0, m_0)
        return tf.divide(beta0_m0 + Nk_xk, tf.expand_dims(beta_k, axis=1), name='mk_new')


def update_vk(v_0, N_k):
    # eq. 30
    return tf.add(v_0, N_k, name='vk_new')


def update_Ck(C_0, x_k, W_k, m_0, beta_0, beta_k, S_k):
    # eq. 31
    with tf.name_scope('update_C_k'):
        C = C_0 + tf.multiply(tf.expand_dims(tf.expand_dims(W_k, axis=1), axis=2), S_k)
        err = x_k - m_0
        sqerr = tf.einsum('kd,ke->kde', err, err)
        return tf.add(C, tf.einsum('k,kde->kde', np.divide(np.multiply(beta_0, W_k), beta_k), sqerr), name='W_new')


def expct_mahalanobis_dist(x, beta_k, m_k, P_k, v_k):
    # E_{q(mu, Sigma)}[(x-mk)^T Sigma (x-mk)]
    with tf.name_scope('compute_mhlnbs_dist'):
        _, D = x.get_shape().as_list()

        dist = tf.expand_dims(x, axis=1) - tf.expand_dims(m_k, axis=0)  # shape=(N, K, D)
        m = tf.einsum('k,nk->nk', v_k,
                      tf.einsum('nkd,nkd->nk', dist,
                                tf.einsum('kde,nke->nkd', P_k, dist)))
        return tf.add(m, tf.reshape(tf.divide(D, beta_k), (1, -1)), name='expct_mhlnbs_dist')   # shape=(1, K)


def expct_log_det_prec(v_k, P_k):
    with tf.name_scope('compute_log_det_prec'):
        log_det_P = logdet(P_k)

        K, D, _ = P_k.get_shape().as_list()
        D_log_2 = D * tf.log(tf.constant(2., dtype=tf.float32))

        i = tf.expand_dims(tf.range(D, dtype=tf.float32), axis=0)
        sum_digamma = tf.reduce_sum(tf.digamma(0.5 * (tf.expand_dims(v_k, axis=1) + i)), axis=1)

        return tf.identity(sum_digamma + D_log_2 + log_det_P, name='expct_log_det_prec')


def expct_log_pi(alpha_k):
    with tf.name_scope('compute_log_pi'):
        alpha_hat = tf.reduce_sum(alpha_k)
        return tf.subtract(tf.digamma(alpha_k), tf.digamma(alpha_hat), name='expct_log_pi')


def compute_rnk(expct_log_pi, expct_log_det_prec, expct_m_dist, kappa_k, D):
    # Responsibilities: eq. 19
    with tf.name_scope('compute_rnk'):
        log_r_nk = tf.lgamma((D + kappa_k) / 2.) - tf.lgamma(kappa_k / 2.) - (D / 2.) * tf.log(kappa_k * np.pi)
        log_r_nk += expct_log_pi + 0.5 * expct_log_det_prec
        log_r_nk -= 0.5 * (D + kappa_k) * expct_m_dist - tf.log(kappa_k)

        # normalise
        log_Z = tf.reduce_logsumexp(log_r_nk, axis=1, keep_dims=True)
        return tf.exp(log_r_nk - log_Z, name='rnk_new')


def compute_expct_unk(expct_m_dist, kappa_k, D):
    # Scale parameters: eq. 24, eq. 25
    with tf.name_scope('compute_unk'):
        a_nk = 0.5 * (D + kappa_k)
        b_nk = 0.5 * (expct_m_dist + kappa_k)

        return tf.divide(a_nk, b_nk, name='expct_unk')


def e_step(x, alpha_k, beta_k, m_k, P_k, v_k, kappa_k, name='e_step'):
    """
    Variational E-update: update local parameters
    Args:
        x: data
        alpha_k: Dirichlet parameter
        beta_k: NW param; variance of mean
        m_k: NW param; mean
        P_k: NW param: precision
        v_k: NW param: degrees of freedom
        kappa_k: Student-t param: degrees of freedom

    Returns:
        responsibilities, scale variables and mixture coefficients
    """
    with tf.name_scope(name):
        m_dist = expct_mahalanobis_dist(x, beta_k, m_k, P_k, v_k)
        log_det_prec = expct_log_det_prec(v_k, P_k)
        log_pi = expct_log_pi(alpha_k)
        _, D = x.get_shape().as_list()

        r_nk = compute_rnk(log_pi, log_det_prec, m_dist, kappa_k, D)
        u_nk = compute_expct_unk(m_dist, kappa_k, D)

        return r_nk, u_nk, tf.exp(log_pi)


def m_step(x, r_nk, u_nk, alpha_0, beta_0, m_0, C_0, v_0, name='m_step'):
    """
    Variational M-update: Update global parameters
    Args:
        x: data
        r_nk: responsibilities
        u_nk: scale variables
        alpha_0: prior Dirichlet parameters
        beta_0: prior NiW; controls variance of mean
        m_0: prior of mean
        C_0: prior Covariance
        v_0: prior degrees of freedom

    Returns:
        posterior parameters and data statistics
    """
    with tf.name_scope(name):
        ru_nk = tf.multiply(r_nk, u_nk, name='ru_nk')
        N_k = update_Nk(r_nk)
        W_k = update_Wk(ru_nk)
        x_k = update_xk(x, ru_nk, W_k)
        S_k = update_Sk(x, ru_nk, W_k, x_k)

        alpha_k = update_alphak(alpha_0, N_k)
        beta_k = update_betak(beta_0, W_k)
        m_k = update_mk(beta_0, m_0, W_k, x_k, beta_k)
        C_k = update_Ck(C_0, x_k, W_k, m_0, beta_0, beta_k, S_k)
        v_k = update_vk(v_0, N_k)

        return alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k


def inference(x, K, kappa_init, seed, name='inference'):
    """

    Args:
        x: data; shape = N, D
        K: number of components
        kappa_init: student-t degrees of freedom
        seed: random seed

    Returns:

    """
    with tf.name_scope(name):
        N, D = x.get_shape().as_list()

        with tf.name_scope('init_local_vars'):
            r_nk = tf.Variable(
                tf.contrib.distributions.Dirichlet(tf.ones(K)).sample(N, seed=seed),
                dtype=tf.float32,
                name='r_nk')
            u_nk = tf.Variable(tf.ones((N, K)), dtype=tf.float32, name='u_nk')

        with tf.name_scope('init_prior'):
            # returns Dirichlet+NiW natural parameters
            alpha, A, b, beta, v_hat = svae.init_mm_params(K, D, alpha_scale=0.05 / K, beta_scale=0.5,
                                                           m_scale=0,
                                                           C_scale=D + 0.5,
                                                           v_init=D + 0.5, seed=seed, name='prior',
                                                           trainable=False)
            beta_0, m_0, C_0, v_0 = niw.natural_to_standard(A, b, beta, v_hat)
            alpha_0 = dirichlet.natural_to_standard(alpha)
            kappa_k = kappa_init * tf.ones(K)  # student-t degrees of freedom

        with tf.name_scope('em_algorithm'):
            alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k = m_step(x, r_nk, u_nk, alpha_0, beta_0, m_0, C_0, v_0)
            P_k = tf.matrix_inverse(C_k)
            r_nk_new, u_nk_new, pi = e_step(x, alpha_k, beta_k, m_k, P_k, v_k, kappa_k)

            # define step: update r_nk and u_nk
            step = tf.group(r_nk.assign(r_nk_new), u_nk.assign(u_nk_new))

            # group global parameters
            theta = tf.tuple((alpha_k, beta_k, m_k, C_k, v_k, kappa_k), name='theta')

            log_r_nk = tf.log(r_nk_new)

        return step, log_r_nk, theta, (x_k, S_k, pi)


if __name__ == '__main__':
    path_dataset = '../datasets'
    ratio_tr = 0.7
    ratio_val = None
    missing_data_ratio = 0.1

    nb_iters = 200

    seed = 0
    seed_data = 0

    log_dir = '../debug'

    schedule = create_schedule({
        'method': 'smm',
        'dataset': 'aggregation',  # 't4.8k',
        'K': 10,
        'kappa': 9999,
        'seed': 0
    })


    ####################################################################################################################

    for config_id, config in enumerate(schedule):
        K = config['K']
        kappa = config['kappa']
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

            update, log_r_nk, theta, (x_k, S_k, pi) = inference(x, K, kappa, seed)
            r_nk = tf.exp(log_r_nk)

            # get cluster means and covs
            x_rec_means = tf.tile(tf.expand_dims(tf.expand_dims(x_k, 0), 2), (N, 1, 1, 1))  # shape = N, K, 1, D
            x_rec_vars = tf.expand_dims(tf.tile(tf.expand_dims(S_k, 0), (N, 1, 1, 1)), 2)  # shape = N, K, 1, D, D

            mse_tr = weighted_mse(x, x_rec_means, r_nk)
            # loli_tr, _ = gaussian_logprob(x, x_rec_means, x_rec_vars, tf.log(r_nk + 1e-8))
            tf.summary.scalar('mse_tr', mse_tr)
            # tf.summary.scalar('loli_tr', loli_tr)

            with tf.name_scope('test_performance/perf_measures'):
                # use trained theta to predict component responsibilities for test data
                r_nk_te, _, _ = e_step(x_te, *theta)

                # prepare 'reconstructions'
                x_rec_means = tf.tile(tf.expand_dims(tf.expand_dims(x_k, 0), 2), (N_te, 1, 1, 1))  # shape = N, K, 1, D
                x_rec_vars = tf.expand_dims(tf.tile(tf.expand_dims(S_k, 0), (N_te, 1, 1, 1)), 2)  # shape = N, K, 1, D, D

                mse_te = weighted_mse(x_te, x_rec_means, r_nk_te)
                # loli_te, _ = gaussian_logprob(x_te, x_rec_means, x_rec_vars, tf.log(r_nk_te + 1e-8))
                tf.summary.scalar('mse_te', mse_te)
                # tf.summary.scalar('loli_te', loli_te)

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

            # prepare logging
            merged = tf.summary.merge_all()
            log_id = generate_log_id(config)
            summary_writer = tf.summary.FileWriter(log_dir + '/' + log_id, graph=tf.get_default_graph())
            model_saver = tf.train.Saver()

            for i in range(nb_iters):
                # compute means, covariances, responsibilities and mixing coefficients
                print('Iteration %d' % i)
                _, resps, mixing_coeff, centers, covs, summaries = sess.run([update, r_nk, pi, x_k, S_k, merged])
                print('==============================\n\n')
                summary_writer.add_summary(summaries, global_step=i)

                ax.clear()
                plot_clusters(x_np, centers, covs, resps, mixing_coeff, ax=ax)
                plt.pause(0.1)
            model_saver.save(sess, log_dir + '/' + log_id + '/checkpoint', global_step=nb_iters)
            # plt.savefig(log_dir + '/' + log_id + '.png')
            # plt.close()
