from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time

from data import make_minibatch
from helpers.logging_utils import generate_log_id
from losses import weighted_mse, diagonal_gaussian_logprob, bernoulli_logprob, generate_missing_data_mask, \
    imputation_losses
from helpers.scheduling import create_schedule
from helpers.tf_utils import variable_on_device


def make_layer(inputs, units, stddev=1, activation=tf.tanh, name='layer', param_device='/gpu:0', seed=0):
    # make sure that variables are saved on parameter device (important in multi-GPU setup)
    with tf.device(param_device):
        return tf.layers.dense(inputs, units=units,
                               bias_initializer=tf.random_normal_initializer(stddev=stddev, dtype=tf.float32, seed=seed),
                               kernel_initializer=tf.random_normal_initializer(stddev=stddev, dtype=tf.float32, seed=seed),
                               activation=activation,
                               name=name)


def make_gaussian_layer(inputs, output_dim, stddev=1, type='standard', name='gaussian_output', param_device='/gpu:0',
                        seed=0):

    # build activation function that creates (mu, sigma^2) or (eta1, eta2) output.
    def activation(u):
        raw_1, raw_2 = tf.split(u, 2, axis=-1)
        if type == 'standard':
            with tf.name_scope('standard_parameters'):
                mean = raw_1
                var = tf.nn.softplus(raw_2, name='pos_var')
                return mean, var
        elif type == 'natparam':
            with tf.name_scope('natural_parameters'):
                eta2 = -1./2 * tf.nn.softplus(raw_2, name='pos_prec')
                eta1 = raw_1
                return eta1, eta2
        else:
            raise Exception("Type '%s' does not exist." % type)

    # double output dimension, since we have two outputs
    units = 2 * output_dim

    # make layer with defined activation function
    return make_layer(inputs, units, stddev, activation, name, param_device, seed)


def make_bernoulli_layer(input, output_dim, stddev=1, name='bernoulli_output', param_device='/gpu:0', seed=0):
    activation = lambda x: tf.identity(x, name='logits')
    return make_layer(input, output_dim, stddev, activation, name, param_device, seed)


def rand_partial_isometry(m, n, stddev, seed=0):
    """
    Initialization as in MJJ's code (Johnson et. al. 2016)
    Args:
        m: rows
        n: cols
        stddev: standard deviation
        seed: random seed

    Returns:
        matrix of shape (m, n) with orthonormal columns
    """
    d = max(m, n)
    npr = np.random.RandomState(seed)
    return np.linalg.qr(npr.normal(loc=0, scale=stddev, size=(d, d)))[0][:m,:n]


def make_nnet(input, layerspecs, stddev, name, param_device='/gpu:0', seed=0):

    with tf.variable_scope(name):
        # ravel inputs: (M, K, D) -> (M*K, D)
        input_shape = input.get_shape()
        input_dim = int(input_shape[-1])
        input = tf.reshape(input, (-1, input_dim))
        prev_layer = input

        # create all layers except the output layer
        for i, (hidden_units, activation) in enumerate(layerspecs[:-1]):
            prev_layer = make_layer(prev_layer, hidden_units, stddev, activation, 'layer_%d' % i, param_device, seed)

        # create output layer
        output_dim, type = layerspecs[-1]

        if type == 'bernoulli':
            out_mlp = make_bernoulli_layer(prev_layer, output_dim, stddev, param_device=param_device, seed=seed)
        else:
            out_mlp = make_gaussian_layer(prev_layer, output_dim, stddev, type, param_device=param_device, seed=seed)

        # create resnet-like shortcut (as in Johnson's SVAE code)
        with tf.variable_scope('shortcut'):
            orthonormal_cols = tf.constant(rand_partial_isometry(input_dim, output_dim, 1., seed=seed), dtype=tf.float32)
            W = variable_on_device('W', shape=None, initializer=orthonormal_cols, trainable=True, device=param_device,
                                   dtype=tf.float32)
            b1 = variable_on_device('b1', shape=None, initializer=tf.zeros(output_dim), trainable=True,
                                    device=param_device, dtype=tf.float32)
            out_res = tf.add(tf.matmul(input, W), b1, name='res_shortcut_1')

            # create shortcut for second output (in Gaussian case)
            if type != 'bernoulli':
                b2 = variable_on_device('b2', shape=None, initializer=tf.zeros(output_dim), trainable=True,
                                        device=param_device, dtype=tf.float32)

                if type == 'standard':
                    a = tf.constant(1., dtype=tf.float32)
                elif type == 'natparam':
                    a = tf.constant(-0.5, dtype=tf.float32)
                else:
                    raise NotImplementedError
                out_res = (out_res, tf.multiply(a, tf.log1p(tf.exp(b2)), name='res_shortcut_2'))

        with tf.variable_scope('resnet_out'):
            # unravel output: (M*K, D) -> (M, K, D)
            output_shape = input_shape[:-1].concatenate(output_dim)
            if type == 'bernoulli':
                outputs = tf.reshape(tf.add(out_mlp, out_res), output_shape, name='output')
            else:
                outputs = (
                    tf.reshape(tf.add(out_mlp[0], out_res[0]), output_shape, name='output_0'),
                    tf.reshape(tf.add(out_mlp[1], out_res[1]), output_shape, name='output_1')
                )
            return outputs


def make_encoder(input, layerspecs=None, stddev_init=1., param_device='/gpu:0', seed=0):
    if layerspecs is None:
        layerspecs = [(100, tf.tanh), (100, tf.tanh), (10, 'standard')]
    eta1, eta2_diag = make_nnet(input, layerspecs, stddev_init, 'encoder_net', param_device, seed)
    return eta1, eta2_diag


def make_decoder(input, layerspecs=None, stddev_init=1., param_device='/gpu:0', seed=0):
    if layerspecs is None:
        layerspecs = [(100, tf.tanh), (100, tf.tanh), (784, 'standard')]
    output_type = layerspecs[-1][1]

    # outputs are either Gaussian params or bernoulli logits.
    output = make_nnet(input, layerspecs, stddev_init, 'decoder_net', param_device, seed)

    # to make single-output bernoulli net compatible to Gaussian net's output, we return as well two values:
    if output_type == 'bernoulli':
        probas = tf.nn.sigmoid(output, name='pixel_probas')
        output = probas, output  # tuple (probas, logits)

    return output


def build_kl_divergence(enc_mean, enc_var, name='kl_divergence'):
    """
    Compute KL divergence according to Auto-Encoding Variational Bayes (Kingma and Welling, 2014)
    """
    # compute KL divergence between approximation and prior q(z|phi) = N(z|enc_mean, enc_sigma) and p(z) = N(z|0, 1)
    with tf.name_scope(name):
        d_kl = -tf.divide(
            # take mean of all images
            tf.reduce_mean(
                # for each image, take sum over all latent dimensions
                tf.reduce_sum(
                    1 + tf.log(enc_var) - tf.pow(enc_mean, 2) - enc_var,
                    axis=1
                )
            ),
            2.,
            name='output'
        )
    return d_kl


def expected_bernoulli_loglike(y_binary, logits, r_nk=None, name='bernoulli_expct_loglike'):
    # E[log p(y|x)]
    with tf.name_scope(name):
        if r_nk is None:
            N, S, D = logits.get_shape().as_list()
            assert y_binary.get_shape() == (N, D)
        else:
            N, K, S, D = logits.get_shape().as_list()
            assert y_binary.get_shape() == (N, D)
            assert r_nk.get_shape() == (N, K)

        # add dimensions for K and S
        y_binary = tf.expand_dims(y_binary, 1)  # N, 1, D
        if r_nk is not None:
            y_binary = tf.expand_dims(y_binary, 1)  # N, 1, 1, D

        pixel_log_probs = -tf.log(tf.add(1., tf.exp(tf.multiply(-logits, y_binary))))
        sample_log_probs = tf.reduce_sum(pixel_log_probs, axis=-1)  # sum over pixels
        img_log_probs = tf.reduce_mean(sample_log_probs, axis=-1)  # average over samples

        if r_nk is not None:
            img_log_probs = tf.reduce_sum(tf.multiply(r_nk, img_log_probs), axis=1)  # average over components

        return tf.reduce_sum(img_log_probs, name='expct_bernoulli_loglik')  # sum over minibatch


def expected_diagonal_gaussian_loglike(y, means, vars, weights=None, name='diag_gauss_expct'):
    """
    computes expected diagonal log-likelihood SUM_{n=1} E_{q(z)}[log N(x_n|mu(z), sigma(z))]
    Args:
        y: data
        means: predicted means; shape (size_minibatch, nb_samples, dims) or (size_minimbatch, nb_comps, nb_samps, dims)
        vars: predicted variances; shape is same as for means
        weights: None or matrix of shape (N, K) containing normalized weights

    Returns:

    """
    # todo refactor (merge the ifs)
    with tf.name_scope(name):
        if weights is None:
            # required dimension: size_minibatch, nb_samples, dims
            means = means if len(means.get_shape()) == 3 else tf.expand_dims(means, axis=1)
            vars = vars if len(vars.get_shape()) == 3 else tf.expand_dims(vars, axis=1)
            M, S, L = means.get_shape()
            assert y.get_shape() == (M, L)

            sample_mean = tf.reduce_sum(tf.pow(tf.expand_dims(y, axis=1) - means, 2) / vars) + tf.reduce_sum(tf.log(vars))

            S = tf.constant(int(S), dtype=tf.float32, name='number_samples')
            M = tf.constant(int(M), dtype=tf.float32, name='size_minibatch')
            L = tf.constant(int(L), dtype=tf.float32, name='latent_dimensions')
            pi = tf.constant(np.pi, dtype=tf.float32, name='pi')

            sample_mean /= S
            loglik = -1/2 * sample_mean - M * L/2. * tf.log(2. * pi)

        else:
            M, K, S, L = means.get_shape()
            assert vars.get_shape() == means.get_shape()
            assert weights.get_shape() == (M, K)

            # adjust y's shape (add component and sample dimensions)
            y = tf.expand_dims(tf.expand_dims(y, axis=1), axis=1)

            sample_mean = tf.einsum('nksd,nk->', tf.square(y - means)/ vars + tf.log(vars + 1e-8), weights)

            M = tf.constant(int(M), dtype=tf.float32, name='size_minibatch')
            S = tf.constant(int(S), dtype=tf.float32, name='number_samples')
            L = tf.constant(int(L), dtype=tf.float32, name='latent_dimensions')
            pi = tf.constant(np.pi, dtype=tf.float32, name='pi')

            sample_mean /= S
            loglik = -1/2 * sample_mean - M * L/2. * tf.log(2. * pi)

        return tf.identity(loglik, name='expct_gaussian_loglik')


def compute_elbo(y_true, enc_mu, enc_var, dec_output, decoder_type='standard', name='elbo'):
    """
    Builds tf graph for computing ELBO according to (8) in Auto-Encoding Variational Bayes (Kingma and Welling, 2014)
    Args:
        y_true: minibatch from dataset
        enc_mu: mean of q(z|x, phi) = N(z|enc_mu, enc_sigma)
        dec_output: either Gaussian std parameters (mean, var) or Bernoulli (prop, logits)
        decoder_type: specifies the type of the decoder output. either 'bernoulli' or 'standard'
        name: name of operation
    Returns:
        ELBO
    """

    with tf.name_scope(name):
        M, D = y_true.get_shape().as_list()
        d_kl = build_kl_divergence(enc_mu, enc_var)
        if decoder_type == 'bernoulli':
            _, dec_logits = dec_output
            neg_rec_err = expected_bernoulli_loglike(y_true, dec_logits)
        elif decoder_type == 'standard':
            dec_mu, dec_var = dec_output
            neg_rec_err = expected_diagonal_gaussian_loglike(y_true, dec_mu, dec_var)
        else:
            raise NotImplementedError
        neg_rec_err /= M   # mean reconstruction error over images
        elbo = tf.subtract(neg_rec_err, d_kl, name='output')
        tf.summary.scalar('D_kl', d_kl)
        tf.summary.scalar('neg_rec_err', neg_rec_err)
        return elbo


def reparam_trick_sampling(mean, var_diag, nb_samples, seed):
    with tf.name_scope('reparameterization_trick'):
        M, L = mean.get_shape()
        sample_shape = (M, tf.Dimension(nb_samples), L)

        # add sample-dimension
        expa_mean = tf.expand_dims(mean, axis=1)
        expa_var = tf.expand_dims(var_diag, axis=1)

        epsilon = tf.contrib.distributions.Normal(loc=tf.zeros(sample_shape), scale=tf.ones(sample_shape))
        var_sample = tf.einsum('msl,msl->msl', tf.sqrt(expa_var), epsilon.sample(seed=seed))

        return expa_mean + var_sample


if __name__ == '__main__':

    path_dataset = '../datasets'
    ratio_tr = 0.7
    ratio_val = None

    ratio_missing_data = 0.1

    size_minibatch = 64
    size_testbatch = 100
    nb_samples = 10
    nb_samples_te = 50
    nb_samples_pert = 20

    nb_iters = 120000
    plot_freq = 5000
    measurement_freq = 2500
    imputation_freq = 50000
    checkpoint_freq = 25000

    seed_data = 0

    log_dir = '../logs_vae'

    nb_threads = 5     # for input queue

    stddev_init = 0.01

    schedule = create_schedule({
        'dataset': 'auto',
        'method': 'vae',
        'lr': [0.0003],  # adam stepsize
        'L': [6],  # latent dimensionality
        'U': 50,  # hidden units
        'seed': range(0, 10)
    })

    # schedule = create_schedule({
    #     'dataset': 'mnist-small',
    #     'method': 'vae',
    #     'lr': [0.001],  # adam stepsize
    #     'L': [10],  # latent dimensionality
    #     'U': 50,  # hidden units
    #     'seed': 0
    # })


    ####################################################################################################################

    for config_id, config in enumerate(schedule):
        L = config['L']
        U = config['U']
        seed = config['seed']
        dataset = config['dataset']

        decoder_type = 'bernoulli' if dataset in ['fashion', 'mnist', 'mnist-small'] else 'standard'

        print("Experiment %d with config\n%s\n" % (config_id, str(config)))

        # reset Tensorflow graph
        with tf.Graph().as_default():
            # set graph-level seed
            tf.set_random_seed(config['seed'])

            y_tr, lbl_tr, y_te, lbl_te = make_minibatch(config['dataset'], ratio_tr=ratio_tr, ratio_val=ratio_val,
                                                        path_datadir=path_dataset, size_minibatch=size_minibatch,
                                                        size_testbatch=size_testbatch, nb_threads=nb_threads,
                                                        nb_towers=1, binarise=(dataset in ['mnist', 'fashion']),
                                                        seed_split=seed_data, seed_minibatch=config['seed'],
                                                        dtype=tf.float32)

            # binarise data to {-1, 1} for image datasets
            binarise_data = config['dataset'] in ['mnist', 'mnist-small', 'fashion']

            # keep original data for MSE compuation
            y_tr_01 = tf.concat(y_tr, axis=0)
            y_te_01 = y_te
            if binarise_data:
                y_tr_01 = tf.where(tf.equal(y_tr_01, -1),
                                   tf.zeros_like(y_tr_01, dtype=tf.float32),
                                   tf.ones_like(y_tr_01, dtype=tf.float32))
                y_te_01 = tf.where(tf.equal(y_te, -1),
                                   tf.zeros_like(y_te, dtype=tf.float32),
                                   tf.ones_like(y_te, dtype=tf.float32))

            N, D = y_tr.get_shape().as_list()
            N_te, _ = y_te.get_shape().as_list()

            # build neural nets
            encoder_layers = [(U, tf.tanh), (U, tf.tanh), (L, 'standard')]
            decoder_layers = [(U, tf.tanh), (U, tf.tanh), (D, decoder_type)]

            x_mean, x_var_diag = make_encoder(y_tr, layerspecs=encoder_layers, stddev_init=stddev_init, seed=seed)
            x_samp = reparam_trick_sampling(x_mean, x_var_diag, nb_samples, seed)
            y_mean, out_2 = make_decoder(x_samp, layerspecs=decoder_layers, stddev_init=stddev_init, seed=seed)

            neg_elbo = -compute_elbo(y_tr, x_mean, x_var_diag, dec_output=(y_mean, out_2), decoder_type=decoder_type)
            tf.summary.scalar('plotting_prep/elbo/elbo_normed', neg_elbo)

            opt = tf.train.AdamOptimizer(learning_rate=config['lr'], use_locking=True)
            training_op = opt.minimize(neg_elbo)

            with tf.name_scope('train_performance'):
                y_mean = tf.expand_dims(y_mean, 1)
                out_2 = tf.expand_dims(out_2, 1)
                weights = tf.ones((N, 1))  # fake weights for being able to re-use the mse method used for SVAE
                mse_tr = weighted_mse(y_tr_01, y_mean, weights)
                if decoder_type == 'bernoulli':
                    loli_tr = bernoulli_logprob(y_tr, out_2, tf.log(weights))
                else:
                    loli_tr = diagonal_gaussian_logprob(y_tr, y_mean, out_2, tf.log(weights))
                tf.summary.scalar('perf_measures/mse_tr', mse_tr)
                tf.summary.scalar('perf_measures/loli_tr', loli_tr)

            with tf.name_scope('test_performance'):
                # use current weights to reconstruct test batch
                tf.get_variable_scope().reuse_variables()
                x_te_mean, x_te_var_diag = make_encoder(y_te, layerspecs=encoder_layers, stddev_init=stddev_init, seed=seed)
                x_te_samp = reparam_trick_sampling(x_te_mean, x_te_var_diag, nb_samples_te, seed)
                y_te_mean_rec, y_te_out_2 = make_decoder(x_te_samp, layerspecs=decoder_layers, stddev_init=stddev_init, seed=seed)

                # prepare inputs for performance measurmemt
                y_te_mean_rec = tf.expand_dims(y_te_mean_rec, 1)  # shape = N, 1, S, D
                y_te_out_2 = tf.expand_dims(y_te_out_2, 1)
                weights = tf.ones((N_te, 1))  # fake weights for being able to re-use the mse method used for SVAE
                mse_te = weighted_mse(y_te_01, y_te_mean_rec, weights)
                if decoder_type == 'bernoulli':
                    loli_te = bernoulli_logprob(y_te, y_te_out_2, tf.log(weights))
                else:
                    loli_te = diagonal_gaussian_logprob(y_te, y_te_mean_rec, y_te_out_2, tf.log(weights))
                tf.summary.scalar('perf_measures/mse_te', mse_te)
                tf.summary.scalar('perf_measures/loli_te', loli_te)

            # init tensorboard
            merged = tf.summary.merge_all()
            log_id = generate_log_id(config)
            log_path = log_dir + '/' + log_id
            summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

            with tf.name_scope('test_imputation'):
                # mask is random, but constant while training run.
                missing_data_mask = generate_missing_data_mask(y_te, ratio_missing_data, seed=seed)

                # define imputation procedure for VAE (encode perturbed data end decode it again)
                def impute(y_perturbed):
                    with tf.name_scope('missing_data_imputation'):
                        tf.get_variable_scope().reuse_variables()
                        N_pert, _ = y_perturbed.get_shape().as_list()
                        x_imp_mean, x_imp_var_diag = make_encoder(y_perturbed, layerspecs=encoder_layers,
                                                                  stddev_init=stddev_init, seed=seed)
                        x_imp_samp = reparam_trick_sampling(x_imp_mean, x_imp_var_diag, nb_samples_te, seed)
                        y_imp_mean, y_imp_var_diag = make_decoder(x_imp_samp, layerspecs=decoder_layers,
                                                                  stddev_init=stddev_init, seed=seed)

                        y_imp_mean = tf.expand_dims(y_imp_mean, 1)  # shape = N, 1, S, D
                        y_imp_var_diag = tf.expand_dims(y_imp_var_diag, 1)  # shape = N, 1, S, D
                        log_weights = tf.zeros((N_pert, 1))  # fake weights for re-using the mse method

                        return y_imp_mean, y_imp_var_diag, log_weights

                imp_mse, imp_lopr = imputation_losses(y_te, missing_data_mask, impute, nb_samples_pert, nb_samples_te,
                                                      decoder_type=decoder_type, seed=seed)

                imp_smry_mse = tf.summary.scalar('imp_mse', imp_mse)
                imp_smry_lopr = tf.summary.scalar('imp_logprob', imp_lopr)
                imp_summaries = [imp_smry_mse, imp_smry_lopr]

            # save some images in summary to look at them (and their reconstruction) in tensorboard
            if config['dataset'] in ['mnist', 'mnist-small', 'fashion']:
                with tf.name_scope('sample_recs'):
                    nb_rec_samps = 6  # nb of images to be saved in summary
                    # keep a constant random sample throughout training (don't resample the images at each summary save)
                    y_te_cnst_samp = tf.Variable(y_te[:nb_rec_samps, :], trainable=False, name='y_te_samp_fixed')

                    # reconstruct images using current weights
                    tf.get_variable_scope().reuse_variables()
                    x_rec_mean, x_rec_var_diag = make_encoder(y_te_cnst_samp, layerspecs=encoder_layers,
                                                              stddev_init=stddev_init, seed=seed)
                    x_rec_samp = reparam_trick_sampling(x_rec_mean, x_rec_var_diag, nb_samples=1, seed=seed)
                    y_rec, _ = make_decoder(x_rec_samp, layerspecs=decoder_layers, stddev_init=stddev_init, seed=seed)

                    # create summaries for samples and their reconstructions
                    smp_te_true = tf.summary.image('test_samps', tf.reshape(y_te_cnst_samp, (nb_rec_samps, 28, 28, 1)),
                                                   max_outputs=nb_rec_samps)
                    smp_te_rec = tf.summary.image('test_rec_samps', tf.reshape(y_rec, (nb_rec_samps, 28, 28, 1)),
                                                  max_outputs=nb_rec_samps)


            # creat session and init variables
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            # start input queue threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(init)

            # init model saver to store trained variables
            model_saver = tf.train.Saver(max_to_keep=nb_iters // checkpoint_freq + 2)

            # save missing_data_mask to check it in tensorboard (add two axes for batch size and channel)
            mdm_summary = tf.summary.image('missing_data_mask',
                                           tf.expand_dims(tf.expand_dims(tf.to_float(missing_data_mask), 0), 3))
            summary_writer.add_summary(sess.run(mdm_summary), global_step=0)

            # save sample images to be reconstructed
            if config['dataset'] in ['mnist', 'mnist-small', 'fashion']:
                summary_writer.add_summary(sess.run(smp_te_true), 0)

            # training loop
            start_time = time.time()
            for i in range(nb_iters):
                _, loss, x, y, true, summaries = sess.run([training_op, neg_elbo, x_samp, y_mean, y_tr, merged])

                if i % checkpoint_freq == 0 or i == nb_iters - 1:
                    model_saver.save(sess, log_path + '/checkpoint', global_step=i)

                if i % measurement_freq == 0 or i == nb_iters - 1:
                    summary_writer.add_summary(summaries, global_step=i)
                    print('Iteration %5d\t\t%.4fsec\t\t%.4f' % (i, time.time() - start_time, loss))

                if i % plot_freq == 0 or i == nb_iters - 1 or i == 1:
                    if config['dataset'] in ['mnist', 'mnist-small', 'fashion']:
                        summary_writer.add_summary(sess.run(smp_te_rec), global_step=i)

                if i % imputation_freq == 0 or i == nb_iters - 1 or i == 1:
                    imp_sum_evaluated = sess.run(imp_summaries)
                    for summary in imp_sum_evaluated:
                        summary_writer.add_summary(summary, global_step=i)

                # if i % (plot_freq * 10) == 0:
                #     plt.imshow(np.hstack([y[0, 0, :].reshape((28, 28)), true[0, :].reshape((28, 28))]), cmap='gray')
                #     plt.show()
