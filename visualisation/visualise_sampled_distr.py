import numpy as np
import tensorflow as tf

from data import make_minibatch
from distributions import dirichlet, niw
from helpers.logging_utils import generate_log_id
from matplotlib.colors import ColorConverter, ListedColormap
from models import svae, vae, gmm
from sklearn.decomposition import PCA
from helpers.tf_utils import variable_on_device
from visualisation.plotting_utils import newfig, save_plot

data_dot_size = 30
grid_density = 300
nb_samples = 100000
save_path = './'

dark_colors = ('b', 'k', 'firebrick', 'g', 'olive', 'navy', 'm', 'darkviolet', 'darkseagreen')
bright_colors = ('aqua', 'magenta', 'lime', 'pink', 'yellow', 'c', 'orange', 'khaki', 'r')
markers = [',', '*', '+', 'x', 'd', 'v', '>', '.', '<', '^', 'o']


def make_colormap(color='r'):
    c_list = np.zeros((100, 4))
    c_list[:, 0:3] = ColorConverter.to_rgb(color)
    c_list[:, 3] = np.linspace(0, 1, 100)
    return ListedColormap(c_list)


def visualize_svae(ax, config, log_path, ratio_tr=0.7, nb_samples=20, grid_density=100, window=((-20, 20), (-20, 20)),
                   param_device='/cpu:0'):

    with tf.device(param_device):

        if config['dataset'] in ['mnist', 'fashion']:
            binarise = True
            size_minibatch = 1024
            output_type = 'bernoulli'
        else:
            binarise = False
            size_minibatch = -1
            output_type = 'standard'

        # First we build the model graph so that we can load the learned parameters from a checkpoint.
        # Initialisations don't matter, they'll be overwritten with saver.restore().
        data, lbl, _, _ = make_minibatch(config['dataset'], path_datadir='../datasets', ratio_tr=ratio_tr,
                                         seed_split=0, size_minibatch=size_minibatch, size_testbatch=-1,
                                         binarise=binarise)

        # define nn-architecture
        encoder_layers = [(config['U'], tf.tanh), (config['U'], tf.tanh), (config['L'], 'natparam')]
        decoder_layers = [(config['U'], tf.tanh), (config['U'], tf.tanh), (int(data.get_shape()[1]), output_type)]
        sample_size = 100

        if config['dataset'] in ['mnist', 'fashion']:
            data = tf.where(tf.equal(data, -1),
                            tf.zeros_like(data, dtype=tf.float32),
                            tf.ones_like(data, dtype=tf.float32))

        with tf.name_scope('model'):
            gmm_prior, theta = svae.init_mm(config['K'], config['L'], seed=config['seed'], param_device='/gpu:0')
            theta_copied = niw.natural_to_standard(tf.identity(gmm_prior[1]), tf.identity(gmm_prior[2]),
                                                   tf.identity(gmm_prior[3]), tf.identity(gmm_prior[4]))
            _, sigma_k = niw.expected_values(theta_copied)
            pi_k_init = tf.nn.softmax(tf.random_normal(shape=(config['K'], ), mean=0.0, stddev=1., seed=config['seed']))
            L_k = tf.cholesky(sigma_k)
            mu_k = tf.random_normal(shape=(config['K'], config['L']), stddev=1, seed=config['seed'])
            with tf.variable_scope('phi_gmm'):
                mu_k = variable_on_device('mu_k', shape=None, initializer=mu_k, trainable=True, device=param_device)
                L_k = variable_on_device('L_k', shape=None, initializer=L_k, trainable=True, device=param_device)
                pi_k = variable_on_device('log_pi_k', shape=None, initializer=pi_k_init, trainable=True, device=param_device)
            phi_gmm = mu_k, L_k, pi_k
            _ = vae.make_encoder(data, layerspecs=encoder_layers, stddev_init=.1, seed=config['seed'])

        with tf.name_scope('random_sampling'):
            # compute expected theta_pgm
            beta_k, m_k, C_k, v_k = niw.natural_to_standard(*theta[1:])
            alpha_k = dirichlet.natural_to_standard(theta[0])
            mean, cov = niw.expected_values((beta_k, m_k, C_k, v_k))
            expected_log_pi = dirichlet.expected_log_pi(alpha_k)
            pi = tf.exp(expected_log_pi)

            # sample from prior (first from
            x_k_samples = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov).sample(sample_size)
            z_samples = tf.multinomial(logits=tf.reshape(tf.log(pi), (1, -1)), num_samples=sample_size, name='k_samples')
            z_samples = tf.squeeze(z_samples)

            assert z_samples.get_shape() == (sample_size, )
            assert x_k_samples.get_shape() == (sample_size, config['K'], config['L'])

            # compute reconstructions
            y_k_samples, _ = vae.make_decoder(x_k_samples, layerspecs=decoder_layers, stddev_init=.1,
                                           seed=config['seed'])

            assert y_k_samples.get_shape() == (sample_size, config['K'], data.get_shape()[1])

        with tf.name_scope('cluster_sample_data'):
            tf.get_variable_scope().reuse_variables()
            _, clustering = svae.predict(data, phi_gmm, encoder_layers, decoder_layers, seed=config['seed'])

        # load trained model
        saver = tf.train.Saver()
        model_path = log_path + '/' + generate_log_id(config)
        print(model_path)
        latest_ckpnt = tf.train.latest_checkpoint(model_path)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=sess_config)
        saver.restore(sess, latest_ckpnt)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        collected_y_samps = []
        collected_z_samps = []
        for s in range(nb_samples):
            y_samps, z_samps = sess.run((y_k_samples, z_samples))
            collected_y_samps.append(y_samps)
            collected_z_samps.append(z_samps)
        collected_y_samps = np.concatenate(collected_y_samps, axis=0)
        collected_z_samps = np.concatenate(collected_z_samps, axis=0)
        assert collected_y_samps.shape == (nb_samples * sample_size, config['K'], data.shape[1])
        assert collected_z_samps.shape == (nb_samples * sample_size,)

        # use 300 sample points from the dataset
        data, lbl, clustering = sess.run((data[:300], lbl[:300], clustering[:300]))

        # compute PCA if necessary
        samples_2d = []
        if data.shape[1] > 2:
            pca = PCA(n_components=2).fit(data)
            data2d = pca.transform(data)

            for z_samples in range(config['K']):
                chosen = collected_z_samps == z_samples
                samps_k = collected_y_samps[chosen, z_samples, :]
                if samps_k.size > 0:
                    samples_2d.append(pca.transform(samps_k))
        else:
            data2d = data
            for z_samples in range(config['K']):
                chosen = (collected_z_samps == z_samples)
                samps_k = collected_y_samps[chosen, z_samples, :]
                if samps_k.size > 0:
                    samples_2d.append(samps_k)

        # plot 2d-histogram (one histogram for each of the K components)
        from matplotlib.colors import LogNorm
        for z_samples, samples in enumerate(samples_2d):
            ax.hist2d(samples[:, 0], samples[:, 1], bins=grid_density, range=window,
                      cmap=make_colormap(dark_colors[z_samples % len(dark_colors)]), normed=True, norm=LogNorm())

        # overlay histogram with sample datapoints (coloured according to their most likely cluster allocation)
        labels = np.argmax(lbl, axis=1)
        for c in np.unique(labels):
            in_class_c = (labels == c)
            color = bright_colors[int(c % len(bright_colors))]
            marker = markers[int(c % len(markers))]
            ax.scatter(data2d[in_class_c, 0], data2d[in_class_c, 1], c=color, marker=marker, s=data_dot_size,
                       linewidths=0)


def visualize_vae(ax, config, log_path, ratio_tr=0.7, nb_samples=20, grid_density=100, window=((-20, 20), (-20, 20)),
                   param_device='/cpu:0'):

    with tf.device(param_device):
        data, lbl, _, _= make_minibatch(config['dataset'], path_datadir='../datasets', ratio_tr=ratio_tr,
                                         seed_split=0, size_minibatch=-1, size_testbatch=-1)

        # First we build the model graph so that we can load the learned parameters from a checkpoint.
        # Initialisations don't matter, they'll be overwritten with saver.restore().
        encoder_layers = [(config['U'], tf.tanh), (config['U'], tf.tanh), (config['L'], 'natparam')]
        decoder_layers = [(config['U'], tf.tanh), (config['U'], tf.tanh), (int(data.get_shape()[1]), 'standard')]
        sample_size = 100
        with tf.name_scope('model'):
            x_mean, x_var_diag = vae.make_encoder(data, layerspecs=encoder_layers, stddev_init=.1, seed=config['seed'])
            x_samp = vae.reparam_trick_sampling(x_mean, x_var_diag, nb_samples, config['seed'])

            # generate random samples from prior N(x|0,1)
            x_samples = tf.contrib.distributions.MultivariateNormalDiag(loc=tf.zeros((config['L'])),
                                                                        scale_diag=tf.ones((config['L']))).sample(sample_size)
            y_mean, _ = vae.make_decoder(x_samples, layerspecs=decoder_layers, stddev_init=.1,
                                           seed=config['seed'])
            assert y_mean.get_shape() == (sample_size, data.get_shape()[1])

        saver = tf.train.Saver()
        model_path = log_path + '/' + generate_log_id(config)
        print(model_path)
        latest_ckpnt = tf.train.latest_checkpoint(model_path)
        latest_ckpnt = model_path + '/checkpoint-100000'

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=sess_config)
        saver.restore(sess, latest_ckpnt)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        collected_samples = []
        for s in range(nb_samples):
            y_samps = sess.run((y_mean))
            collected_samples.append(y_samps)

        data, lbl = sess.run((data[:300], lbl[:300]))

        collected_samples = np.concatenate(collected_samples, axis=0)
        assert collected_samples.shape == (nb_samples * sample_size, data.shape[1])

        if data.shape[1] > 2:
            pca = PCA(n_components=2).fit(data)
            data2d = pca.transform(data)
            samples_2d = pca.transform(collected_samples)
        else:
            data2d = data
            samples_2d = collected_samples

        from matplotlib.colors import LogNorm
        ax.hist2d(samples_2d[:, 0], samples_2d[:, 1], bins=grid_density, range=window,
                  cmap=make_colormap('black'), normed=True, norm=LogNorm())

        labels = np.argmax(lbl, axis=1)
        for c in np.unique(labels):
            in_class_c = (labels == c)
            color = bright_colors[int(c % len(bright_colors))]
            marker = markers[int(c % len(markers))]
            ax.scatter(data2d[in_class_c, 0], data2d[in_class_c, 1], c=color, marker=marker, s=data_dot_size,
                       linewidths=0)


def visualize_gmm(ax, config, log_path, ratio_tr=0.7, nb_samples=20, grid_density=100, window=((-20, 20), (-20, 20)),
                   param_device='/cpu:0'):

    with tf.device(param_device):
        data, lbl, _, _ = make_minibatch(config['dataset'], path_datadir='../datasets', ratio_tr=ratio_tr,
                                         seed_split=0, size_minibatch=-1, size_testbatch=-1)

        _, D = data.get_shape().as_list()

        # define nn-architecture
        sample_size = 100

        update, log_r_nk, theta, (x_k, S_k, pi) = gmm.inference(data, config['K'], config['seed'])

        tf.get_variable_scope().reuse_variables()
        r_nk_te, _ = gmm.e_step(data, *theta)
        clustering = tf.argmax(r_nk_te, axis=1)

        components = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=x_k, covariance_matrix=S_k + 1e-8 * tf.eye(D))
        # sample from components
        y_k_samples = components.sample(sample_size)
        c = tf.multinomial(logits=tf.reshape(tf.log(pi), (1, -1)), num_samples=sample_size, name='k_samples')
        c = tf.squeeze(c)

        assert c.get_shape() == (sample_size, )
        assert y_k_samples.get_shape() == (sample_size, config['K'], D)

        saver = tf.train.Saver()
        model_path = log_path + '/' + generate_log_id(config)
        print(model_path)
        latest_ckpnt = tf.train.latest_checkpoint(model_path)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=sess_config)
        saver.restore(sess, latest_ckpnt)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        collected_samples = []
        collected_ks = []
        for s in range(nb_samples):
            y_samps, k_samps = sess.run((y_k_samples, c))
            collected_samples.append(y_samps)
            collected_ks.append(k_samps)

        data, lbl, clustering = sess.run((data[:300], lbl[:300], clustering[:300]))

        collected_samples = np.concatenate(collected_samples, axis=0)
        collected_ks = np.concatenate(collected_ks, axis=0)
        assert collected_samples.shape == (nb_samples * sample_size, config['K'], data.shape[1])
        assert collected_ks.shape == (nb_samples * sample_size,)

        samples_2d = []
        if data.shape[1] > 2:
            pca = PCA(n_components=2).fit(data)
            data2d = pca.transform(data)

            for c in range(config['K']):
                chosen = collected_ks == c
                samps_k = collected_samples[chosen, c, :]
                if samps_k.size > 0:
                    samples_2d.append(pca.transform(samps_k))
        else:
            data2d = data
            for c in range(config['K']):
                chosen = (collected_ks == c)
                samps_k = collected_samples[chosen, c, :]
                if samps_k.size > 0:
                    samples_2d.append(samps_k)

        from matplotlib.colors import LogNorm
        for k, samples in enumerate(samples_2d):
            ax.hist2d(samples[:, 0], samples[:, 1], bins=grid_density, range=window,
                      cmap=make_colormap(dark_colors[k]), normed=True, norm=LogNorm())

        labels = np.argmax(lbl, axis=1)
        for c in np.unique(labels):
            in_class_c = (labels == c)
            color = bright_colors[int(c % len(bright_colors))]
            marker = markers[int(c % len(markers))]
            ax.scatter(data2d[in_class_c, 0], data2d[in_class_c, 1], c=color, marker=marker, s=data_dot_size,
                       linewidths=0)


if __name__ == '__main__':
    def ratio_hw(width):
        fig_width_pt = 345.0
        inches_per_pt = 1.0 / 72.27
        w_inch = fig_width_pt * width * inches_per_pt

        return w_inch, 0.9*w_inch

    def ratio_hw_2h(width):
        fig_width_pt = 345.0
        inches_per_pt = 1.0 / 72.27
        w_inch = fig_width_pt * width * inches_per_pt

        return 0.6*w_inch, w_inch

    # fig, ax = newfig(1, ratio_hw=ratio_hw)
    # ax.axis('off')
    # visualise_dataset(ax, 'pinwheel', window=[(-20, 20), (-20, 20)])
    # save_plot('pinwheel_visu', path=save_path, dpi=200)

    # config = {
    #     'dataset': 'pinwheel',
    #     'method': 'gmm',
    #     'K': 10,
    #     'seed': 0
    # }
    # fig, ax = newfig(1, ratio_hw=ratio_hw)
    # ax.axis('off')
    # visualize_gmm(ax, config=config, grid_density=grid_density, nb_samples=nb_samples, window=[(-20, 20), (-20, 20)],
    #               log_path='../pinwheel')
    # save_plot('pinwheel_visu_gmm', path=save_path, dpi=200)

    # config = {
    #     'dataset': 'pinwheel',
    #     'method': 'vae',
    #     'lr': 0.001,  # adam stepsize
    #     'L': 2,  # latent dimensionality
    #     'U': 50,  # hidden units
    #     'seed': 0
    # }
    # fig, ax = newfig(1, ratio_hw=ratio_hw)
    # ax.axis('off')
    # visualize_vae(ax, config=config, grid_density=grid_density, nb_samples=nb_samples,
    # log_path='../pinwheel', window=[(-20, 20), (-20, 20)])
    # save_plot('pinwheel_visu_vae', path=save_path, dpi=200)

    # config = {
    #     'dataset': 'pinwheel',
    #     'method': 'svae-cvi',
    #     'lr': 0.001,  # adam stepsize
    #     'lrcvi': 0.05,  # cvi stepsize (convex combination)
    #     'decay_rate': 1,
    #     'delay': 0,
    #     'K': 10,  # nb components
    #     'L': 2,  # latent dimensionality
    #     'U': 50,  # hidden units
    #     'seed': 0
    # }
    # fig, ax = newfig(1, ratio_hw=ratio_hw)
    # ax.axis('off')
    # visualize_svae(ax, config=config, grid_density=grid_density, nb_samples=nb_samples, window=[(-20, 20), (-20, 20)],
    #                log_path='../pinwheel')
    # save_plot('pinwheel_visu_svae', path=save_path, dpi=200)

    ####################################################################################################################

    # config = {
    #     'dataset': 'auto',
    #     'method': 'svae-cvi',
    #     'lr': 0.0003,    # adam stepsize
    #     'lrcvi': 0.2,    # cvi stepsize (convex combination)
    #     'decay_rate': 0.95,
    #     'K': 10,           # nb components
    #     'L': 6,          # latent dimensionality
    #     'U': 50,           # hidden units
    #     'seed': 1
    # }
    # fig, ax3 = newfig(1, ratio_hw=ratio_hw)
    # ax3.axis('off')
    # visualize_svae(ax3, config=config, grid_density=grid_density, nb_samples=nb_samples, window=[(-22, 28), (-15, 15)],
    #                log_path='../logs_svae')
    # save_plot('../auto_visu_svae', path=save_path, dpi=200)

    config = {
        'dataset': 'auto',
        'method': 'vae',
        'lr': 0.0003,  # adam stepsize
        'L': 6,  # latent dimensionality
        'U': 50,  # hidden units
        'seed': 0
    }
    fig, ax = newfig(1, ratio_hw=ratio_hw)
    # fig, (ax31, ax32) = new_subplots(1, 2, 1, ratio_hw=ratio_hw_2h)
    ax.axis('off')
    visualize_vae(ax, config=config, grid_density=grid_density, nb_samples=nb_samples, window=[(-22, 28), (-15, 15)],
                  log_path='../logs_vae')
    save_plot('auto_visu_vae', path=save_path, dpi=200)

    # config = {
    #     'dataset': 'auto',
    #     'method': 'gmm',
    #     'K': 10,
    #     'seed': 10
    # }
    # fig, ax = newfig(1, ratio_hw=ratio_hw)
    # ax.axis('off')
    # visualize_gmm(ax, config=config, grid_density=grid_density, nb_samples=nb_samples, window=[(-22, 28), (-15, 15)],
    #               log_path='../auto_gmm')
    # save_plot('auto_visu_gmm', path=save_path, dpi=200)
