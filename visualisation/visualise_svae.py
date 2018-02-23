import matplotlib.pyplot as plt
import numpy as np

from visualisation import visualise_gmm
from sklearn.decomposition import PCA

plt.ion()
fig, ax = plt.subplots(2, 2)


def svae_dashboard(iter, y_te, y_te_rec, x_samps, r_nk, cluster_alloc, theta, size_minibatch, perf_meas_iters, measurement_freq, elbo, debug_meas, debug):

    means, covs, pi = theta

    for rows in ax:
        for subplot in rows:
            subplot.clear()

    if y_te.shape[1] > 2:
        pca = PCA(n_components=2).fit(y_te)
        y_te_2d = pca.transform(y_te)
        y_te_rec_2d = pca.transform(y_te_rec)
    else:
        y_te_2d = y_te
        y_te_rec_2d = y_te_rec


    ax[0, 0].set_title('Weights (pi_theta)')
    ax[0, 0].bar(np.arange(len(pi)), pi, color=visualise_gmm.colours)

    measurement_iter = int(iter / measurement_freq)
    idx = slice(max(0, measurement_iter - 50), measurement_iter)
    ax[0, 1].plot(perf_meas_iters[idx], -elbo[idx] * size_minibatch, 'r', label='ELBO')
    ax[0, 1].plot(perf_meas_iters[idx], debug_meas[idx, 0], 'g:', label='reconstr')
    ax[0, 1].plot(perf_meas_iters[idx], debug_meas[idx, 1], 'b-', label='num')
    ax[0, 1].plot(perf_meas_iters[idx], debug_meas[idx, 2], 'c.-', label='denom')
    ax[0, 1].plot(perf_meas_iters[idx], debug_meas[idx, 3], 'm:', label='KL')
    ax[0, 1].legend(loc='lower left')
    ax[0, 1].set_title('Debugging')
    ax[0, 1].grid(True)

    visualise_gmm.plot_clusters(x_samps, means, covs, r_nk, pi, ax=ax[1, 0], title='Latent Space')

    if debug is not None:
        means_p, covs_p, pi_p = debug
        visualise_gmm.plot_clusters(x_samps, means_p, covs_p, r_nk, pi_p, ax=ax[1, 0], title='Latent Space')

    visualise_gmm.plot_clustered_data(y_te_2d, y_te_rec_2d, cluster_alloc, ax=ax[1, 1])
    ax[1, 1].set_title('Data')

    plt.pause(0.001)

    return plt
    # print('iter=%d' % iter)
    # plt.pause(10)
