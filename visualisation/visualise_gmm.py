import matplotlib.pyplot as plt
import numpy as np


# ordered colours (neighbouring colours differ strongly)
colours = ('b', 'g', 'r', 'c', 'm', 'navy', 'lime', 'y', 'k', 'pink', 'orange', 'magenta', 'firebrick', 'olive',
          'aqua', 'sienna', 'khaki', 'teal', 'darkviolet', 'darkseagreen')

# ordered markers (neighbouring markers differ)
markers = [',', '+', '.', '*', 'x', 'd', 'v', '>', '<', '^', 'o']

size_datapoint_tr = 5
size_datapoint_te = 30
size_clustercenter = 100
linewidth_cluster_ellipsis = 2.


def plot_clustered_data(y_tr, y_te, clusters, ax=None):
    min = clusters.min()
    max = clusters.max()
    if ax is None:
        ax = plt
    if y_tr is not None:
        ax.scatter(y_tr[:, 0], y_tr[:, 1], color='lightgray', s=size_datapoint_tr)
    for i in range(min, max + 1):
        ax.scatter(y_te[:, 0][clusters == i], y_te[:, 1][clusters == i], marker=markers[i%len(markers)],
                   s=size_datapoint_te, color=colours[i%len(colours)])


def plot_components(mu_k, sigma_k, pi, ax):

    nb_components, _ = mu_k.shape

    # cluster centers
    ax.scatter(mu_k[:, 0], mu_k[:, 1], color=colours, marker='D', s=size_clustercenter)

    # ellipse
    if mu_k.shape[1] == 2:
        for k, weight in enumerate(pi):
            # only plot cluster cov if component weight is large enough
            if weight > 0.3/nb_components:
                t = np.linspace(0, 2 * np.pi, 100) % (2. * np.pi)
                circle = np.vstack((np.sin(t), np.cos(t)))
                ellipse = 2. * np.dot(np.linalg.cholesky(sigma_k[k, :]), circle) + mu_k[k, :, None]
                ax.plot(ellipse[0], ellipse[1], alpha=weight, linestyle='-', linewidth=linewidth_cluster_ellipsis,
                        color=colours[k % len(colours)])


def plot_clusters(data, mu_k, sigma_k, r_nk, pi, ax=None, title='Clusters'):

    if ax is None:
        f, ax = plt.subplots()

    clusters = r_nk.argmax(axis=1)
    plot_clustered_data(None, data, clusters, ax=ax)
    if title is not None:
        ax.set_title(title)

    plot_components(mu_k, sigma_k, pi, ax)

