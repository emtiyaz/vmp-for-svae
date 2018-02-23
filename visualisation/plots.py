# import matplotlib.pyplot as plt
import numpy as np
from visualisation.plotting_utils import newfig, save_plot, mpl
from helpers.logging_utils import get_summaries, get_summaries_np
from helpers.scheduling import create_schedule

cvi_color = 'C3'
width_gridlines = 0.1


def customized_figsize(width):
    fig_width_pt = 345.0  # get this value from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27

    w_inch = width * fig_width_pt * inches_per_pt
    return w_inch, w_inch*0.75


def load_m_std(path, schedule, summary_tag):
    if schedule[0]['method'] == 'svae_mjj':
        generate_log_id = lambda config: '%s_%s_ssnat%.5f_ss%.5f_K%d_i%d_seed%d.npy' % (config['dataset'], 'baseline',
                                                                                        config['sgd_step_size'],
                                                                                        config['adam_step_size'],
                                                                                        config['K'],
                                                                                        config['inner_loop'],
                                                                                        config['seed'])
        steps, data = get_summaries_np(path, schedule, summary_tag, generate_log_id)
    else:
        steps, data = get_summaries(path, schedule, summary_tag)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return steps, data, mean, std


def plot_summary(path, schedule, summary_tag, ax, label, marker, start=0, end=-1, neg=False):
    steps, _, mean, std = load_m_std(path, schedule, summary_tag)

    if neg:
        mean = -mean
    steps = steps[start:end]
    mean = mean[start:end]
    std = std[start:end]

    marker_pos = np.array([500, 2000, 7000, 20000, 75000])

    if schedule[0]['method'] == 'gmm':
        steps = [1]
        steps.extend(marker_pos.tolist())
        mean = np.repeat([mean[-1]], len(steps))
        std = np.array([0] * len(steps))
        color = 'C0'
    elif schedule[0]['method'] == 'vae':
        color = 'C2'
    elif schedule[0]['method'] == 'svae-cvi':
        color = 'C3'
    elif schedule[0]['method'] == 'svae-cvi-smm':
        color = 'C8'
    elif schedule[0]['method'] == 'svae_mjj':
        color = 'C1'
    else:
        raise NotImplementedError

    marker_pos = np.where(np.isin(steps, marker_pos))[0].tolist()

    ax.plot(steps, mean, label=label, marker=marker, markevery=marker_pos, mfc='white', ms=7, color=color)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.3, color=color)


def compare_auto():
    mpl.rcParams.update({"legend.fontsize": 4})

    # path_svae = '../raiden_auto/auto_svae'
    path_svae_smm = '../auto_svae_smm'
    path_svae = '../logs_svae'
    path_vae = '../logs_vae'
    path_gmm = '../auto_gmm'

    # load SVAE-GMM
    schedule_svae = create_schedule({
        'dataset': 'auto',
        'method': 'svae-cvi',
        'lr': [0.0003],  # adam stepsize
        'lrcvi': [0.2],  # cvi stepsize (convex combination)
        'decay_rate': [0.95],
        'K': 10,  # nb components
        'L': [6],  # latent dimensionality
        'U': 50,  # hidden units
        'seed': range(0, 10)  # todo
    })

    # load SVAE-SMM
    schedule_svae_smm = create_schedule({
        'dataset': 'auto',
        'method': 'svae-cvi-smm',
        'lr': [0.0003],  # adam stepsize
        'lrcvi': [0.2],  # cvi stepsize (convex combination)
        'decay_rate': [0.95],
        'K': 10,  # nb components
        'L': [6],  # latent dimensionality
        'U': 50,  # hidden units
        'DoF': 5,
        'seed': range(10)  # todo
    })

    # load VAE
    schedule_vae = create_schedule({
        'dataset': 'auto',
        'method': 'vae',
        'lr': [0.0003],  # adam stepsize
        'L': [6],  # latent dimensionality
        'U': 50,  # hidden units
        'seed': range(10)
    })

    # load VAE
    schedule_gmm = create_schedule({
        'method': 'gmm',
        'dataset': 'auto',
        'K': 5,
        'seed': range(10)
    })

    label_svae_smm = 'SAN-SMM'
    label_svae = 'SAN-GMM'
    label_vae = 'VAE'
    label_gmm = 'GMM'

    fig, ax = newfig(0.5, ratio_hw=customized_figsize)
    plot_summary(path_gmm, schedule_gmm, 'mse_te', ax, label_gmm, marker='o')
    plot_summary(path_vae, schedule_vae, 'mse_te', ax, label_vae, marker='+')
    plot_summary(path_svae, schedule_svae, 'mse_te', ax, label_svae, marker='D')
    plot_summary(path_svae_smm, schedule_svae_smm, 'mse_te', ax, label_svae_smm, marker='s')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([400, 100000])
    ax.set_ylim([1, 200])
    ax.grid(which='both', linewidth=width_gridlines)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('MSE')

    # import matplotlib.lines as mlines
    # handles, labels = ax.get_legend_handles_labels()
    # handles.insert(2, mlines.Line2D([], [], color='C1', marker='x', markersize=7))
    # labels.insert(2, 'SVAE (Johnson et. al.)')
    # ax.legend(handles, labels, loc='lower left')
    ax.legend()

    save_plot('auto_mse_te_smm', path='./')


def compare_pinwheel():

    # path = '../pinwheel'
    path_san = '../pinwheel_new'
    path_mjj = '../../svae/experiments/pinwheel_new'

    # load SVAE
    schedule_svae = create_schedule({
        'dataset': 'pinwheel',
        'method': 'svae-cvi',
        'lr': [0.01],  # adam stepsize
        'lrcvi': [0.1],  # cvi stepsize (convex combination)
        'decay_rate': 1,
        'delay': 0,
        'K': 10,  # nb components
        'L': 2,   # latent dimensionality
        'U': 40,  # hidden units
        'seed': range(10)
    })

    # load VAE
    schedule_vae = create_schedule({
        'dataset': 'pinwheel',
        'method': 'vae',
        'lr': [0.005],  # adam stepsize
        'L': 2,  # latent dimensionality
        'U': 40,  # hidden units
        'seed': range(10)
    })

    # load GMM
    schedule_gmm = create_schedule({
        'method': 'gmm',
        'dataset': 'pinwheel',
        'K': 10,
        'seed': range(10)
    })

    # load SVAE-Johnson
    schedule_mjj = create_schedule({
        'method': 'svae_mjj',
        'dataset': 'pinwheel',
        'inner_loop': [100],
        'sgd_step_size': [10],
        'adam_step_size': [0.005],
        'K': 10,
        'seed': range(10)
    })

    label_svae = 'SAN-GMM'
    label_mjj = 'SVAE (Johnson et. al.)'
    label_vae = 'VAE'
    label_gmm = 'GMM'

    # ax = ax3
    fig, ax = newfig(0.5, ratio_hw=customized_figsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(which='both', linewidth=width_gridlines)
    ax.set_xlim([100, 10000])
    ax.set_ylim([0.5, 100])
    plot_summary(path_san, schedule_gmm, 'mse_te', ax, label_gmm, marker='o', start=1)
    plot_summary(path_san, schedule_vae, 'mse_te', ax, label_vae, marker='+', start=1)
    plot_summary(path_mjj, schedule_mjj, 'test_rmse_100', ax, label_mjj, marker='x', start=1)
    plot_summary(path_san, schedule_svae, 'mse_te', ax, label_svae, marker='D', start=1)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('MSE')
    ax.grid(which='both', linewidth=width_gridlines)
    # ax.legend()
    # plt.show()
    save_plot('pinwheel_mse_te', path='./')


def boxplot_noisy_pinwheel():
    # plots performance of latent SMM for increasing noise level

    path = '../noisy_pinwheel_wu'

    schedule_svae_smm = create_schedule({
        'dataset': 'noisy-pinwheel',
        'method': 'svae-cvi-smm',
        'lr': [0.01],  # adam stepsize
        'lrcvi': [0.005],  # cvi stepsize (convex combination)
        'K': 10,  # nb components
        'L': 2,  # latent dimensionality
        'U': 40,  # hidden units
        'DoF': 5,
        'noise_level': 0.1,
        'seed': range(1, 4)
    })
    _, mse1, mse1m, mse1s = load_m_std(path, schedule_svae_smm, 'mse_te')

    schedule_svae_smm = create_schedule({
        'dataset': 'noisy-pinwheel',
        'method': 'svae-cvi-smm',
        'lr': [0.01],  # adam stepsize
        'lrcvi': [0.005],  # cvi stepsize (convex combination)
        'K': 10,  # nb components
        'L': 2,  # latent dimensionality
        'U': 40,  # hidden units
        'DoF': 5,
        'noise_level': 0.3,
        'seed': range(1, 4)
    })
    _, mse3, mse3m, mse3s = load_m_std(path, schedule_svae_smm, 'mse_te')

    schedule_svae_smm = create_schedule({
        'dataset': 'noisy-pinwheel',
        'method': 'svae-cvi-smm',
        'lr': [0.01],  # adam stepsize
        'lrcvi': [0.0005],  # cvi stepsize (convex combination)
        'K': 10,  # nb components
        'L': 2,  # latent dimensionality
        'U': 40,  # hidden units
        'DoF': 5,
        'noise_level': 0.5,
        'seed': range(1, 4)
    })
    _, mse5, mse5m, mse5s = load_m_std(path, schedule_svae_smm, 'mse_te')

    schedule_svae_smm = create_schedule({
        'dataset': 'noisy-pinwheel',
        'method': 'svae-cvi-smm',
        'lr': [0.01],  # adam stepsize
        'lrcvi': [0.0005],  # cvi stepsize (convex combination)
        'K': 10,  # nb components
        'L': 2,  # latent dimensionality
        'U': 40,  # hidden units
        'DoF': 5,
        'noise_level': 0.7,
        'seed': range(1, 4)
    })
    _, mse7, mse7m, mse7s = load_m_std(path, schedule_svae_smm, 'mse_te')

    lsmm_mean = np.vstack([mse1m[-1], mse3m[-1], mse5m[-1], mse7m[-1]])
    lsmm_std = np.vstack([mse1s[-1], mse3s[-1], mse5s[-1], mse7s[-1]])

    # extract svae gmm
    lgmm_mean = np.zeros(4)
    lgmm_std = np.zeros(4)
    for i, lvl in enumerate([0.1, 0.3, 0.5, 0.7]):
        schedule_svae_gmm = create_schedule({
            'dataset': 'noisy-pinwheel',
            'method': 'svae-cvi',
            'lr': [0.01],  # adam stepsize
            'lrcvi': [0.005],  # cvi stepsize (convex combination)
            'K': 10,  # nb components
            'L': 2,  # latent dimensionality
            'U': 40,  # hidden units
            'DoF': 5,
            'noise_level': lvl,
            'seed': range(1, 4)
        })
        _, mse, mean, std = load_m_std(path, schedule_svae_gmm, 'mse_te')

        lgmm_mean[i] = mean[-1]
        lgmm_std[i] = std[-1]

    # extract simple gmm
    gmm_mean = np.zeros(4)
    gmm_std = np.zeros(4)
    for i, lvl in enumerate([0.1, 0.3, 0.5, 0.7]):
        schedule_svae_gmm = create_schedule({
            'dataset': 'noisy-pinwheel',
            'method': 'gmm',
            'K': 10,  # nb components
            'noise_level': lvl,
            'seed': range(1, 4)
        })
        _, mse, mean, std = load_m_std(path, schedule_svae_gmm, 'mse_te')
        gmm_mean[i] = mean[-1]
        gmm_std[i] = std[-1]


    fig, ax = newfig(0.5, ratio_hw=customized_figsize)
    # ax.set_xscale('log')
    ax.set_yscale('log')

    #(4, 3)
    ax.errorbar(np.transpose([1, 2, 3, 4]), gmm_mean, yerr=gmm_std, color="C0", label="GMM")
    ax.errorbar(np.transpose([1, 2, 3, 4]), lgmm_mean, yerr=lgmm_std, color="C3", label="SAN-GMM")
    ax.errorbar(np.transpose([1, 2, 3, 4]), lsmm_mean, yerr=lsmm_std, color="C8", label="SAN-TMM")

    ax.set_xticklabels(["10%", "30%", "50%", "70%"])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim([0.7, 4.3])
    ax.set_ylim([0.5, 200])
    ax.set_xlabel('Ratio Outliers')
    ax.set_ylabel('MSE')
    ax.grid(which='both', linewidth=width_gridlines)
    ax.legend()
    save_plot('noisy-pinwheel_errorbars', path='./')


if __name__ == '__main__':
    boxplot_noisy_pinwheel()
    # compare_pinwheel()
    # compare_auto()