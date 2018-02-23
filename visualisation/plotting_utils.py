"""
Helper functions for saving nice plots ready to add to latex document
Strongly inspired by http://bkanuka.com/articles/native-latex-plots/
"""


import numpy as np
import math
import matplotlib as mpl
mpl.use('pgf')


def default_figsize(scale):
    fig_width_pt = 345.0                            # Hard coded; get this from LaTeX using \the\textwidth

    if scale <= 0:
        raise AttributeError('The scale must be greater than 0.')
    elif scale > 1:
        print('This plot will not fit into a LaTex document of textwidth = %f.' % fig_width_pt)

    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic height-width ratio

    # increase height-width ratio if 0<scale<0.5 to make sure that plot is not squeezed because of axis labels
    # hard coded; logistic_func_05_1: [0.5, 1] -> [0, 1]
    logistic_func_05_1 = lambda x: 1 - 1 / (1 + math.exp(-17 * (x - 0.75)))
    height_ratio_adjustment = (1 - golden_mean) * logistic_func_05_1(scale)
    hw_ratio = golden_mean + height_ratio_adjustment

    # width and height in inches:
    fig_width = fig_width_pt * inches_per_pt*scale
    fig_height = fig_width * hw_ratio
    return fig_width, fig_height


def figsize_equal_hw(scale):
    fig_width_pt = 345.0                # get this value from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27

    # width and height in inches:
    fig_side_length = fig_width_pt * inches_per_pt * scale
    return fig_side_length, fig_side_length


# set plotting parameters
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "text.fontsize": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "figure.titlesize": 'large',
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": default_figsize(1),   # default fig size of 1 textwidth

    # figure borders
    "figure.subplot.left": 0.0,         # the left side of the subplots of the figure
    "figure.subplot.right": 1,          # the right side of the subplots of the figure
    "figure.subplot.bottom": 0,      # the bottom of the subplots of the figure
    "figure.subplot.top": 1,
    "figure.subplot.wspace": 0,
    "figure.subplot.hspace": 0,

    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        ]
    }
mpl.rcParams.update(pgf_with_latex)

# import pyplot AFTER setting the schedule
import matplotlib.pyplot as plt


def newfig(width, ratio_hw=default_figsize):
    """
    Creates new figure of desired width
    Args:
        width: float in [0, 1). If width>1, it won't fit in LaTex document.
    """
    plt.clf()
    fig = plt.figure(figsize=ratio_hw(width), frameon=False)
    ax = fig.add_subplot(111)
    return fig, ax


def new_subplots(width=1, nrows=1, ncols=1, compute_wh_inch=default_figsize):
    """
    Creates new figure of desired width with nrows * ncols subplots
    Args:
        width: float in [0, 1). If width>1, it won't fit in LaTex document.
    """
    plt.clf()
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=compute_wh_inch(width))
    return fig, axs


def save_plot(plot_name, path='', dpi=None):
    """
    Saves current figure.
    """
    # make sure that the axis labels and the title are visible
    plt.tight_layout()

    # save figure
    plt.savefig(path + '/' + plot_name + '.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(path + '/' + plot_name + '.pdf', dpi=dpi, bbox_inches='tight')
    # plt.savefig(path + '/' + plot_name + '.pgf', dpi=dpi, bbox_inches='tight')
