import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, rc
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns

# Plotting settings
TITLE_SIZE = 14
LABEL_SIZE = 11
LEGEND_TITLE_SIZE = 12
LEGEND_SIZE = 11
TICK_SIZE = 11
FONT = 'serif'
params = {}
params['legend.title_fontsize'] = LEGEND_TITLE_SIZE
params['axes.labelsize'] = LABEL_SIZE
params['axes.titlesize'] = TITLE_SIZE
params['legend.fontsize'] = LEGEND_SIZE
params["xtick.labelsize"] = TICK_SIZE
params["ytick.labelsize"] = TICK_SIZE
params["font.family"] = "Times New Roman"
CONTEXT = sns.plotting_context("paper", rc=params)


def adjust_color(color, amount=1.0):
    """
    Lightens the given color when amount is in [0, 1).
    Darkens the given color when amount > 1.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> adjust_color('g', 0.3)  # Lighten
    >> adjust_color('#F034A3', 1.5)  # Darken
    >> adjust_color((.3,.55,.1), 2)  # Darken
    """
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = np.array(mcolors.to_rgb(c))

    if amount < 1:  # Lighten
        c += (1-c) * abs(amount)
    else:  # Darken
        c -= c * (amount-1)

    return c


def set_axis_style(axis, title, xlabel='Time t', ylabel='Risk', legend=False):
    """Sets the style for an axis."""
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if legend:
        axis.legend()


def plot_data(axis, data, label_prefix, color, linestyle='-', linewidth=2, legend=False):
    """Plots data on the given axis."""
    for i, series in enumerate(data):
        axis.plot(series, color=color[i], ls=linestyle,
                  lw=linewidth, label=f"{label_prefix} {i + 1}")
    set_axis_style(axis, f"{label_prefix} risk", ylabel='', legend=legend)


def get_colors(n_colors, special_color=None):
    """Generates or selects colors for plotting."""
    default_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                      'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    if special_color:
        return [special_color] * n_colors
    else:
        if n_colors <= len(default_colors):
            return default_colors[:n_colors]
        else:
            # Cycle through the default colors
            additional_colors = [default_colors[i % len(default_colors)] for i in range(
                n_colors - len(default_colors))]
            return default_colors + additional_colors


def plot_experiment(average_risks_subpop, average_risks_learner, all_risks, axs,
                    plot_subpop=[0], plot_learners=[0],
                    ls='-', lw=2, color_adj=1,  legend=False, special_c=None):

    n_learners, n_subpops = all_risks[0].shape
    c = get_colors(max(n_subpops, n_learners), special_color=special_c)
    for subpop in plot_subpop:
        # Average Losses of the subpops
        color = adjust_color(c[subpop], color_adj)
        axs[0].plot([a[subpop] for a in average_risks_subpop],
                    color=color, ls=ls, lw=lw, label=f"Subpop {subpop + 1}")
        axs[0].set_xlabel('Time t')
        axs[0].set_ylabel('Risk')
        axs[0].set_title('Subpopulation 1 risk')
        if legend:
            axs[0].legend()

    for learner in plot_learners:
        # Learner Risks
        color = adjust_color(c[learner], color_adj)
        axs[1].plot([a[learner] for a in average_risks_learner],
                    color=color, ls=ls, lw=lw, label=f"Learner {learner + 1}")
        axs[1].set_xlabel('Time t')
        axs[1].set_ylabel('')
        axs[1].set_title('Learner 2 risk')
        if legend:
            axs[1].legend()

    total_risk = [np.sum(round_risks) for round_risks in all_risks]
    color = adjust_color('k', color_adj)
    if special_c is not None:
        color = special_c
    axs[2].plot(total_risk, ls=ls, lw=lw+1, c=color)
    axs[2].set_xlabel('Time t')
    axs[2].set_ylabel('')
    axs[2].set_title('Total risk')
