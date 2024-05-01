import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, rc
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns

# Plotting settings
rc('text', usetex=False)
rc('font', family='serif')
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
context = sns.plotting_context("paper", rc=params)
sns.set_theme(style="whitegrid", font=FONT)


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
        return default_colors[:n_colors]
