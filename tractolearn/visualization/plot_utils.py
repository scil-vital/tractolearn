# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from tractolearn.tractoio.utils import read_data_from_json_file


def generate_decoration_rc_parameters(
    default_family="sans-serif",
    default_size=16,
    default_weight="bold",
    axes_title_weight="bold",
    axes_title_size=22,
    axes_label_size=22,
    axes_label_weight="bold",
    xtick_label_size=16,
    ytick_label_size=16,
    legend_font_size=16,
    legend_title_font_size=22,
    figure_title_size=22,
    figure_title_weight="bold",
    dpi=300,
):

    # Default text
    default_font = {
        "font": {
            "family": default_family,
            "size": default_size,
            "weight": default_weight,
        }
    }

    # Font of the axes title and x and y labels
    axes_font = {
        "axes": {
            "titleweight": axes_title_weight,
            "titlesize": axes_title_size,
            "labelsize": axes_label_size,
            "labelweight": axes_label_weight,
        }
    }

    # Font of the x and y tick labels
    xtick_font = {"xtick": {"labelsize": xtick_label_size}}
    ytick_font = {"ytick": {"labelsize": ytick_label_size}}

    # Font of the legend
    legend_font = {
        "legend": {
            "fontsize": legend_font_size,
            "title_fontsize": legend_title_font_size,
        }
    }

    # Figure title
    figure_font = {
        "figure": {
            "titlesize": figure_title_size,
            "titleweight": figure_title_weight,
            "dpi": dpi,
        }
    }

    rc_parameters = default_font
    rc_parameters.update(axes_font)
    rc_parameters.update(xtick_font)
    rc_parameters.update(ytick_font)
    rc_parameters.update(legend_font)
    rc_parameters.update(figure_font)

    return rc_parameters


def plot_loss_history(
    train_loss_history_fname,
    valid_loss_history_fname,
    out_fname,
    test_loss_history_fname=None,
):

    train_loss_history = read_data_from_json_file(train_loss_history_fname)
    valid_loss_history = read_data_from_json_file(valid_loss_history_fname)
    if test_loss_history_fname is not None:
        test_loss_history = read_data_from_json_file(test_loss_history_fname)

    n_epochs = list(range(1, len(train_loss_history) + 1))

    fig = plt.figure(figsize=(16, 10))
    ax = plt.subplot(111)
    ax.plot(n_epochs, train_loss_history, color="C0", label="train")
    ax.plot(n_epochs, valid_loss_history, color="C1", label="valid")
    if test_loss_history_fname is not None:
        ax.plot(n_epochs, test_loss_history, color="C2", label="test")

    title = "Reconstruction accuracy"
    plt.xlabel("epoch")
    plt.ylabel("loss value")
    plt.legend(loc="upper right")
    plt.title(title)

    fig.savefig(out_fname)


def upsample_coords(coord_list, num_interp_points=1000):

    # Interpolate using a B-Spline

    # Smoothness of the B-Spline
    s = 0.0
    # Degree of the  B-Spline. Setting to 1 for linear spline
    k = 1
    tck, u = interpolate.splprep(coord_list, k=k, s=s)
    upsampled_coords = interpolate.splev(np.linspace(0, 1, num_interp_points), tck)

    return upsampled_coords
