# -*- coding: utf-8 -*-
import enum
import logging
import os
from os.path import join as pjoin
from time import time

import numpy as np
import umap
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from numpy import interp
from scipy.stats import mode
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from tractolearn.anatomy.bundles_additional_labels import (
    BundlesAdditionalLabels,
)
from tractolearn.filtering import LatentSpaceKeys
from tractolearn.filtering.latent_space_distance_informer import (
    LatentSpaceDistanceInformer,
)
from tractolearn.logger import LoggerKeys
from tractolearn.tractoio.utils import (
    save_data_to_pickle_file,
    save_streamlines,
)
from tractolearn.utils.layer_utils import PredictWrapper
from tractolearn.visualization.plot_utils import (
    generate_decoration_rc_parameters,
    upsample_coords,
)

# fixme: FD
logger = logging.getLogger("root")

ROC_MIN_NUM_POINTS = 1000
ROC_MAX_NUM_POINTS = 5000


# ToDo
# Think if this should rather be FilteringThresholdPoint
class ROCSalientPoint(enum.Enum):
    MAX_ACCURACY = "max_accuracy"
    INV_DIAGONAL_INTERSECT = "inv_diagonal_intersect"


# ToDo
# Generalize and make this into a utils/anatomy script
def get_dataset_long_name_from_dataset_name(dataset_name):

    if dataset_name == "fibercup":
        dataset_long_name = '"Fiber Cup"'
    elif dataset_name == "ismrm2015_phantom":
        dataset_long_name = "ISMRM 2015 Tractography Challenge"
    elif dataset_name == "bil_gin_cc_homotopic" or dataset_name == "bil_gin":
        dataset_long_name = "BIL&GIN"
    elif dataset_name == "penthera":
        dataset_long_name = "Penthera"
    elif dataset_name == "hcp":
        dataset_long_name = "Human Connectome Project"
    else:
        raise ValueError("Unknown dataset name")

    return dataset_long_name


def plot_vertical_lines(xs, ax=None, **plot_kwargs):
    """Draw vertical lines on plot.

    Parameters
    ----------
    xs : A scalar, list, or 1D array
        Horizontal offsets.
    ax : the axis
        The axis. None to use gca.
    plot_kwargs :
        Keyword arguments to be passed to plot.

    Returns
    -------
        The plot object corresponding to the lines.
    """

    if ax is None:
        ax = plt.gca()

    xs = np.array((xs,) if np.isscalar(xs) else xs, copy=False)
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(
        np.array(y_lims + (np.nan,))[None, :], repeats=len(xs), axis=0
    ).flatten()
    # ToDo
    # plot_kwargs should be a list of dicts: one per line
    _ = ax.plot(x_points, y_points, scaley=False, **plot_kwargs)
    offset_x = (x_lims[-1] - x_lims[0]) / 50
    offset_y = (y_lims[-1] - y_lims[0]) / 10
    # ToDo
    # Should be a function to which arrays are passed: loops are especially
    # slow when plotting
    for x_point in x_points:
        ax.text(
            x_point + offset_x,
            y_lims[-1] - offset_y,
            "th={:.3f}".format(x_point),
            color=plt.gca().lines[-1].get_color(),
        )

    # return plot


def plot_horizontal_lines(ys, ax=None, **plot_kwargs):
    """Draw horizontal lines on plot.

    Parameters
    ----------
    ys : A scalar, list, or 1D array
        Vertical offsets.
    ax : the axis
        The axis. None to use gca.
    plot_kwargs :
        Keyword arguments to be passed to plot.

    Returns
    -------
        The plot object corresponding to the lines.
    """

    if ax is None:
        ax = plt.gca()
    ys = np.array((ys,) if np.isscalar(ys) else ys, copy=False)
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    y_points = np.repeat(ys[:, None], repeats=3, axis=1).flatten()
    x_points = np.repeat(
        np.array(x_lims + (np.nan,))[None, :], repeats=len(ys), axis=0
    ).flatten()
    # ToDo
    # plot_kwargs should be a list of dicts: one per line
    _ = ax.plot(x_points, y_points, scalex=False, **plot_kwargs)
    # ToDo
    # The dividing factors should be computed otherwise, since they also depend
    # on the dpi
    offset_x = (x_lims[-1] - x_lims[0]) / 6
    offset_y = (y_lims[-1] - y_lims[0]) / 120
    # ToDo
    # Should be a function to which arrays are passed: loops are especially
    # slow when plotting
    for y_point in y_points:
        ax.text(
            x_lims[-1] - offset_x,
            y_point + offset_y,
            "th={:.3f}".format(y_point),
            color=plt.gca().lines[-1].get_color(),
        )

    # return plot


def filter_streamlines_only(
    streamline_classes,
    latent_streamline_data,
    latent_atlas_all,
    y_latent_atlas_all,
    model,
    thresholds,
    reference,
    output_path,
    trk_ids,
    trk,
    num_neighbors=1,
    id_bundle=None,
):
    logger.info(
        "Filtering streamlines using the latent space nearest "
        "neighbor optimal distance criterion..."
    )

    encoder = PredictWrapper(model.encode)

    # Filter streamlines based on the latent space nearest neighbor distance
    logger.info(
        "Computing nearest neighbor distances in the latent space. \n"
        "May take several minutes with large tractogram..."
    )

    distance_informer = LatentSpaceDistanceInformer(
        encoder, latent_atlas_all, num_neighbors=num_neighbors
    )

    (
        distances,
        nearest_indices,
    ) = distance_informer.compute_distance_on_latent(latent_streamline_data)

    nearest_indices_class = y_latent_atlas_all[nearest_indices]

    if len(nearest_indices_class.shape) == 1:
        nearest_indices_class_final = nearest_indices_class
        distances_final = distances

    elif len(nearest_indices_class.shape) == 2:
        nearest_indices_class_final, _ = mode(nearest_indices_class, axis=1)
        pos_mode = nearest_indices_class == nearest_indices_class_final
        nearest_indices_class_final = nearest_indices_class_final.squeeze()
        distances_final = np.mean(distances, where=pos_mode, axis=1)

    else:
        raise ValueError(
            f"An error occured. Wrong neighbor format. Got {len(nearest_indices_class.shape)}"
        )

    logger.info("Filtering streamlines into plausibles/implausibles...")

    for y in tqdm(np.unique(nearest_indices_class_final)):
        k = [k for k, v in streamline_classes.items() if v == y][0]

        idx = np.argwhere(nearest_indices_class_final == y).squeeze()

        if np.isscalar(idx[()]):
            idx = idx[None]

        dist = distances_final[idx]

        indices_plausibles = np.argwhere(dist <= thresholds[k]).squeeze()

        if np.isscalar(indices_plausibles[()]):
            indices_plausibles = indices_plausibles[None]

        if len(indices_plausibles) == 0:
            continue

        if id_bundle is not None:
            name_bundle = f"{k}_{id_bundle}.trk"
        else:
            name_bundle = f"{k}.trk"

        data_per_streamlines = {"ids": trk_ids[idx[indices_plausibles]]}

        streamline_slices = trk_ids[idx[indices_plausibles]].squeeze()

        if np.isscalar(streamline_slices[()]):
            streamline_slices = streamline_slices[None]

        save_streamlines(
            trk.streamlines[streamline_slices],
            reference,
            pjoin(output_path, name_bundle),
            data_per_streamline=data_per_streamlines,
        )

    logger.info("Finished filtering streamlines into plausibles/implausibles.")


def plot_latent_space(
    latent_samples, classes, latent_space_dims, fname_root, rbx_classes=True
):
    logger.info("Plotting latent space...")

    dpi = 300
    figure_rc = {"figure": {"dpi": dpi}}

    rc_parameters = generate_decoration_rc_parameters()
    rc_parameters.update(figure_rc)

    # Use the UMAP dimensionality reduction
    logger.info("Computing UMAP dimensionality reduction...")

    method = "umap"
    filename = (
        fname_root
        + LoggerKeys.underscore.value
        + method
        + LoggerKeys.fname_extension_sep.value
        + LoggerKeys.fname_extension_sep.plot_extension.value
    )
    reducer = umap.UMAP(random_state=42)
    umap_results = reducer.fit_transform(latent_samples)

    logger.info("Finished computing UMAP dimensionality reduction.")

    logger.debug("Generating UMAP latent space plot...")

    if rbx_classes:
        # TODO avoid hard coding the classes?
        non_right_classes = {
            "AC": 1,
            "AF_L": 2,
            "CC_Fr_1": 4,
            "CC_Fr_2": 5,
            "CC_Oc": 6,
            "CC_Pa": 7,
            "CC_Pr_Po": 8,
            "CC_Te": 9,
            "CG_L": 10,
            "FAT_L": 12,
            "FPT_L": 14,
            "FX_L": 16,
            "ICP_L": 18,
            "IFOF_L": 20,
            "ILF_L": 22,
            "MCP": 24,
            "MdLF_L": 25,
            "OR_ML_L": 27,
            "PC": 29,
            "POPT_L": 30,
            "PYT_L": 32,
            "SCP_L": 34,
            "SLF_L": 36,
            "UF_L": 38,
        }
        selected_classes = list(non_right_classes.items())[
            :20
        ]  # Select the first 20 non-right classes, we have 20 colors
    else:
        unique, counts = np.unique(classes, return_counts=True)
        zip_unique_count = list(zip(unique, counts))
        res = sorted(zip_unique_count, key=lambda x: x[1], reverse=True)
        selected_classes = [(str(c), c) for c, count in res[:20]]

    # TODO combine all CC classes in plot

    fig = plt.figure(figsize=(12, 10))

    # plot all plausible with black dots
    mask = np.array(classes) != 0
    plt.scatter(umap_results[mask, 0], umap_results[mask, 1], color=(0, 0, 0))

    # plot implausible as black empty circles
    mask = np.array(classes) == 0
    m = MarkerStyle(marker="o", fillstyle="none")
    plt.scatter(
        umap_results[mask, 0], umap_results[mask, 1], color=(0, 0, 0), marker=m
    )

    # plot selected classes with colors
    colors = list(plt.cm.tab20(np.arange(20)))
    plt.gca().set_prop_cycle("color", colors)
    for class_name, class_idx in selected_classes:
        mask = np.array(classes) == class_idx
        plt.scatter(
            umap_results[mask, 0], umap_results[mask, 1], label=class_name
        )

    plt.legend()
    plt.title("Latent space UMAP (latent dim={})".format(latent_space_dims))
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    logger.debug(
        "Finished generating UMAP latent space plot to:\n{}".format(filename)
    )

    logger.info("Finished plotting latent space.")

    return filename


def find_tractogram_filtering_threshold_v2(
    y_latent_atlas_brain_plaus,
    current_class,
    latent_space_dims,
    bundles_classes_dict,
    fname_root,
    dataset_name,
    y_thres_plaus,
    y_thres_implaus,
    distances_plaus,
    distances_implaus,
    nearest_indices_plaus,
    nearest_indices_implaus,
    roc_optimal_point=ROCSalientPoint.INV_DIAGONAL_INTERSECT,
):

    logger.info("Finding filtering threshold...")

    filtering_time_probe = dict()
    filtering_time_probe.update({"encoding": []})
    filtering_time_probe.update({"latent_distances": []})
    filtering_time_probe.update({"filtering_th": []})
    filtering_time_probe.update({"filter": []})

    nearest_indices_implaus_class = y_latent_atlas_brain_plaus[
        nearest_indices_implaus
    ]
    nearest_indices_plaus_class = y_latent_atlas_brain_plaus[
        nearest_indices_plaus
    ]
    indices_plausible_class = np.argwhere(
        nearest_indices_plaus_class == 1000 + current_class
    ).squeeze()
    indices_implausible_class = np.argwhere(
        nearest_indices_implaus_class == 1000 + current_class
    ).squeeze()
    distances_plaus_roc = distances_plaus[indices_plausible_class]
    distances_implaus_roc = distances_implaus[indices_implausible_class]

    y_thres_plaus_roc = y_thres_plaus[indices_plausible_class]
    y_thres_implaus_roc = y_thres_implaus[indices_implausible_class]

    roc_dict = roc_curve_computation(
        (y_thres_plaus_roc >= 0).astype(np.float32)
        * bundles_classes_dict["plausible"],
        (y_thres_implaus_roc >= 0).astype(np.float32)
        * bundles_classes_dict["implausible"],
        distances_plaus_roc,
        distances_implaus_roc,
        fname_root,
        filtering_time_probe,
        dataset_name,
    )

    # Plot the features for the validation streamlines
    plot_filtering_threshold_features(
        roc_dict,
        distances_plaus_roc,
        distances_implaus_roc,
        (y_thres_plaus_roc >= 0).astype(np.float32)
        * bundles_classes_dict["plausible"],
        (y_thres_implaus_roc >= 0).astype(np.float32)
        * bundles_classes_dict["implausible"],
        latent_space_dims,
        bundles_classes_dict,
        fname_root,
        dataset_name,
    )

    if roc_optimal_point == ROCSalientPoint.INV_DIAGONAL_INTERSECT:
        threshold = roc_dict["optimal_intersect_th"]
    else:
        raise ValueError(
            "Unknown ROC optimal point type.\n"
            "Found: {}; Available: {}".format(
                roc_optimal_point, ROCSalientPoint._member_names_
            )
        )

    logger.info("Finished finding filtering threshold.")

    return threshold


def roc_curve_computation(
    y_plaus_track_classes,
    y_implaus_track_classes,
    plaus_streamlines_latent_distances,
    implaus_streamlines_latent_distances,
    fname_root,
    filtering_time_probe,
    dataset_name,
):

    dpi = 300
    figure_rc = {"figure": {"dpi": dpi}}

    rc_parameters = generate_decoration_rc_parameters()
    rc_parameters.update(figure_rc)
    # Plot the ROC curve

    set_label = _get_set_label(fname_root)

    file_rootname, ext = LatentSpaceKeys.roc_data_file_basename.value.split(
        LoggerKeys.fname_extension_sep.value
    )
    filename = (
        set_label
        + LoggerKeys.underscore.value
        + file_rootname
        + LoggerKeys.fname_extension_sep.value
        + ext
    )
    roc_data_fname = pjoin(os.path.dirname(fname_root), filename)

    start = time()

    roc_dict = compute_filtering_roc_curve(
        y_plaus_track_classes,
        y_implaus_track_classes,
        plaus_streamlines_latent_distances,
        implaus_streamlines_latent_distances,
        roc_data_fname,
    )

    elapsed = time() - start
    filtering_time_probe["filtering_th"] = elapsed

    roc_plot_fname = (
        fname_root
        + LoggerKeys.underscore.value
        + LatentSpaceKeys.latent_space_roc_fname_label.value
        + LoggerKeys.fname_extension_sep.value
        + LoggerKeys.fname_extension_sep.plot_extension.value
    )

    plot_roc(
        roc_dict["fpr"],
        roc_dict["tpr"],
        roc_dict["thresholds"],
        roc_dict["auc"],
        roc_plot_fname,
        dataset_name,
        roc_dict["optimal_diff_idx"],
        roc_dict["optimal_intersect_idx"],
        roc_dict["optimal_max_acc_idx"],
        **rc_parameters,
    )

    return roc_dict


def plot_filtering_threshold_features(
    roc_dict,
    plaus_streamlines_latent_distances,
    implaus_streamlines_latent_distances,
    y_plaus_track_classes,
    y_implaus_track_classes,
    latent_space_dims,
    bundles_classes_dict,
    fname_root,
    dataset_name,
    roc_optimal_point=ROCSalientPoint.INV_DIAGONAL_INTERSECT,
):

    logger.info("Plotting filtering threshold features.")

    dpi = 300
    figure_rc = {"figure": {"dpi": dpi}}

    rc_parameters = generate_decoration_rc_parameters()
    rc_parameters.update(figure_rc)

    class_names = list(bundles_classes_dict.keys())

    # Plot latent space distance histogram and stats
    track_classes = np.hstack([y_plaus_track_classes, y_implaus_track_classes])

    plot_latent_space_distance_features(
        plaus_streamlines_latent_distances,
        implaus_streamlines_latent_distances,
        track_classes,
        class_names,
        latent_space_dims,
        fname_root,
        dataset_name,
        bundles_classes_dict,
        **rc_parameters,
    )

    if roc_optimal_point == ROCSalientPoint.INV_DIAGONAL_INTERSECT:
        latent_space_distance_threshold = roc_dict["optimal_intersect_th"]
    elif roc_optimal_point == ROCSalientPoint.MAX_ACCURACY:
        latent_space_distance_threshold = roc_dict["optimal_max_acc_th"]
    else:
        raise ValueError(
            "Unknown ROC optimal point type.\n"
            "Found: {}; Available: {}".format(
                roc_optimal_point, ROCSalientPoint._member_names_
            )
        )

    logger.info(
        "Using filtering threshold computed according to ROC optimal "
        "point: {}.".format(roc_optimal_point)
    )

    fname_root += (
        LoggerKeys.underscore.value
        + LatentSpaceKeys.threshold_fname_label.value
    )
    plot_latent_space_distance_features(
        plaus_streamlines_latent_distances,
        implaus_streamlines_latent_distances,
        track_classes,
        class_names,
        latent_space_dims,
        fname_root,
        dataset_name,
        bundles_classes_dict,
        threshold=latent_space_distance_threshold,
        **rc_parameters,
    )

    logger.info("Finished plotting filtering threshold features.")


# ToDo
# In other scripts the very same split indicates the subj_id so a consistent
# naming convention is necessary.
def _get_set_label(fname):

    return os.path.basename(fname).split(LoggerKeys.underscore.value)[0]


def compute_filtering_roc_curve(
    y_plaus_track_classes,
    y_implaus_track_classes,
    distances_plaus,
    distances_implaus,
    roc_data_fname,
):

    # ToDo
    # Do this only once, otherwise it is error-prone: we may stack the
    # implausibles first inadvertently, or have this and the classes be
    # stacked in opposite orders
    # find_tractogram_filtering_threshold
    distances_orig = np.hstack([distances_plaus, distances_implaus])
    y_track_classes = np.hstack(
        [y_plaus_track_classes, y_implaus_track_classes]
    )

    # The computation of the optimal threshold based on the computation of the
    # accuracy at each point of the ROC curve is computationally expensive.
    # Hence, if the number of points is larger than ROC_MAX_NUM_POINTS, get a
    # ROC_MAX_NUM_POINTS number of equally-spaced points. This may, however,
    # not be optimal since at locations where the ROC curve shows a bending a
    # higher resolution would be desirable.
    num_samples = len(distances_orig)
    if num_samples > ROC_MAX_NUM_POINTS:
        idx = np.linspace(
            0, num_samples, num=ROC_MAX_NUM_POINTS, endpoint=False, dtype=int
        )
        distances = distances_orig[idx]
        y_true = y_track_classes[idx]
    else:
        distances = np.copy(distances_orig)
        y_true = np.copy(y_track_classes)

    # Plot the ROC curve
    feature_range = (0, 1)
    scaler = MinMaxScaler(feature_range=feature_range)

    y_true[y_true < BundlesAdditionalLabels.invalid_connection_class.value] = 1
    y_true[
        y_true == BundlesAdditionalLabels.invalid_connection_class.value
    ] = 0
    y_score = np.copy(distances).reshape(-1, 1)

    # Transform ranges
    scaler.fit(y_score)
    y_score = scaler.transform(y_score)

    # Invert the score: a low distance indicates the positive label
    y_score = 1.0 - y_score
    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
        pos_label=None,
        sample_weight=None,
        drop_intermediate=False,
    )
    # In order to get accurate results, especially for degenerated cases (e.g.
    # the ROC curve computation has dropped a significant number of points),
    # we need to interpolate the values.
    if len(thresholds) < ROC_MIN_NUM_POINTS:
        fpr_interp, tpr_interp = upsample_coords(
            [fpr, tpr], ROC_MIN_NUM_POINTS
        )
        # Keeping the first upsampled FPR coordinates
        # ToDo
        # Investigate the differences between the upsampled horizontal
        # coordinates
        _, thresholds_interp = upsample_coords(
            [fpr, thresholds], ROC_MIN_NUM_POINTS
        )
        fpr = fpr_interp
        tpr = tpr_interp
        thresholds = thresholds_interp

    auc = roc_auc_score(y_true, y_score)

    optimal_diff_idx = compute_optimal_roc_difference_index(tpr, fpr)
    optimal_diff_th = thresholds[optimal_diff_idx]
    optimal_diff_tpr = tpr[optimal_diff_idx]
    optimal_diff_fpr = fpr[optimal_diff_idx]

    optimal_intersect_idx = compute_optimal_roc_intersection_index(tpr, fpr)

    # Get the optimal threshold from the intersection of the inverted diagonal
    # and the ROC curve by undoing the scoring inversion
    optimal_intersect_th = scaler.inverse_transform(
        (1.0 - thresholds)[optimal_intersect_idx].reshape(1, -1)
    )[0][0]
    optimal_intersect_tpr = tpr[optimal_intersect_idx]
    optimal_intersect_fpr = fpr[optimal_intersect_idx]

    thr_original_scale = scaler.inverse_transform(
        1.0 - thresholds.reshape(-1, 1)
    ).squeeze()

    optimal_max_acc_idx = compute_maximum_accuracy_roc_threshold_index(
        thr_original_scale, y_true, distances
    )

    optimal_max_acc_th = scaler.inverse_transform(
        (1.0 - thresholds)[optimal_max_acc_idx].reshape(1, -1)
    )[0][0]
    optimal_max_acc_tpr = tpr[optimal_max_acc_idx]
    optimal_max_acc_fpr = fpr[optimal_max_acc_idx]

    roc_dict = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc,
        "optimal_diff_idx": optimal_diff_idx,
        "optimal_diff_th": optimal_diff_th,
        "optimal_diff_tpr": optimal_diff_tpr,
        "optimal_diff_fpr": optimal_diff_fpr,
        "optimal_intersect_idx": optimal_intersect_idx,
        "optimal_intersect_th": optimal_intersect_th,
        "optimal_intersect_tpr": optimal_intersect_tpr,
        "optimal_intersect_fpr": optimal_intersect_fpr,
        "optimal_max_acc_idx": optimal_max_acc_idx,
        "optimal_max_acc_th": optimal_max_acc_th,
        "optimal_max_acc_tpr": optimal_max_acc_tpr,
        "optimal_max_acc_fpr": optimal_max_acc_fpr,
    }

    save_data_to_pickle_file(roc_dict, roc_data_fname)

    return roc_dict


def plot_roc(
    fpr,
    tpr,
    thresholds,
    auc,
    filename,
    dataset_name,
    optimal_diff_idx=None,
    optimal_intersect_idx=None,
    optimal_max_acc_idx=None,
    **rc_parameters,
):

    logger.info("Plotting ROC curve...")

    # Set the plot rc parameters before creating the figure
    for key, val in rc_parameters.items():
        plt.rc(key, **val)

    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(111)

    delta_x = 0.04
    delta_y = 0.01
    if optimal_diff_idx is not None:
        ax.plot(
            fpr,
            tpr,
            label=r"AUC = %0.2f" % auc,
            marker="s",
            markevery=[optimal_diff_idx],
        )
        for idx in [optimal_diff_idx]:
            ax.annotate(
                "%0.2f, %0.2f" % (fpr[idx], tpr[idx]),
                (fpr[idx] - delta_x, tpr[idx] + delta_y),
                color=ax.get_lines()[0].get_c(),
            )
    else:
        ax.plot(fpr, tpr, label=r"AUC = %0.2f" % auc)

    if optimal_intersect_idx is not None:
        delta_x = 0.02
        num_points = len(fpr)
        diagonal_y = np.linspace(1, 0, num_points)
        diagonal_x = np.linspace(0, 1, num_points)
        fpr_optimal_intersect = fpr[optimal_intersect_idx]
        tpr_optimal_intersect = tpr[optimal_intersect_idx]
        ax.plot(
            diagonal_x,
            diagonal_y,
            transform=ax.transAxes,
            linestyle="dashed",
            color="r",
        )
        ax.plot(
            [fpr_optimal_intersect],
            [tpr_optimal_intersect],
            marker="s",
            color="r",
        )
        ax.annotate(
            "%0.2f, %0.2f" % (fpr_optimal_intersect, tpr_optimal_intersect),
            (fpr_optimal_intersect + delta_x, tpr_optimal_intersect - delta_y),
            color="r",
        )

    if optimal_max_acc_idx is not None:
        delta_x = 0.008
        delta_y = 0.04
        fpr_optimal_max_acc = fpr[optimal_max_acc_idx]
        tpr_optimal_max_acc = tpr[optimal_max_acc_idx]
        ax.plot(
            [fpr_optimal_max_acc],
            [tpr_optimal_max_acc],
            marker="s",
            color="C1",
        )
        ax.annotate(
            "%0.2f, %0.2f" % (fpr_optimal_max_acc, tpr_optimal_max_acc),
            (fpr_optimal_max_acc + delta_x, tpr_optimal_max_acc - delta_y),
            color="C1",
        )

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    # plt.title('ROC curve')
    long_dataset_name = get_dataset_long_name_from_dataset_name(dataset_name)
    plt.title("ROC curve\n(dataset: {})".format(long_dataset_name))
    fig.savefig(filename)
    plt.close(fig)

    logger.info("Finished plotting ROC curve to:\n{}".format(filename))


def plot_latent_space_distance_features(
    distances_plaus,
    distances_implaus,
    track_classes,
    class_names,
    latent_space_dims,
    fname_root,
    dataset_name,
    bundles_classes_dict,
    threshold=None,
    **rc_parameters,
):

    distances = np.hstack([distances_plaus, distances_implaus])

    plot_latent_space_histogram(
        distances,
        track_classes,
        class_names,
        latent_space_dims,
        fname_root,
        dataset_name,
        bundles_classes_dict,
        threshold=threshold,
        **rc_parameters,
    )

    plot_latent_space_stats(
        distances,
        track_classes,
        class_names,
        latent_space_dims,
        fname_root,
        dataset_name,
        bundles_classes_dict,
        threshold=threshold,
        **rc_parameters,
    )


def compute_maximum_accuracy_roc_threshold_index(
    thresholds, y_true, distances
):

    logger.info("Computing maximum accuracy ROC curve threshold index...")

    thr_accuracy = dict()
    for thr in thresholds:

        # Predict/classify the streamlines
        indices_implausibles = np.argwhere(distances > thr).squeeze()

        y_pred = np.ones_like(y_true)
        y_pred[indices_implausibles] = 0

        classif_report = classification_report(
            y_true, y_pred, output_dict=True
        )
        thr_accuracy[thr] = classif_report["accuracy"]

    # Keep the point that achieves the best accuracy
    acc_values = np.array(list(thr_accuracy.values()))
    max_acc_value = np.max(acc_values)
    max_acc_thr = list(thr_accuracy.keys())[
        list(thr_accuracy.values()).index(max_acc_value)
    ]
    max_accuracy_idx = np.argwhere(thresholds == max_acc_thr).ravel()

    logger.info(
        "Finished computing maximum accuracy ROC curve threshold " "index."
    )

    return max_accuracy_idx


def compute_optimal_roc_difference_index(tpr, fpr):

    logger.info("Computing optimal ROC difference indices...")

    optimal_diff_idx = np.argmax(tpr - fpr)

    logger.info(
        "Finished computing optimal ROC difference index. Found: {}".format(
            optimal_diff_idx.size
        )
    )

    return optimal_diff_idx


def compute_optimal_roc_intersection_index(tpr, fpr):

    logger.info(
        "Computing optimal ROC-inverted diagonal intersection " "indices..."
    )

    num_points = len(fpr)
    diagonal_y = np.linspace(1, 0, num_points)
    diagonal_x = np.linspace(0, 1, num_points)
    interp_y = interp(fpr, diagonal_x, diagonal_y)

    # tpr_coords = np.column_stack([fpr, tpr])
    # diagonal_coords = np.column_stack([fpr, interp_y])
    # intersect_idx = find_intersection(tpr_coords, diagonal_coords)

    intersect_idx = np.argwhere(np.diff(np.sign(tpr - interp_y)) != 0).reshape(
        -1
    )

    length = intersect_idx.size

    logger.info(
        "Finished computing optimal ROC-inverted diagonal "
        "intersection indices. Found: {}".format(length)
    )

    # Keep only the first point if multiple are found. Finding multiple points
    # is just a consequence of the discrete nature of the curves and the
    # tolerance.
    # ToDo
    # To be correct the middle index along each tpr and frp axes should be
    # computed to account for the cases where the diagonal intersects a
    # straight piece of the ROC curve
    if length > 1:
        middle_index = length // 2 if length % 2 else length // 2 - 1
        intersect_idx = intersect_idx[middle_index]

        logger.info("Kept the middle index only.")

    return intersect_idx


def plot_latent_space_histogram(
    distances,
    classes,
    class_names,
    latent_space_dims,
    fname_root,
    dataset_name,
    bundles_classes_dict,
    threshold=None,
    **rc_parameters,
):

    logger.info("Plotting latent space histogram...")

    # ToDo
    # The matplotlib 2.2.* version required by `scilpy` does not have the
    # `title_fontsize` key, so remove it until a more recent version is allowed
    if "legend" in rc_parameters.keys():
        rc_parameters["legend"].pop("title_fontsize", None)

    # Set the plot rc parameters before creating the figure
    for key, val in rc_parameters.items():
        plt.rc(key, **val)

    filename = (
        fname_root
        + LoggerKeys.underscore.value
        + LatentSpaceKeys.latent_space_histogram_fname_label.value
        + LoggerKeys.fname_extension_sep.value
        + LoggerKeys.fname_extension_sep.plot_extension.value
    )

    # Get colormap from colormap name
    cmap = plt.cm.get_cmap("rainbow")
    unique_classes = np.unique(classes)
    unique_classes_ord = []
    for c in class_names:
        unique_classes_ord.append(bundles_classes_dict[c])
    custom_markers, colors = _generate_plot_decorators(
        unique_classes_ord, class_names, cmap
    )

    # Separate the distances according to the bundle class to boxplot each
    # bundle's distances stats in a separate boxplot
    classes_distances = []
    for unique_class in unique_classes:
        indices = np.where(classes == unique_class)
        class_distances = distances[indices]
        classes_distances.append(class_distances)

    fig = plt.figure(figsize=(12, 10))
    _ = plt.subplot(111)
    num_bins = 100  # len(distances)
    n, bins, patches = plt.hist(
        classes_distances, bins=num_bins, density=False
    )

    if len(unique_classes) == 1:
        for i, p in enumerate(patches):
            plt.setp(p, "facecolor", colors[0])
    else:
        for i, p in enumerate(patches):
            plt.setp(p, "facecolor", colors[i])

    plt.legend(custom_markers, class_names, loc="upper right")
    plt.xlabel("distance")
    plt.ylabel("frequency")
    # plt.title('Nearest neighbor latent space distance histogram '
    #           '(Latent dims={})'.format(latent_space_dims))
    # plt.title('Latent space distance histogram')
    long_dataset_name = get_dataset_long_name_from_dataset_name(dataset_name)
    plt.title(
        "Latent space distance histogram\n(dataset: {})".format(
            long_dataset_name
        )
    )
    # Make y-ticks be integers
    yint = []
    locs, labels = plt.yticks()
    for each in locs:
        yint.append(int(each))
    plt.yticks(yint)

    if threshold is not None:
        plot_kwargs = {"color": "C0", "linestyle": "--"}
        plot_vertical_lines(threshold, **plot_kwargs)

    # Use log scale
    plt.yscale("log")

    fig.savefig(filename)
    plt.close(fig)

    logger.info(
        "Finished plotting latent space histogram to:\n{}".format(filename)
    )


def plot_latent_space_stats(
    distances,
    classes,
    class_names,
    latent_space_dims,
    fname_root,
    dataset_name,
    bundles_classes_dict,
    threshold=None,
    **rc_parameters,
):

    logger.info("Plotting latent spaces stats...")

    # Set the plot rc parameters before creating the figure
    for key, val in rc_parameters.items():
        plt.rc(key, **val)

    filename = (
        fname_root
        + LoggerKeys.underscore.value
        + LatentSpaceKeys.latent_space_stats_fname_label.value
        + LoggerKeys.fname_extension_sep.value
        + LoggerKeys.fname_extension_sep.plot_extension.value
    )

    # Get colormap from colormap name
    cmap = plt.cm.get_cmap("rainbow")
    unique_classes = np.unique(classes)
    unique_classes_ord = []
    for c in class_names:
        unique_classes_ord.append(bundles_classes_dict[c])
    _, colors = _generate_plot_decorators(
        unique_classes_ord, class_names, cmap
    )

    # Separate the distances according to the bundle class to boxplot each
    # bundle's distances stats in a separate boxplot
    classes_distances = []
    for unique_class in unique_classes:
        indices = np.where(classes == unique_class)
        class_distances = distances[indices]
        classes_distances.append(class_distances)

    classes_distances = [
        classes_distances[unique_classes_ord.index(c)]
        for c in unique_classes_ord
    ]

    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(111)
    distances_bplot = ax.boxplot(
        classes_distances,
        notch=True,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        labels=class_names,
    )
    ax.set_xlabel("class")  # 'bundle')
    ax.set_ylabel("distance")
    # plt.xticks(rotation=45, ha='right')
    # plt.title('Latent space distance stats (Latent dims={})'.format(
    #    latent_space_dims))
    # plt.title('Latent space distance statistics')
    long_dataset_name = get_dataset_long_name_from_dataset_name(dataset_name)
    plt.title(
        "Latent space distance statistics\n(dataset: {})".format(
            long_dataset_name
        )
    )

    for box, color in zip(distances_bplot["boxes"], colors):
        box.set_facecolor(color)

    if threshold is not None:
        plot_kwargs = {"color": "C0", "linestyle": "--"}
        plot_horizontal_lines(threshold, **plot_kwargs)

    # ToDo
    # This should be a parameter of the method
    # Other options include changing the width of the boxes (which may be more
    # complicated to deal with in grouped boxplots); setting manually the
    # positions argument to the boxplot constructor; or setting the limits and
    # ticks of the x axis (ax.set_xlim(-1.5,2.5))
    # Also, the aspect ratio should be computed according to the min/max values
    # across the groups since a given ration may not work for other values
    # Make the boxes be closer
    # aspect_ratio = 0.005
    # ax.set_aspect(aspect_ratio, share=True)

    # Alternatively, we use log scale
    plt.yscale("log")

    fig.savefig(filename)
    plt.close(fig)

    logger.info(
        "Finished plotting latent spaces stats to:\n{}".format(filename)
    )


def _generate_plot_decorators(
    classes,
    class_names,
    cmap,
    foreign_class_color=None,
    markeredgecolor=None,
    markeredgewidth=None,
):

    indexes = np.unique(classes, return_index=True)[1]
    unique_classes = [classes[index] for index in sorted(indexes)]

    # ToDo
    # Improve all this internal logic
    if foreign_class_color is None:
        num_classes = len(class_names)
    else:
        num_classes = len(class_names) - len(foreign_class_color.keys())
        unique_classes = [
            bundle_class
            for bundle_class in unique_classes
            if bundle_class not in foreign_class_color.keys()
        ]

    colormap_points = np.linspace(0, 1, num=num_classes)

    point_color_lut = dict(
        {
            point_class: colormap_point
            for point_class, colormap_point in zip(
                unique_classes, colormap_points
            )
        }
    )

    # Create markers for legend
    custom_markers = []
    for point_class in unique_classes:
        marker = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor=markeredgecolor,
            markeredgewidth=markeredgewidth,
            markerfacecolor=cmap(point_color_lut[point_class]),
            markersize=15,
        )
        custom_markers.append(marker)

    # ToDo
    # Improve all this internal logic and allow for custom markers for the
    # foreign class
    if foreign_class_color is not None:
        for elem in foreign_class_color.keys():
            marker = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=foreign_class_color[elem],
                markersize=15,
            )
            custom_markers.append(marker)

    # Create the colors for each point
    colors = []
    for elem in classes:
        # ToDo
        # Improve this in case we have multiple foreign classes or to ensure
        # that there is only one
        if (
            foreign_class_color is None
            or elem not in foreign_class_color.keys()
        ):
            color = cmap(point_color_lut[elem])
        else:
            color = foreign_class_color[elem]
        colors.append(color)

    return custom_markers, colors
