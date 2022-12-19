# -*- coding: utf-8 -*-

import logging
from os.path import join as pjoin

import numpy as np
import umap
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
from scipy.stats import mode
from tqdm import tqdm

from tractolearn.Logger import LoggerKeys
from tractolearn.utils.layer_utils import PredictWrapper

from tractolearn.filtering.latent_space_distance_informer import (
    LatentSpaceDistanceInformer,
)
from tractolearn.tractoio.utils import (
    save_streamlines,
)

from tractolearn.visualization.plot_utils import (
    generate_decoration_rc_parameters,
)

# fixme: FD
logger = logging.getLogger("root")


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
    plt.scatter(umap_results[mask, 0], umap_results[mask, 1], color=(0, 0, 0), marker=m)

    # plot selected classes with colors
    colors = list(plt.cm.tab20(np.arange(20)))
    plt.gca().set_prop_cycle("color", colors)
    for class_name, class_idx in selected_classes:
        mask = np.array(classes) == class_idx
        plt.scatter(umap_results[mask, 0], umap_results[mask, 1], label=class_name)

    plt.legend()
    plt.title("Latent space UMAP (latent dim={})".format(latent_space_dims))
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Finished generating UMAP latent space plot to:\n{}".format(filename))

    logger.info("Finished plotting latent space.")

    return filename
