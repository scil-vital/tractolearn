# -*- coding: utf-8 -*-

import logging
from os.path import join as pjoin

import numpy as np
from scipy.stats import mode
from tqdm import tqdm
from tractolearn.utils.layer_utils import PredictWrapper

from tractolearn.filtering.latent_space_distance_informer import (
    LatentSpaceDistanceInformer,
)
from tractolearn.tractoio.utils import (
    save_streamlines,
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
