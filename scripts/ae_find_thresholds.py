#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for the inference of FINTA
"""

import argparse
import logging
import os
import shutil
from os.path import join as pjoin
from typing import Tuple, List

import numpy as np
import torch
import yaml
from scilpy.io.utils import add_verbose_arg, add_overwrite_arg
from tqdm import tqdm

from tractolearn.Logger import LoggerKeys, _set_up_logger
from tractolearn.config.Experiment import ThresholdTestKeys
from tractolearn.filtering.latent_space_distance_informer import (
    LatentSpaceDistanceInformer,
)
from tractolearn.filtering.latent_space_featuring import (
    ROCSalientPoint,
    plot_latent_space,
    find_tractogram_filtering_threshold_v2,
)
from tractolearn.learning.Datasets import OnTheFlyDataset
from tractolearn.models.autoencoding_utils import encode_data
from tractolearn.models.model_pool import get_model
from tractolearn.tractoio.utils import (
    load_streamlines,
    read_data_from_json_file,
    save_data_to_json_file,
)
from tractolearn.utils.layer_utils import PredictWrapper
from tractolearn.utils.utils import make_run_dir

torch.set_flush_denormal(True)
logger = logging.getLogger("root")


def sort_streamline_per_bundle(X: np.array, y: np.array) -> Tuple[np.array, np.array]:
    indices = y.argsort()
    y_sorted = y[indices]
    X_sorted = X[indices]

    return X_sorted, y_sorted


def separate_pl_impl(streamlines_dict: dict, key_list: List):
    x_plaus = np.empty((0, 256, 3))
    x_implaus = np.empty((0, 256, 3))
    y_plaus = np.empty((0,))
    y_implaus = np.empty((0,))
    for k, v in streamlines_dict.items():
        if k not in key_list:
            x_implaus = np.vstack((x_implaus, v[0]))
            y_implaus = np.hstack((y_implaus, v[1]))
        else:
            x_plaus = np.vstack((x_plaus, v[0]))
            y_plaus = np.hstack((y_plaus, v[1]))

    return x_plaus, x_implaus, y_plaus, y_implaus


def separate_pl_impl_latent(latent_dict: dict, key_list: List, latent_dims: int = 32):
    x_plaus = np.empty((0, latent_dims))
    x_implaus = np.empty((0, latent_dims))
    y_plaus = np.empty((0,))
    y_implaus = np.empty((0,))
    for k, v in latent_dict.items():
        if k not in key_list:
            x_implaus = np.vstack((x_implaus, v[0]))
            y_implaus = np.hstack((y_implaus, v[1]))
        else:
            x_plaus = np.vstack((x_plaus, v[0]))
            y_plaus = np.hstack((y_plaus, v[1]))

    return x_plaus, x_implaus, y_plaus, y_implaus


def set_threshold(
    config,
    streamlines_dict,
    latent_atlas_dict,
    latent_dict,
    streamline_classes,
    model,
    experiment_dir,
):
    logger.info("Setting threshold ...")

    latent_atlas_all = np.empty((0, 32))
    y_latent_atlas_all = np.empty((0,))

    for _, (latent_a, y_latent_a) in latent_atlas_dict.items():
        latent_atlas_all = np.vstack((latent_atlas_all, latent_a))
        y_latent_atlas_all = np.hstack((y_latent_atlas_all, y_latent_a))

    latent_all = np.empty((0, 32))
    y_latent_all = np.empty((0,))

    for _, (latent_test, y_latent_test) in latent_dict.items():
        latent_all = np.vstack((latent_all, latent_test))
        y_latent_all = np.hstack((y_latent_all, y_latent_test))

    X_all = np.empty((0, 256, 3))
    y_all = np.empty((0,))

    for _, (X_test, y_test) in streamlines_dict.items():
        X_all = np.vstack((X_all, X_test))
        y_all = np.hstack((y_all, y_test))

    assert np.all(y_all == y_latent_all)

    encoder = PredictWrapper(model.encode)

    # Filter streamlines based on the latent space nearest neighbor distance
    logger.info("Filtering streamlines into plausibles/implausibles...")

    distance_informer = LatentSpaceDistanceInformer(
        encoder, latent_atlas_all, num_neighbors=1
    )

    (
        distances,
        nearest_indices,
    ) = distance_informer.compute_distance_on_latent(latent_all)

    threshold_dict = {}
    for k, latent_space_distance_threshold in tqdm(streamlines_dict.items()):

        if k == "implausible":
            continue

        logger.info(f"Finding thresholds for: {k}")

        os.makedirs(pjoin(experiment_dir, k))

        indices_plausible = np.argwhere(y_all == streamline_classes[k]).squeeze()
        indices_implausible = np.argwhere(y_all != streamline_classes[k]).squeeze()
        y_thres_plaus = y_all[indices_plausible]
        y_thres_implaus = y_all[indices_implausible]
        plaus_streamlines_latent_distances = distances[indices_plausible]
        implaus_streamlines_latent_distances = distances[indices_implausible]

        nearest_indices_plaus = nearest_indices[indices_plausible]
        nearest_indices_implaus = nearest_indices[indices_implausible]

        try:
            threshold = find_tractogram_filtering_threshold_v2(
                y_latent_atlas_all,
                streamline_classes[k],
                config[ThresholdTestKeys.LATENT_DIMS.value],
                {"plausible": 0, "implausible": 100},
                pjoin(experiment_dir, k, "thres"),
                "hcp",
                y_thres_plaus,
                y_thres_implaus,
                plaus_streamlines_latent_distances,
                implaus_streamlines_latent_distances,
                nearest_indices_plaus,
                nearest_indices_implaus,
                roc_optimal_point=ROCSalientPoint.INV_DIAGONAL_INTERSECT,
            )
        except Exception as e:
            logger.info(f"Catched exception: {e}")
            logger.info(f"Bundle: {k}")
            logger.info(f"Setting threshold to 0")
            threshold = 0

        threshold_dict[k] = threshold

    return threshold_dict


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "config_file",
        help="Configuration YAML file",
    )

    add_overwrite_arg(parser)
    add_verbose_arg(parser)

    return parser.parse_args()


def main():
    args = _build_arg_parser()

    with open(args.config_file) as f:
        config = yaml.safe_load(f.read())

    experiment_dir = make_run_dir(out_path=config[ThresholdTestKeys.OUTPUT.value])
    shutil.copy(args.config_file, experiment_dir)
    device = config[ThresholdTestKeys.DEVICE.value]

    logger_fname = pjoin(experiment_dir, LoggerKeys.logger_file_basename.name)
    _set_up_logger(logger_fname)

    if args.verbose:
        logging.basicConfig(level=logging.INFO if args.verbose == 1 else logging.DEBUG)

    logger.info(" +++ Loading generic files +++")

    valid_bundles_files = os.listdir(config[ThresholdTestKeys.VALID_BUNDLE_PATH.value])
    atlas_files = os.listdir(config[ThresholdTestKeys.ATLAS_PATH.value])

    checkpoint = torch.load(
        config[ThresholdTestKeys.MODEL.value],
        map_location=device,
    )
    state_dict = checkpoint["state_dict"]
    model = get_model(
        config[ThresholdTestKeys.MODEL_TYPE.value],
        config[ThresholdTestKeys.LATENT_DIMS.value],
        device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    implausible_file = pjoin(config[ThresholdTestKeys.INVALID_BUNDLE_FILE.value])
    logger.info(
        f" +++ Loading Threshold Implausible Streamlines: {implausible_file} +++"
    )

    X_impl, y_impl = load_streamlines(
        implausible_file,
        config[ThresholdTestKeys.REFERENCE.value],
        100,
        resample=True,
        num_points=config[ThresholdTestKeys.STREAMLINE_LENGTH.value],
    )

    if config[ThresholdTestKeys.MAX_IMPLAUSIBLE.value] is not None:
        if config[ThresholdTestKeys.MAX_IMPLAUSIBLE.value] < X_impl.shape[0]:
            logger.info(f" +++ Subsampling Threshold Implausible Streamlines  +++")
            indices = np.random.choice(
                X_impl.shape[0],
                size=config[ThresholdTestKeys.MAX_IMPLAUSIBLE.value],
                replace=False,
            )
            X_impl = X_impl[indices]
            y_impl = y_impl[indices]

    logger.info(" +++ Loading valid bundle files +++")

    latent_atlas_dict = {}
    latent_thres_dict = {}
    streamlines_dict = {}
    streamline_classes = read_data_from_json_file(
        config[ThresholdTestKeys.STREAMLINE_CLASSES.value]
    )

    latent_all_f = np.empty((0, config[ThresholdTestKeys.LATENT_DIMS.value]))
    y_latent_all_f = np.empty((0,))
    class_names = []

    latent_all = np.empty((0, config[ThresholdTestKeys.LATENT_DIMS.value]))
    y_latent_all = np.empty((0,))
    class_names_all = []

    for f in valid_bundles_files:
        if f not in atlas_files:
            continue

        key = f.split(".")[-2]

        # if "CG_L" not in key and "CG_R" not in key:
        #     continue
        streamline_class = streamline_classes[key]

        a = [i for i in atlas_files if i == f][0]

        threshold_bundle_file = pjoin(
            config[ThresholdTestKeys.VALID_BUNDLE_PATH.value], f
        )

        logger.info(f" +++ Loading Threshold Bundle File: {threshold_bundle_file} +++")

        X_f_not_flipped, y_f_not_flipped = load_streamlines(
            threshold_bundle_file,
            config[ThresholdTestKeys.REFERENCE.value],
            streamline_class,
            resample=True,
            num_points=config[ThresholdTestKeys.STREAMLINE_LENGTH.value],
        )

        X_f_flipped, y_f_flipped = load_streamlines(
            threshold_bundle_file,
            config[ThresholdTestKeys.REFERENCE.value],
            streamline_class,
            resample=True,
            num_points=config[ThresholdTestKeys.STREAMLINE_LENGTH.value],
            flip_all_streamlines=True,
        )

        X_f = np.vstack((X_f_not_flipped, X_f_flipped))
        y_f = np.hstack((y_f_not_flipped, y_f_flipped))

        if config[ThresholdTestKeys.MAX_PLAUSIBLE.value] is not None:
            assert 0 < config[ThresholdTestKeys.MAX_PLAUSIBLE.value] <= 1
            logger.info(f" +++ Subsampling Threshold Plausible Streamlines  +++")
            indices = np.random.choice(
                X_f.shape[0],
                size=int(config[ThresholdTestKeys.MAX_PLAUSIBLE.value] * X_f.shape[0]),
                replace=False,
            )
            X_f = X_f[indices]
            y_f = y_f[indices]

        streamlines_dict[key] = (X_f, y_f)

        atlas_bundle_file = pjoin(config[ThresholdTestKeys.ATLAS_PATH.value], a)

        logger.info(f" +++ Loading Atlas Bundle File: {atlas_bundle_file} +++")

        X_atlas_not_flipped, y_atlas_not_flipped = load_streamlines(
            atlas_bundle_file,
            config[ThresholdTestKeys.REFERENCE.value],
            streamline_class + 1000,
            resample=True,
            num_points=config[ThresholdTestKeys.STREAMLINE_LENGTH.value],
        )

        X_atlas_flipped, y_atlas_flipped = load_streamlines(
            atlas_bundle_file,
            config[ThresholdTestKeys.REFERENCE.value],
            streamline_class + 1000,
            resample=True,
            num_points=config[ThresholdTestKeys.STREAMLINE_LENGTH.value],
            flip_all_streamlines=True,
        )

        X_atlas = np.vstack((X_atlas_not_flipped, X_atlas_flipped))
        y_atlas = np.hstack((y_atlas_not_flipped, y_atlas_flipped))

        logger.info(f" +++ Encoding Atlas and Threshold {key} +++")

        f_dataset = OnTheFlyDataset(X_f, y_f)
        a_dataset = OnTheFlyDataset(X_atlas, y_atlas)
        f_dataloader = torch.utils.data.DataLoader(
            f_dataset, batch_size=128, shuffle=True
        )
        a_dataloader = torch.utils.data.DataLoader(
            a_dataset, batch_size=128, shuffle=True
        )

        latent_f, y_latent_f = encode_data(f_dataloader, device, model)
        latent_a, y_latent_a = encode_data(a_dataloader, device, model)

        latent_atlas_dict[key] = (latent_a, y_latent_a)
        latent_thres_dict[key] = (latent_f, y_latent_f)
        class_names.append(key)
        latent_all_f = np.vstack((latent_all_f, latent_f))
        y_latent_all_f = np.hstack((y_latent_all_f, y_latent_f))

        class_names_all.append(key)
        latent_all = np.vstack((latent_all, latent_f))
        y_latent_all = np.hstack((y_latent_all, y_latent_f))

        class_names_all.append(key + "_atlas")
        latent_all = np.vstack((latent_all, latent_a))
        y_latent_all = np.hstack((y_latent_all, y_latent_a))

    logger.info(f" +++ Encoding Threshold Implausible Streamlines +++")

    impl_dataset = OnTheFlyDataset(X_impl, y_impl)
    impl_dataloader = torch.utils.data.DataLoader(
        impl_dataset, batch_size=128, shuffle=True
    )
    latent_impl, y_latent_impl = encode_data(impl_dataloader, device, model)

    logger.info(f" +++ Stacking implausible streamlines +++")
    latent_all_f = np.vstack((latent_impl, latent_all_f))
    y_latent_all_f = np.hstack((y_latent_impl, y_latent_all_f))
    class_names.insert(0, "implausible")

    latent_thres_dict["implausible"] = (latent_impl, y_latent_impl)
    streamlines_dict["implausible"] = (X_impl, y_impl)

    fname_root = pjoin(
        experiment_dir, LoggerKeys.latent_plot_fname_label.value + "_trk"
    )
    fname_root_atlas = pjoin(
        experiment_dir, LoggerKeys.latent_plot_fname_label.value + "_trk_atlas"
    )

    if config[ThresholdTestKeys.VIZ.value]:
        logger.info(f" +++ Latent Space Visualisation  +++")
        plot_latent_space(
            latent_all_f,
            y_latent_all_f,
            class_names,
            config[ThresholdTestKeys.LATENT_DIMS.value],
            fname_root,
        )
        plot_latent_space(
            latent_all,
            y_latent_all,
            class_names_all,
            config[ThresholdTestKeys.LATENT_DIMS.value],
            fname_root_atlas,
        )

    thresholds_dict = set_threshold(
        config,
        streamlines_dict,
        latent_atlas_dict,
        latent_thres_dict,
        streamline_classes,
        model,
        experiment_dir,
    )

    save_data_to_json_file(thresholds_dict, pjoin(experiment_dir, "thresholds.json"))

    logger.info("Thresholds saved !")


if __name__ == "__main__":
    main()
