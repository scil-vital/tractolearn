#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for used for the bundling of a whole-brain tractogram based on an autoencoder latent space
"""


import argparse
import logging
import os
import sys
from os.path import join as pjoin, exists

import numpy as np
import torch
from dipy.io.stateful_tractogram import Space
from dipy.io.streamline import load_tractogram
from scilpy.io.utils import add_verbose_arg, add_overwrite_arg
from scilpy.tracking.tools import resample_streamlines_num_points
from tqdm import tqdm
from tractolearn.logger import _set_up_logger, LoggerKeys
from tractolearn.filtering.latent_space_featuring import filter_streamlines_only
from tractolearn.learning.dataset import OnTheFlyDataset
from tractolearn.models.autoencoding_utils import encode_data
from tractolearn.models.model_pool import get_model
from tractolearn.tractoio.utils import (
    read_data_from_json_file,
    load_ref_anat_image,
    load_streamlines,
)
from tractolearn.utils.timer import Timer


torch.set_flush_denormal(True)
logger = logging.getLogger("root")


from math import ceil


def batch_filtering(
    args,
    batch_size,
    num_points,
    streamline_classes,
    device,
    model,
    latent_atlas_all,
    y_latent_atlas_all,
    thresholds,
):

    mni_img = load_ref_anat_image(args.common_space_reference)

    logger.info("Loading tractogram ...")

    common_space_tractogram = load_tractogram(
        args.common_space_tractogram,
        mni_img.header,
        to_space=Space.RASMM,
        trk_header_check=False,
        bbox_valid_check=False,
    )

    tractogram = common_space_tractogram
    reference = args.common_space_reference

    if args.original_tractogram:
        original_img = load_ref_anat_image(args.original_reference)
        original_tractogram = load_tractogram(
            args.original_tractogram,
            original_img.header,
            to_space=Space.RASMM,
            trk_header_check=False,
            bbox_valid_check=False,
        )

        if len(original_tractogram) == 0:
            raise RuntimeError(
                "Original Tractogram in filename {} contains no "
                "streamlines. Please remove the file from the "
                "experiment.".format(args.original_tractogram)
            )

        assert len(common_space_tractogram) == len(original_tractogram)

        tractogram = original_tractogram
        reference = args.original_reference

    if len(common_space_tractogram) == 0:
        raise RuntimeError(
            "MNI Tractogram in filename {} contains no "
            "streamlines. Please remove the file from the "
            "experiment.".format(args.common_space_tractogram)
        )

    batch_num = ceil(len(common_space_tractogram) / batch_size)

    tractogram.data_per_streamline["ids"] = list(range(len(tractogram)))

    for i in tqdm(range(batch_num)):
        with Timer():
            if len(common_space_tractogram[i * batch_size : (i + 1) * batch_size]) == 0:
                continue
            streamlines = resample_streamlines_num_points(
                common_space_tractogram[i * batch_size : (i + 1) * batch_size],
                num_points,
            ).streamlines

            ids = tractogram.data_per_streamline["ids"][
                i * batch_size : (i + 1) * batch_size
            ]

        # Dump streamline data to array
        X = np.vstack(
            [
                streamlines[i][
                    np.newaxis,
                ]
                for i in range(len(streamlines))
            ]
        )

        y = np.arange(0, len(X))

        dataset = OnTheFlyDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

        latent_f, y_f = encode_data(dataloader, device, model)
        assert np.all(y == y_f)

        filter_streamlines_only(
            streamline_classes,
            latent_f,
            latent_atlas_all,
            y_latent_atlas_all,
            model,
            thresholds,
            reference=reference,
            output_path=args.output,
            trk_ids=ids,
            trk=tractogram,
            num_neighbors=args.num_neighbors,
            id_bundle=i,
        )


def whole_filtering(
    args,
    num_points,
    streamline_classes,
    device,
    model,
    latent_atlas_all,
    y_latent_atlas_all,
    thresholds,
):

    mni_img = load_ref_anat_image(args.common_space_reference)

    logger.info("Loading tractogram ...")

    common_space_tractogram = load_tractogram(
        args.common_space_tractogram,
        mni_img.header,
        to_space=Space.RASMM,
        trk_header_check=False,
        bbox_valid_check=False,
    )

    tractogram = common_space_tractogram
    reference = args.common_space_reference

    if args.original_tractogram:
        original_img = load_ref_anat_image(args.original_reference)
        original_tractogram = load_tractogram(
            args.original_tractogram,
            original_img.header,
            to_space=Space.RASMM,
            trk_header_check=False,
            bbox_valid_check=False,
        )

        if len(original_tractogram) == 0:
            raise RuntimeError(
                "Original Tractogram in filename {} contains no "
                "streamlines. Please remove the file from the "
                "experiment.".format(args.original_tractogram)
            )

        assert len(common_space_tractogram) == len(original_tractogram)

        tractogram = original_tractogram
        reference = args.original_reference

    if len(common_space_tractogram) == 0:
        raise RuntimeError(
            "MNI Tractogram in filename {} contains no "
            "streamlines. Please remove the file from the "
            "experiment.".format(args.common_space_tractogram)
        )

    streamlines = resample_streamlines_num_points(
        common_space_tractogram, num_points
    ).streamlines

    # Dump streamline data to array
    X = np.vstack(
        [
            streamlines[i][
                np.newaxis,
            ]
            for i in range(len(streamlines))
        ]
    )

    y = np.arange(0, len(X))

    dataset = OnTheFlyDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    latent_f, y_f = encode_data(dataloader, device, model)
    assert np.all(y == y_f)

    tractogram.data_per_streamline["ids"] = list(range(len(tractogram)))

    filter_streamlines_only(
        streamline_classes,
        latent_f,
        latent_atlas_all,
        y_latent_atlas_all,
        model,
        thresholds,
        reference=reference,
        output_path=args.output,
        trk_ids=tractogram.data_per_streamline["ids"],
        trk=tractogram,
        num_neighbors=args.num_neighbors,
    )


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "common_space_tractogram",
        help="Tractogram to bundle [ *.trk ]. Must be in the same as the one used to train the AE",
    )

    parser.add_argument(
        "atlas_path",
        help="Path containing all atlas bundles [ *.trk ] used to bundle the tractogram. "
        "Bundles must be in the same space as the common_space_tractogram.",
    )

    parser.add_argument(
        "model",
        help="AutoEncoder model file (AE) [ *.pt ]",
    )

    parser.add_argument(
        "common_space_reference",
        help="Reference T1 file [ *.nii/.nii.gz ]",
    )

    parser.add_argument(
        "thresholds_file",
        help="Config file [ *.json ] containing the per-bundle latent space distance thresholds. ",
    )

    parser.add_argument(
        "anatomy_file",
        help="Anatomy file [ *.json ]. JSON file containing bundle names with corresponding class label.",
    )

    parser.add_argument(
        "output",
        help="Output path",
    )

    parser.add_argument(
        "--original_tractogram",
        help="Tractogram in the original space."
        "If a file is passed, output bundles will be in "
        "its space. Else, it will be in common_space_tractogram space [ *.trk ]",
    )

    parser.add_argument(
        "--original_reference",
        help="Reference T1 file in the native space [ *.nii/.nii.gz ]",
    )

    parser.add_argument(
        "-d",
        "--device",
        help="Device to use for inference [ cpu | cuda ]",
        default="cpu",
    )

    parser.add_argument(
        "-b",
        "--batch_loading",
        help="If the size of the tractogram is too big, "
        "you can filter it by batches. Will produce many files for one bundles",
        type=int,
        default=None,
    )

    parser.add_argument(
        "-n",
        "--num_neighbors",
        help="Number of neighbors to consider for classification. Maximum allowed (30)",
        type=int,
        default=1,
    )

    add_overwrite_arg(parser)
    add_verbose_arg(parser)

    return parser.parse_args()


def main():
    args = _build_arg_parser()
    device = torch.device(args.device)

    print(args)

    if exists(args.output):
        if not args.overwrite:
            print(f"Outputs directory {args.output} exists. Use -f to for overwriting.")
            sys.exit(1)
    else:
        os.makedirs(args.output)

    if args.verbose:
        logging.basicConfig(level=logging.INFO if args.verbose == 1 else logging.DEBUG)

    if args.num_neighbors < 1:
        raise ValueError(
            f"Number of specified neighbors are below 1."
            f"Please specify a number between 1 and 30. Got {args.num_neighbors}. "
        )

    if args.num_neighbors > 30:
        raise ValueError(
            f"Number of specified neighbors are above 30."
            f"Please specify a number between 1 and 30. Got {args.num_neighbors}. "
        )

    logging.info(args)

    _set_up_logger(pjoin(args.output, LoggerKeys.logger_file_basename.name))

    checkpoint = torch.load(
        args.model,
        map_location=device,
    )
    state_dict = checkpoint["state_dict"]
    model = get_model("IncrFeatStridedConvFCUpsampReflectPadAE", 32, device)
    model.load_state_dict(state_dict)
    model.eval()

    streamline_classes = read_data_from_json_file(args.anatomy_file)

    thresholds = read_data_from_json_file(args.thresholds_file)

    latent_atlas_all = np.empty((0, 32))
    y_latent_atlas_all = np.empty((0,))

    atlas_file = os.listdir(args.atlas_path)

    logger.info("Loading atlas files ...")

    for f in tqdm(atlas_file):

        key = f.split(".")[-2]

        assert key in thresholds.keys(), f"[!] Threshold: {key} not in threshold file"

        X_a_not_flipped, y_a_not_flipped = load_streamlines(
            pjoin(args.atlas_path, f),
            args.common_space_reference,
            streamline_classes[key],
            resample=True,
            num_points=256,
        )

        X_a_flipped, y_a_flipped = load_streamlines(
            pjoin(args.atlas_path, f),
            args.common_space_reference,
            streamline_classes[key],
            resample=True,
            num_points=256,
            flip_all_streamlines=True,
        )

        X_a = np.vstack((X_a_not_flipped, X_a_flipped))
        y_a = np.hstack((y_a_not_flipped, y_a_flipped))

        dataset = OnTheFlyDataset(X_a, y_a)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        latent_a, y_latent_a = encode_data(dataloader, device, model)

        latent_atlas_all = np.vstack((latent_atlas_all, latent_a))
        y_latent_atlas_all = np.hstack((y_latent_atlas_all, y_latent_a))

    if args.batch_loading:
        batch_filtering(
            args,
            args.batch_loading,
            256,
            streamline_classes,
            device,
            model,
            latent_atlas_all,
            y_latent_atlas_all,
            thresholds,
        )

    else:
        whole_filtering(
            args,
            256,
            streamline_classes,
            device,
            model,
            latent_atlas_all,
            y_latent_atlas_all,
            thresholds,
        )


if __name__ == "__main__":
    main()
