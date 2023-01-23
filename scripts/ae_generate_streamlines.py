#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script used to generate WM streamlines based on an autoencoder latent space
"""

import argparse
import logging
import os
import random
import sys
from os.path import join as pjoin, exists
from random import randint

import nibabel as nib
import numpy as np
import torch
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram
from nibabel.streamlines import ArraySequence
from scilpy.io.utils import (
    add_verbose_arg,
    add_overwrite_arg,
    load_matrix_in_any_format,
)
from scilpy.tracking.tools import filter_streamlines_by_length
from scilpy.utils.streamlines import transform_warp_sft
from tqdm import tqdm

from tractolearn.filtering.streamline_space_filtering import (
    filter_grid_roi,
    StreamlineLocalOrientationAnalyzer,
    StreamlineLocalOrientationChecker,
    cut_streamlines_outside_mask,
)
from tractolearn.generative.generate_points import generate_points
from tractolearn.logger import _set_up_logger, LoggerKeys
from tractolearn.models.model_pool import get_model
from tractolearn.tractoio.utils import (
    read_data_from_json_file,
    load_streamlines,
    save_streamlines,
)
from tractolearn.utils.timer import Timer

torch.set_flush_denormal(True)
logger = logging.getLogger("root")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--in_bundles_common_space",
        help="Seed bundles [ *.trk ]. "
        "Subject specific bundles obtained from a whole-brain tractogram bundling process. "
        "These bundles are used to seed the AE latent space.",
        nargs="+",
    )

    parser.add_argument(
        "--model", help="AutoEncoder model file (AE) [ *.pt ].", required=True
    )

    parser.add_argument(
        "--reference_common_space",
        help="Reference T1 file [ *.nii/.nii.gz ].",
        required=True,
    )
    parser.add_argument(
        "--reference_native",
        help="Reference T1 file [ *.nii/.nii.gz ]. "
        "Used to apply the transform Common Space -> Native space"
        " for the peak filtering process (done in native space)",
        required=True,
    )

    parser.add_argument(
        "--anatomy_file",
        help="Anatomy file [ *.json ]. JSON file containing bundle names with corresponding class label.",
        required=True,
    )

    parser.add_argument("--output", help="Output path", required=True)

    parser.add_argument(
        "--atlas_path",
        help="Path containing all atlas bundles [ *.trk ] used to seed the latent space. "
        "Bundles must be in the same space as the argument --in_bundles_common_space.",
    )

    parser.add_argument(
        "--wm_parc_common_space",
        help="White matter parcellation in the common space used for WM filtering [ *.nii/.nii.gz ].",
        required=True,
    )
    parser.add_argument(
        "--fa_common_space",
        help="FA image in common space used for WM filtering [ *.nii/.nii.gz ].",
        required=True,
    )
    parser.add_argument(
        "--threshold_fa",
        help="Threshold value for FA WM filtering",
        type=float,
        default=0.1,
        required=True,
    )
    parser.add_argument(
        "--peaks",
        help="FODF peaks [ *.nii/.nii.gz ] used for peak filtering in native space",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--device",
        help="Device to use for inference [ cpu | cuda ]",
        default="cpu",
    )

    parser.add_argument(
        "-n",
        "--num_generated_streamlines",
        help="Config file [ *.json ] for the per-bundle desired number of generated streamlines.",
    )

    parser.add_argument(
        "--max_total_sampling",
        help="Config file [ *.json ] for the per-bundle maximum number of generated streamlines."
        "Use this argument to limit the time spent on RS for hard bundles.",
    )

    parser.add_argument(
        "-r",
        "--ratio",
        help="Config file [ *.json ] for the per-bundle AE latent space desired seed ratio "
        "( subject bundle|atlas bundle )",
    )

    parser.add_argument(
        "-b",
        "--bandwidth",
        help="Bandwidth size used for the parzen window. Default is none, meaning that we used the "
        "Silverman’s rule of thumb for the bandwidth estimation",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-m",
        "--max_seeds",
        help="Maximum number of seeds to use per bundle for the AE latent space",
        type=int,
    )
    parser.add_argument(
        "--minL",
        help="Minimum length of generated streamlines, in mm.",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--maxL",
        help="Maximum length of generated streamlines, in mm.",
        type=int,
        default=220,
    )
    parser.add_argument("-p", "--plot_umaps", help="Plot umaps", action="store_true")
    parser.add_argument(
        "-a",
        "--generate_all_bundles",
        help="If flagged, will generate all bundles present in the atlas, "
        "even if they were not present in the --in_bundles_common_space argument."
        "Seed from missing bundle, will entirely be taken from the atlas even if the ratio is not ( 0 | 1 )",
        action="store_true",
    )
    parser.add_argument(
        "--use_rs",
        help="If not flagged, will use the default sklearn gaussian sampling process. ",
        action="store_true",
    )
    parser.add_argument(
        "--batch_sampling",
        help="Sampling batch size. Use this argument if you want to prevent excessive memory usage. ",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--gmm_n_component",
        help="Number of components for the Gaussian Mixture Model used as a proposal "
        "distribution for RS. Only used if --use_rs is flagged.",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--degree",
        help="Config file [ *.json ] for the per-bundle maximum degree angle for the peaks filtering",
    )
    parser.add_argument(
        "--in_transfo",
        help="Path of the file containing the 4x4 transformation, matrix (.txt, .npy or .mat).",
    )
    parser.add_argument(
        "--in_deformation",
        help="Path to the file containing a deformation field [ *.nii/.nii.gz ].",
    )

    parser.add_argument(
        "--white_matter_config",
        help="Config file [ *.json ] used to determine the type of WM filtering ( WM mask | thresholded FA ).",
    )

    add_overwrite_arg(parser)
    add_verbose_arg(parser)

    return parser.parse_args()


def main():
    args = _build_arg_parser()
    device = torch.device(args.device)

    if exists(args.output):
        if not args.overwrite:
            print(f"Outputs directory {args.output} exists. Use -f to for overwriting.")
            sys.exit(1)
    else:
        os.makedirs(args.output)

    if args.verbose:
        logging.basicConfig(level=logging.INFO if args.verbose == 1 else logging.DEBUG)

    logger.info(args)

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
    degree_config = read_data_from_json_file(args.degree)
    ratio_config = read_data_from_json_file(args.ratio)
    max_total_sampling_config = read_data_from_json_file(args.max_total_sampling)
    num_generated_streamlines_config = read_data_from_json_file(
        args.num_generated_streamlines
    )
    white_matter_config = read_data_from_json_file(args.white_matter_config)

    assert degree_config.keys() <= streamline_classes.keys()
    assert ratio_config.keys() <= streamline_classes.keys()
    assert max_total_sampling_config.keys() <= streamline_classes.keys()
    assert num_generated_streamlines_config.keys() <= streamline_classes.keys()
    assert white_matter_config.keys() <= streamline_classes.keys()

    peaks = nib.load(args.peaks).get_fdata("unchanged")

    atlas_files = None

    if args.atlas_path:
        atlas_files = os.listdir(args.atlas_path)

    logger.info("Loading atlas files ...")

    if args.generate_all_bundles:
        assert (
            atlas_files is not None
        ), "To generate all bundles, you need to provide a bundle atlas"
        bundles = atlas_files
    else:
        bundles = args.in_bundles_common_space

    for f in tqdm(bundles):
        key = [k for k in streamline_classes if k in f][0]

        logger.info(f"Generating: {key}")
        mni_bundle_file = [file for file in args.in_bundles_common_space if key in file]

        if len(mni_bundle_file) > 1:
            raise ValueError(
                f"Only one mni file must be provided for every bundle."
                f"Got {len(mni_bundle_file)} for {key}."
            )

        if not args.generate_all_bundles:
            mni_bundle_file = mni_bundle_file[0]

        else:
            if len(mni_bundle_file) == 0:
                mni_bundle_file = None
            else:
                mni_bundle_file = mni_bundle_file[0]

        if mni_bundle_file is not None:
            X_f, _ = load_streamlines(
                mni_bundle_file,
                args.reference_common_space,
                streamline_classes[key],
                resample=True,
                num_points=256,
            )

            logger.debug(f"X_f: {X_f.shape}")

        else:
            X_f = None

        if atlas_files:
            atlas_file = [file for file in atlas_files if key in file][0]
            X_a_not_flipped, _ = load_streamlines(
                pjoin(args.atlas_path, atlas_file),
                args.reference_common_space,
                streamline_classes[key],
                resample=True,
                num_points=256,
            )

            X_a_flipped, _ = load_streamlines(
                pjoin(args.atlas_path, atlas_file),
                args.reference_common_space,
                streamline_classes[key],
                resample=True,
                num_points=256,
                flip_all_streamlines=True,
            )

            X_a = np.vstack((X_a_not_flipped, X_a_flipped))
            logger.debug(f"X_a: {X_a.shape}")

        else:
            X_a = None

        logger.info(f"Generating {num_generated_streamlines_config[key]} streamlines")

        assert (X_f is not None) or (
            X_a is not None
        ), "Seeds are needed to generate new streamlines"

        streamline_count = 0
        iteration_count = 0
        total_sampling = 0
        batch = args.batch_sampling

        random.seed(0)
        seed = randint(0, 2**32)

        while streamline_count < num_generated_streamlines_config[key]:

            with Timer():
                X_f_generated = generate_points(
                    output=args.output,
                    name=key,
                    device=args.device,
                    model=model,
                    bundle=X_f,
                    num_generate_points=batch,
                    atlas_bundle=X_a,
                    max_seeds=args.max_seeds,
                    composition=tuple(ratio_config[key]),
                    bandwidth=args.bandwidth,
                    plot_seeds_generated=args.plot_umaps,
                    use_rs=args.use_rs,
                    optimization="max_seeds",
                    gmm_n_component=args.gmm_n_component,
                    random_seed=seed,
                )

            save_streamlines(
                X_f_generated,
                args.reference_common_space,
                pjoin(args.output, f"{key}_generated.trk"),
            )

            tractogram = StatefulTractogram(
                X_f_generated, args.reference_common_space, space=Space.RASMM
            )

            logger.info(f"Hair Cut")
            with Timer():
                X_f_generated_cut = cut_streamlines_outside_mask(
                    tractogram,
                    nib.load(args.wm_parc_common_space).get_fdata("unchanged"),
                    X_f_generated,
                )

            tractogram = StatefulTractogram(
                X_f_generated_cut, args.reference_common_space, space=Space.RASMM
            )

            save_tractogram(
                tractogram,
                pjoin(args.output, f"{key}_generated_haircut.trk"),
                bbox_valid_check=False,
            )

            logger.info(f"White matter mask filtering")

            if white_matter_config[key] == "wm":
                logger.info(f"Using WM mask")
                wm_image = nib.load(args.wm_parc_common_space).get_fdata("unchanged")
            elif white_matter_config[key] == "fa":
                logger.info(f"Using thresholded fa mask")
                wm_image = (
                    nib.load(args.fa_common_space).get_fdata("unchanged")
                    > args.threshold_fa
                ).astype(float)
            else:
                raise ValueError(
                    f"Unrecognized white_matter_config type. Got {white_matter_config[key]}."
                )

            with Timer():
                _, ids = filter_grid_roi(
                    tractogram,
                    wm_image,
                    filter_type="soft_all",
                    soft_percentage=0.95,
                    is_exclude=False,
                )

            save_streamlines(
                X_f_generated_cut[ids],
                args.reference_common_space,
                pjoin(args.output, f"{key}_generated_filtered_mask.trk"),
            )

            logger.info(f"Registering in native.")
            transfo = load_matrix_in_any_format(args.in_transfo)
            deformation_data = np.squeeze(
                nib.load(args.in_deformation).get_fdata(dtype=np.float32)
            )
            with Timer():
                tractogram_native = transform_warp_sft(
                    tractogram,
                    transfo,
                    args.reference_native,
                    inverse=False,
                    reverse_op=True,
                    deformation_data=deformation_data,
                    remove_invalid=False,
                    cut_invalid=False,
                )

            with Timer():
                local_orient_analyzer = StreamlineLocalOrientationAnalyzer()
                logger.info(f"Orientation filtering")
                local_orient_checker = StreamlineLocalOrientationChecker(
                    local_orient_analyzer=local_orient_analyzer,
                    allowed_angle=degree_config[key],
                    mask_value=90,
                    allowed_ratio=0.1,
                )

                local_orient_checker.verify_conditions(
                    tractogram_native, peaks, tractogram_native.affine
                )

            save_streamlines(
                X_f_generated_cut[
                    local_orient_checker._compliant_indices["LOCAL_ORIENTATION_ANGLE"]
                ],
                args.reference_common_space,
                pjoin(args.output, f"{key}_generated_filtered_fodf.trk"),
            )

            sft = StatefulTractogram(
                X_f_generated_cut[
                    list(
                        set(
                            local_orient_checker._compliant_indices[
                                "LOCAL_ORIENTATION_ANGLE"
                            ]
                        ).intersection(set(ids))
                    )
                ],
                args.reference_common_space,
                space=Space.RASMM,
            )

            save_tractogram(
                sft,
                pjoin(args.output, f"{key}_generated_filtered_fodf_mask.trk"),
                bbox_valid_check=False,
            )

            with Timer():
                new_sft = filter_streamlines_by_length(sft, args.minL, args.maxL)
            streamline_count += len(new_sft)

            if streamline_count > num_generated_streamlines_config[key]:
                new_sft = StatefulTractogram.from_sft(
                    ArraySequence(
                        random.sample(
                            list(sft.streamlines),
                            len(new_sft)
                            - (
                                streamline_count - num_generated_streamlines_config[key]
                            ),
                        )
                    ),
                    sft,
                    data_per_point=new_sft.get_data_per_point_keys(),
                    data_per_streamline=new_sft.get_data_per_streamline_keys(),
                )

            ori_len = len(new_sft)
            new_sft.remove_invalid_streamlines()
            logger.info("Removed {} invalid streamlines".format(ori_len - len(new_sft)))
            save_tractogram(
                new_sft,
                pjoin(
                    args.output,
                    f"{key}_generated_filtered_fodf_mask_{args.minL}_{args.maxL}_{iteration_count}.trk",
                ),
            )
            total_sampling += batch

            if total_sampling >= max_total_sampling_config[key]:
                break

            iteration_count += 1


if __name__ == "__main__":
    main()
