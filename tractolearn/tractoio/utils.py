# -*- coding: utf-8 -*-

import functools
import json
import os
import pickle
from os.path import join as pjoin
from typing import Tuple

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram

from tractolearn import anatomy
from tractolearn.anatomy import BundleSettings
from tractolearn.anatomy.bundles_additional_labels import BundlesAdditionalLabels
from tractolearn.transformation.streamline_transformation import (
    flip_random_streamlines,
    resample_streamlines,
    flip_streamlines,
)

# fixme: FD
asterisk_wildcard = "*"
extension_sep = "."
underscore_sep = "_"

csv_extension = "csv"
mat_extension = "mat"
nii_gz_extension = "nii.gz"
trk_extension = "trk"
vtk_extension = "vtk"

bundle_label = "bundle"

invalid_bundle_label = "IB"
invalid_connection_label = "IC"
valid_bundle_label = "VB"
valid_connection_label = "VC"
nc_label = "NC"

clustering_parameters_label = "clustering_parameters"
time_stats_label = "time_stats"

clustering_parameters_file_basename = "clustering_parameters.json"
performance_scores_file_basename = "performance_scores.json"
time_stats_file_basename = "time_stats.json"

brain_tissue_map_fname_label = "brain_mask"
gm_tissue_map_fname_label = "gm_mask"
wm_tissue_map_fname_label = "wm_mask"
fodf_peak_fname_label = "fodf_peaks"  # "dwi_fodf"
structural_data_fname_label = "t1"
surface_fname_label = "interface"

transf_matrix_fname_label = "output0GenericAffine"
warping_fname_label = "output1Warp"

structural_data_dir_label = "structural"
diffusion_data_dir_label = "diffusion"
surface_data_dir_label = "surface"
transformation_data_dir_label = "transformation"


def load_bundles_dict(dataset_name):
    """Loads the bundle dictionary containing their names and classes
    corresponding to a dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name whose bundle names and classes are to be fetched.
        Supported datasets are listed in
        :file:`anatomy.bundles_dictionaries.json`.

    Returns
    -------
    dict or None
        Bundle names and corresponding classes if the dictionary is defined
        for the dataset; None otherwise.
    """

    anatomy_path = os.path.dirname(anatomy.__file__)
    bundles_dictionaries_fname = pjoin(
        anatomy_path, BundleSettings.BUNDLE_DICTIONARIES.value
    )
    with open(bundles_dictionaries_fname) as f:
        bundles_dictionaries = json.load(f)

    if dataset_name in bundles_dictionaries.keys():
        bundles_dictionary_fname = pjoin(
            anatomy_path, bundles_dictionaries[dataset_name]
        )

        with open(bundles_dictionary_fname) as f:
            return json.load(f)
    else:
        return None


def write_bundles(anat_ref_fname, class_lookup, streamlines, predicted_classes, path):

    shifted_origin = False
    space = Space.RASMM

    for bundle_name, bundle_class in class_lookup.items():

        bundle = [
            streamline
            for i, streamline in enumerate(streamlines)
            if predicted_classes[i] == bundle_class
        ]

        tractogram = StatefulTractogram(
            bundle, anat_ref_fname, space=space, shifted_origin=shifted_origin
        )

        tractogram_file_basename = (
            bundle_label + underscore_sep + bundle_name + extension_sep + trk_extension
        )
        fname = pjoin(path, tractogram_file_basename)
        save_tractogram(tractogram, fname)


def load_streamlines(
    fname: str,
    ref_anat_fname: str,
    streamlines_class: int,
    resample: bool = False,
    num_points: int = 256,
    flip_all_streamlines: bool = False,
) -> Tuple[np.array, np.array]:

    img = load_ref_anat_image(ref_anat_fname)

    # Read streamlines
    to_space = Space.RASMM
    trk_header_check = False
    bbox_valid_check = False

    tractogram = load_tractogram(
        fname,
        img.header,
        to_space=to_space,
        trk_header_check=trk_header_check,
        bbox_valid_check=bbox_valid_check,
    )

    streamlines = tractogram.streamlines

    if len(streamlines) == 0:
        raise RuntimeError(
            "Tractogram in filename {} contains no "
            "streamlines. Please remove the file from the "
            "experiment.".format(fname)
        )

    if resample:
        streamlines = resample_streamlines(streamlines, num_points, arc_length=True)

    if flip_all_streamlines:
        streamlines = flip_streamlines(streamlines)

    # Dump streamline data to array
    streamlines_data = np.vstack(
        [
            streamlines[i][
                np.newaxis,
            ]
            for i in range(len(streamlines))
        ]
    )

    streamlines_classes = np.hstack(
        [np.repeat(streamlines_class, len(streamlines_data))]
    )

    return streamlines_data, streamlines_classes


@functools.lru_cache(maxsize=2)
def load_ref_anat_image(ref_anat_fname):

    img = nib.load(ref_anat_fname)
    return img


def load_process_streamlines2(
    fname,
    ref_anat_fname,
    streamline_class_name,
    random_flip=True,
    random_flip_ratio=0.3,
):

    img = load_ref_anat_image(ref_anat_fname)

    # Read streamlines
    to_space = Space.RASMM
    trk_header_check = False
    bbox_valid_check = False
    tractogram = load_tractogram(
        fname,
        img.header,
        to_space=to_space,
        trk_header_check=trk_header_check,
        bbox_valid_check=bbox_valid_check,
    )

    if len(tractogram.streamlines) == 0:
        raise RuntimeError(
            "Tractogram in filename {} contains no "
            "streamlines. Please remove the file from the "
            "experiment.".format(fname)
        )

    # Apply random flip
    if random_flip:
        flipped_streamlines, _ = flip_random_streamlines(
            tractogram.streamlines, random_flip_ratio
        )
    else:
        flipped_streamlines = tractogram.streamlines

    # Dump streamline data to array
    streamlines_data = np.vstack(
        [
            flipped_streamlines[i][
                np.newaxis,
            ]
            for i in range(len(flipped_streamlines))
        ]
    )

    if streamline_class_name == "plausible":
        streamlines_class = 0
    elif streamline_class_name == "implausible":
        streamlines_class = 100
    else:
        streamlines_class = BundlesAdditionalLabels.generic_streamline_class.value

    streamlines_classes = np.hstack(
        [np.repeat(streamlines_class, len(flipped_streamlines))]
    )

    return streamlines_data, streamlines_classes


def load_data2(
    fname,
    ref_anat_fname,
    streamline_class_name,
    random_flip=True,
    random_flip_ratio=0.3,
):

    x, y = load_process_streamlines2(
        fname,
        ref_anat_fname,
        streamline_class_name,
        random_flip=random_flip,
        random_flip_ratio=random_flip_ratio,
    )

    return x, y


def save_streamlines(
    streamlines, ref_anat_fname, tractogram_fname, data_per_streamline: dict = None
):

    space = Space.RASMM
    tractogram = StatefulTractogram(
        streamlines,
        ref_anat_fname,
        space=space,
        data_per_streamline=data_per_streamline,
    )

    bbox_valid_check = False
    save_tractogram(tractogram, tractogram_fname, bbox_valid_check=bbox_valid_check)


def save_data_to_json_file(data, fname):

    with open(fname, "w") as f:
        json.dump(data, f)


def read_data_from_json_file(fname):

    with open(fname, "r") as f:
        return json.load(f)


def save_loss_history(loss_history, fname):

    save_data_to_json_file(loss_history, fname)


def save_data_to_pickle_file(data, fname):

    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_data_from_pickle_file(fname):

    with open(fname, "rb") as f:
        return pickle.load(f)


def load_streamline_learning_data(fname, ref_anat_fname, anatomy, random_flip=False):

    bundle_names = list(anatomy.keys())
    bundle_names.sort()

    streamline_data = []
    streamline_class = []

    # ToDo
    # This should ensure that a given bundle name is not contained in
    # another bundle name
    bundle_class = [name for name in bundle_names if name in fname]

    if len(bundle_class) != 0:
        bundle_class = bundle_class[0]

    data, _ = load_data2(fname, ref_anat_fname, bundle_class, random_flip=random_flip)

    target = [anatomy[bundle_class]] * len(data)
    streamline_data.extend(data)
    streamline_class.extend(target)

    return np.array(streamline_data), np.array(streamline_class)
