# -*- coding: utf-8 -*-
import logging
from enum import Enum

import numpy as np
import torch
from dipy.io.stateful_tractogram import StatefulTractogram
from nibabel.streamlines import ArraySequence
from scilpy.segment.streamlines import streamlines_in_mask
from scipy.ndimage import map_coordinates

from tractolearn.tractomath.utils import unit_vector, compute_angle
from tractolearn.transformation.peaks_utils import (
    get_peak_count,
    reshape_peaks_for_computing,
)
from tractolearn.transformation.streamline_transformation import (
    resample_streamlines,
    sft_voxel_transform,
)
from tractolearn.transformation.volume_utils import interpolate_volume_at_coordinates

torch.set_flush_denormal(True)
logger = logging.getLogger("root")


class StreamlineFeatures(Enum):
    FA = "fa"
    LENGTH = "length"
    LOCAL_ORIENTATION_ANGLE = "local_orientation_angle"
    MEAN_CURVATURE = "mean_curvature"
    REGION = "region"
    WINDING = "winding"
    WM_OCCUPANCY = "wm_occupancy"
    GM_OCCUPANCY = "gm_occupancy"
    WM_MEMBERSHIP = "wm_membership"
    WM_VOXEL_OCCUPANCY_RATIO = "wm_voxel_occupancy_ratio"
    CORTICAL_GAP = "cortical_gap"
    SURFACE_INTERSECTION = "surface_intersection"
    SURFACE_INCIDENCE_ANGLE = "surface_incidence_angle"
    SURFACE_PROJECTED_ANGLE = "surface_projected_angle"


def filter_grid_roi(sft, mask, filter_type, is_exclude, soft_percentage: float = None):
    """
    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.
    target_mask : numpy.ndarray
        Binary mask in which the streamlines should pass.
    filter_type: str
        One of the 3 following choices, 'any', 'all', 'either_end', 'both_ends'.
    is_exclude: bool
        Value to indicate if the ROI is an AND (false) or a NOT (true).
    Returns
    -------
    ids : tuple
        Filtered sft.
        Ids of the streamlines passing through the mask.
    """
    line_based_indices = []
    if filter_type in ["any", "all"]:
        line_based_indices = streamlines_in_mask(sft, mask, all_in=filter_type == "all")
    else:
        sft_voxel_transform(sft)
        streamline_vox = sft.streamlines
        # For endpoint filtering, we need to keep 2 separately
        # Could be faster for either end, but the code look cleaner like this
        line_based_indices_1 = []
        line_based_indices_2 = []
        line_based_indices_3 = []
        for i, line_vox in enumerate(streamline_vox):
            voxel_1 = line_vox[0].astype(np.int16)[:, None]
            voxel_2 = line_vox[-1].astype(np.int16)[:, None]
            voxel_3 = line_vox.astype(np.int16).transpose((1, 0))
            if map_coordinates(mask, voxel_1, order=0, mode="nearest"):
                line_based_indices_1.append(i)
            if map_coordinates(mask, voxel_2, order=0, mode="nearest"):
                line_based_indices_2.append(i)
            if filter_type == "soft_all":
                if np.count_nonzero(
                    map_coordinates(mask, voxel_3, order=0, mode="nearest")
                ) > soft_percentage * len(line_vox):
                    line_based_indices_3.append(i)

        # Both endpoints need to be in the mask (AND)
        if filter_type == "both_ends":
            line_based_indices = np.intersect1d(
                line_based_indices_1, line_based_indices_2
            )
        # Only one endpoint need to be in the mask (OR)
        elif filter_type == "either_end":
            line_based_indices = np.union1d(line_based_indices_1, line_based_indices_2)

        elif filter_type == "soft_all":
            line_based_indices = line_based_indices_3

    # If the 'exclude' option is used, the selection is inverted
    if is_exclude:
        line_based_indices = np.setdiff1d(
            range(len(sft)), np.unique(line_based_indices)
        )
    line_based_indices = np.asarray(line_based_indices, dtype=np.int32)

    # From indices to sft
    streamlines = sft.streamlines[line_based_indices]
    data_per_streamline = sft.data_per_streamline[line_based_indices]
    data_per_point = sft.data_per_point[line_based_indices]

    new_sft = StatefulTractogram.from_sft(
        streamlines,
        sft,
        data_per_streamline=data_per_streamline,
        data_per_point=data_per_point,
    )

    return new_sft, line_based_indices


def cut_streamlines_outside_mask(sft, mask, streamlines):
    """Cut streamlines so their longest segment are within the bounding box.
    This function keeps the data_per_point and data_per_streamline.

    Parameters
    ----------
    sft: StatefulTractogram
        The sft to remove invalid points from.
    mask: np.array
        Mask used to cut streamlines

    Returns
    -------
    new_sft : StatefulTractogram
        Trimmed sft
    """

    sft.to_vox()
    sft.to_corner()
    streamline_vox = sft.streamlines

    cut_streamlines = []
    for i, line_vox in enumerate(streamline_vox):
        voxel_1 = line_vox.astype(np.int16).transpose((1, 0))
        mapped_coord = map_coordinates(mask, voxel_1, order=0, mode="nearest")
        ids = np.nonzero(mapped_coord)

        if len(ids[0]) < 3:
            continue
        else:
            cut_streamline = streamlines[i][ids[0][0] : ids[0][-1] + 1]
            if len(cut_streamline) < len(streamlines[i]):
                cut_streamline = resample_streamlines(
                    ArraySequence(
                        [
                            cut_streamline,
                        ]
                    ),
                    streamlines.shape[1],
                    arc_length=True,
                )[0]
            cut_streamlines.append(cut_streamline)

    return np.asarray(cut_streamlines)


class TractographyFeatureAnalyzer(object):
    def __init__(self):
        pass


class TractographyChecker(object):
    def __init__(self):

        self._analyzer = None
        self._compliant_indices = {}

    def get_compliant_indices(self):
        idx_list = [val for val in self._compliant_indices.values()]
        return sorted(set(idx_list[0]).intersection(*idx_list))


def compute_streamline_local_orientation(streamlines):
    """Compute local orientation of streamlines. A streamline's local
    orientation is computed by subtracting the coordinate data at consecutive
    locations and normalizing the result. Thus, for each streamline we will
    have a list of vectors whose length will be len(streamline)-1.
    Parameters
    ----------
    streamlines : ArraySequence
        Streamlines
    Returns
    -------
    streamline_local_orientations : list
        Streamline local orientations.
    """

    streamline_local_orientations = []

    for streamline_data in streamlines:

        # Compute the streamlines' local orientation
        dirs = np.diff(streamline_data, axis=0)

        # Normalize segments
        u = unit_vector(dirs)

        # Zero out NaNs
        u = np.nan_to_num(u)

        streamline_local_orientations.append(u)

    return streamline_local_orientations


# ToDo
# This should probably be reformatted to receive a list instead of an
# ArraySequence, and be called from a method that receives the streamlines.
# At that moment, it should be renamed since the streamline_local_orientations
# can be generalized to any list of vectors and put into the `peaks_utils`.
def interpolate_peak_dirs_at_streamline_locations(streamlines, peak_dirs):
    """Interpolate peaks at streamline locations. Will return a list of peaks
    with as many elements as streamlines, each element being an array of length
    size len(streamline)-1 and each atom in the array having as many peaks as
    in peak_dirs.
    Parameters
    ----------
    streamlines : ArraySequence
        Streamlines
    peak_dirs : ndarray (X, Y, Z, Q)
        Peak directions. Q must be a multiple of 3.
    Returns
    -------
    interpolated_peak_dirs : list
        Interpolated peak directions.
    """

    peak_count, spatial_dims = get_peak_count(peak_dirs)

    # Peak orientation values (spatial dimensions) are set to the last
    # dimension (..., P, 3)
    rsh_peak_dirs = reshape_peaks_for_computing(peak_dirs, peak_count, spatial_dims)

    interpolated_peak_dirs = []

    for streamline_data in streamlines:

        # Get indices corresponding to the all but the first point along
        # the streamline.
        # Commenting the int conversion
        # idx = streamline_data[1:, :].astype(int)
        idx = streamline_data[1:, :]

        _interp_peak_dirs = []

        # Loop over each peak
        for i in range(rsh_peak_dirs.shape[-2]):

            # Get peak directions at streamline indices: use a `constant` mode,
            # i.e. fill the past-boundary coordinates with `cval`, and use a 0
            # order spline for the interpolation, i.e. return the peak
            # directions at the `idx` coordinates.
            v = interpolate_volume_at_coordinates(
                rsh_peak_dirs[:, :, :, i, :],
                idx,
                mode="constant",
                order=0,
                cval=0.0,
            )
            # Peak orientation values (spatial dimensions) are set to the last
            # dimension (..., P, 3)
            # v = np.reshape(v, (-1, peak_count, spatial_dims))

            # Normalize peaks
            # v = unit_vector(v)  # assume peaks are normalized

            # Zero out NaNs
            v = np.nan_to_num(v)

            _interp_peak_dirs.append(v)

        interpolated_peak_dirs.append(_interp_peak_dirs)

    return interpolated_peak_dirs


# ToDo
# This should be put into the `math` module maybe after taking outside the
# peak-related reformattings. Maybe generalized and renamed to
# `compute_vector_angle`
# ToDo
# Take the sign into account and take the abs:
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
# ToDo
# The `attribute_straight_angles_to_null_peaks` flag to allow the tests to
# succeed when forcing 90 angles with non-null peaks seems a weak strategy to
# bypass the
# assert (v[_angles == 90.] == 0.).all(), v[_angles == 90.]
# assertion.
#
# Maybe the assert should be changed to something like
#
# # Get booleans as indices of peaks generating 90 angles
# >>> indices = _angles == 90.
# # Set such booleans for all coordinate values at the same index
# >>> indices_rsh = np.repeat(indices, 3).reshape(v.shape)
# >>> v_cp = np.copy(v)
# # Set such peak directions to 0
# >>> v_cp[indices_rsh] = 0
# # Check if all peaks are non zero across the entire peak dirs
# >>> all_peaks_nnzero = v_cp.all()  # not relevant
# # Check if all peaks are 0 across columns
# >>> peaks_zero = ~v_cp.any(axis=1).any()
#
# If the value of `peaks_zero` is True, then, all peaks were zero at some
# position. Should be checked with actual streamlines.
#
def compute_streamline_to_peaks_angles(
    streamline_local_orientations,
    interpolated_peak_dirs,
    attribute_straight_angles_to_null_peaks=True,
):
    """Compute streamline local orientation to peak direction angles. Does not
    take into account the sign.
    Parameters
    ----------
    streamline_local_orientations : list
        Streamline local orientations. Contains as many items as streamlines,
        each item containing an ndarray of shape (N, 3). Usually N is the
        length of the streamline-1.
    interpolated_peak_dirs : list
        Interpolated peak directions. Contains as many items as streamlines,
        each item containing a list of P items, P being the peak count, and
        these items being an ndarray of shape (N, 3).
    attribute_straight_angles_to_null_peaks : bool, optional
        Whether straight angles (90°) are attributed to peak directions being
        null.
    Returns
    -------
    angles : list
        Streamline to peak direction angles.
    """

    angles = []

    for u, v in zip(streamline_local_orientations, interpolated_peak_dirs):

        v_arr = np.asarray(v)

        # Compute the angle between streamline segment directions and all
        # peaks.
        # Note that multiple peaks may exist for each segment.
        # Ensure that NaNs values are set to 0 when normalizing vectors. NaNs
        # can appear (especially for peaks) if normalization is attempted on a
        # 0-valued vector.
        _angles = compute_angle(u, v_arr, use_smallest=True, nan_to_zero=True)

        if attribute_straight_angles_to_null_peaks:
            # Test to make sure that 90° angles are due to null peaks
            if (_angles == 90.0).any():
                pass
                # print('Streamline is out of bounds and so has peaks of 0.')
            assert (v_arr[_angles == 90.0] == 0.0).all(), v_arr[_angles == 90.0]

            assert not np.isnan(_angles).any(), _angles

        angles.append(_angles)

    return angles


def get_closest_peaks_from_angles(angles, peak_dirs):
    """Get the closest peaks by identifying the minimum angle values. Typically
    used to identify the minimum values in streamline (local orientation) to
    peak alignment angle values, and the peaks corresponding to such values.
    Parameters
    ----------
    angles : list
        Angle values. Contains as many items as streamlines, each item
        containing an ndarray of shape (P, N), P being the number of peaks and
        N the number of streamline segments. Usually N is the length of the
        streamline-1.
    peak_dirs : list
        Peak directions values. Contains as many items as streamlines, each
        item containing a list with as many items as peaks (P). Each of these
        lists contains an array of shape (N, 3), N being the number of
        streamline segments. Usually N is the length of the streamline-1.
    Returns
    -------
    closest_angles : list
        Closest angle values. Contains as many items as streamlines, each item
        containing an ndarray of shape (N,), N being the number of streamline
        segments.
    closest_peak_dirs : list
        Peak directions corresponding to the closest angle values. Contains as
        many items as streamlines, each item containing an ndarray of shape
        (N, 3), N being the number of streamline segments.
    """

    # Prefer providing a single implementation for array and ragged array cases
    # (i.e. same vs. different number of segments across streamlines) by
    # avoiding to cast the angles to an array.

    # Get the indices of the smallest angles at each local orientation
    # location
    min_angle_idx = [np.argmin(np.vstack(elem.T), axis=1) for elem in angles]

    ax_1 = [np.arange(angle.shape[1])[None, :] for angle in angles]

    # Get the angles corresponding to the indices
    closest_angles = [
        angle[idx, _ax_1].squeeze()
        for angle, idx, _ax_1 in zip(angles, min_angle_idx, ax_1)
    ]

    # Get the interpolated peak directions corresponding to the indices
    closest_peak_dirs = [
        np.dstack(pks).swapaxes(0, 2).swapaxes(1, 2)[idx, _ax_1].squeeze()
        for pks, idx, _ax_1 in zip(peak_dirs, min_angle_idx, ax_1)
    ]

    return closest_angles, closest_peak_dirs


# ToDo
# Split this: one method to compute the angles, the other one to compute
# the closest one. Otherwise, return the two
# Also, every methods repeats the loop, so chances are that this will be slow.
# best would be to refactor them to accept the atomic variables they need to
# do their computations and have the loop in here.
def compute_local_orientation_alignment2(sft, peak_dirs):
    """Compute streamline local orientation to peak direction angle by
    computing the streamline local (i.e. segment-wise) orientation, iterating
    over each peak index to get the (interpolated) peak value at each segment,
    computing the angle between the interpolated peak and the streamline
    segment, and choosing the peak that is closest to the streamline segment
    orientation (i.e. having the lowest angle value).
    Parameters
    ----------
    sft : StatefulTractogram
        Tractogram.
    peak_dirs : ndarray (X, Y, Z, Q)
        Peak directions. Q must be a multiple of 3.
    Returns
    -------
    Streamline_local_orientations : list
        Streamline local orientations.
    closest_peak_dirs : list
        Closest peak directions.
    closest_angles : list
        Closest angles.
    """

    # Save the spatial attributes so that they can be restored
    origin = sft._origin
    space = sft._space

    # Send streamlines to voxel space
    sft.to_vox()

    # Set streamlines origin to center
    sft.to_center()

    streamlines = sft.get_streamlines_copy()

    # Compute streamline local orientation
    streamline_local_orientations = compute_streamline_local_orientation(streamlines)

    # Peaks need to be interpolated since for each streamline we have a single
    # streamline local orientation at each spatial location but we may have
    # multiple peaks. Thus, we cannot do it the other way round. Also, we are
    # interested in knowing the closest peak to the actual tractography
    # direction, which is given by the streamline direction.
    #
    # Note that the below requires the raw streamline data; the streamline
    # local orientations are a list of normalized vectors, so they cannot be
    # used to interpolated the peak directions.

    # Interpolate peaks at the streamline locations
    interpolated_peak_dirs = interpolate_peak_dirs_at_streamline_locations(
        streamlines, peak_dirs
    )

    # Compute the streamline segment to interpolated peak angles
    angles = compute_streamline_to_peaks_angles(
        streamline_local_orientations, interpolated_peak_dirs
    )

    # Get the closest peaks and the corresponding closest angles
    closest_angles, closest_peak_dirs = get_closest_peaks_from_angles(
        angles, interpolated_peak_dirs
    )

    # Restore the tractogram's spatial attributes
    sft.to_origin(origin)
    sft.to_space(space)

    return (
        streamline_local_orientations,
        closest_peak_dirs,
        closest_angles,
    )


def is_feature_plausible(feature_value, thresholds):
    """Checks whether the values of the features are within the range of values
    to determine whether a streamline is plausible.
    Parameters
    ----------
    feature_value : scalar or array-like
        Streamline-wise values of the given feature.
    thresholds : dict or None
        Minimum ('min') and maximum ('max') or `3rd_quartile` tolerated value
        for the feature. For the first case, any value below the minimum or
        above the maximum will lead to the streamline being considered
        implausible; alternatively, the `3rd_quartile` criterion might be used
        to check the streamline plausibility based on local feature values
        (e.g. local orientation alignment, streamline to surface distance or
        strealine to surface normal alignment): a streamline is considered
        implausible if the 3rd quartile of provided feature values are above
        the given threshold. A scalar `mask_value` can be provided to avoid
        outlier values from introducing a bias in the 3rd quartile computation
        (i.e. a mask is built to include only values strictly smaller than the
        provided mask value). For global features, if no thresholds are
        specified, a boolean `true` value is considered to imply a likely
        plausible streamline (i.e. the streamline is within the WM), a `false`
        value implying a possibly implausible one. For the
    Returns
    -------
    is_plausible : list
        `True` if the value of the feature is within the range of values
        implying that the streamline is likely plausible.
    """

    # Check if threshold values are available for the given feature to check
    # the plausibility of the streamlines.
    if isinstance(feature_value, float) or isinstance(feature_value, bool):
        if thresholds:
            is_plausible = (
                thresholds.get("min", 0)
                <= feature_value
                <= thresholds.get("max", np.inf)
            )
        else:
            # is_plausible = not bool(feature_value)
            is_plausible = feature_value
    elif isinstance(feature_value, np.ndarray) or isinstance(feature_value, list):
        # Deal with global (streamline-wise) features vs. local (segment-wise
        # or endpoint-wise) features: if the first element in the features has
        # no length, assume all features are scalars (e.g. a global feature
        # such as the length); else, there is a collection of features for each
        # streamline, such as the segment-wise fODF peak angle or endpoint to
        # surface distances or angles.
        if not hasattr(feature_value[0], "__len__"):
            if thresholds:
                is_plausible = [
                    thresholds.get("min", 0) <= val <= thresholds.get("max", np.inf)
                    for val in feature_value
                ]
            else:
                # is_plausible = not bool(feature_value)
                is_plausible = feature_value
        else:
            if "max" in thresholds.keys():
                is_plausible = [
                    thresholds.get("min", 0)
                    <= np.max(val)
                    <= thresholds.get("max", np.inf)
                    for val in feature_value
                ]
            elif "3rd_quartile" in thresholds.keys():
                # Note that the below check only works for an upper bound-like
                # case (e.g. local orientation alignment values), i.e. for
                # values that are smaller than the provided (upper bound) mask
                # value, the quantile value is required to be smaller than the
                # provided value. For a lower bound-like case (e.g. PVE map
                # values) the signs should be reversed and the default value
                # should be set to 0.0 instead of infinity.
                is_plausible = [
                    np.quantile(val[val < thresholds.get("mask_value", np.inf)], 0.75)
                    <= thresholds.get("3rd_quartile", np.inf)
                    if len(val[val < thresholds.get("mask_value", np.inf)]) != 0
                    else False
                    for val in feature_value
                ]
            else:
                raise ValueError(
                    "Unsupported assessment mode for a local feature type.\n"
                    f"Found: {thresholds.keys()}; "
                    "Available: min/max and 3rd_quartile thresholding"
                )
    else:
        raise ValueError("Unsupported feature type")

    return is_plausible


# Alignment with the fODF
class StreamlineLocalOrientationAnalyzer(TractographyFeatureAnalyzer):
    def __init__(self):

        super().__init__()

    @staticmethod
    def compute_alignment(sft, peaks, affine):
        # return compute_local_orientation_alignment(streamlines, peaks, affine)  # noqa: E501
        return compute_local_orientation_alignment2(sft, peaks)

    def analyze(self, sft, peaks, affine):
        # fodf_alignment = self.compute_alignment(streamlines, peaks, affine)
        (
            streamline_local_orientations,
            closest_peak_dirs,
            fodf_alignment,
        ) = self.compute_alignment(sft, peaks, affine)

        return fodf_alignment


class StreamlineLocalOrientationChecker(TractographyChecker):
    """Check streamline alignment with the fODFs along its trajectory."""

    def __init__(
        self,
        local_orient_analyzer: StreamlineLocalOrientationAnalyzer,
        allowed_angle=20.0,
        mask_value=90.0,
        allowed_ratio=0.1,
    ):

        super().__init__()

        self._analyzer = local_orient_analyzer
        self._allowed_angle = allowed_angle
        self._mask_value = mask_value
        self._allowed_ratio = allowed_ratio

    def verify_conditions(self, sft, peaks, affine):

        features = self._analyzer.analyze(sft, peaks, affine)

        # Get the indices of streamlines complying with the criteria
        thresholds = dict(
            {
                "3rd_quartile": self._allowed_angle,
                "mask_value": self._mask_value,
            }
        )
        local_orient_compliant_mask = is_feature_plausible(
            features,
            thresholds,
        )
        local_orient_compliant_ids = list(np.where(local_orient_compliant_mask)[0])

        self._compliant_indices = dict(
            {
                StreamlineFeatures.LOCAL_ORIENTATION_ANGLE.name: local_orient_compliant_ids
            }
        )
