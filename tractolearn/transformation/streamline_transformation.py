# -*- coding: utf-8 -*-

import copy
import logging

import nibabel as nib
import numpy as np
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.metrics import downsample, length

from tractolearn.tractomath.utils import is_normalized

logger = logging.getLogger(__name__)


def flip_streamlines(streamlines):
    """Flips streamlines by reversing the array ordering. Note that both the
    head/tails and the streamline sorting are reversed. To reverse only the
    heads/tails use `reverse_head_tails;` to reverse only the streamline
    sorting use, `reverse_streamline_sorting`.

    Parameters
    ----------
    streamlines : nib.streamlines.ArraySequence
        Streamlines to be flipped.

    Returns
    -------
    flipped_streamlines : nib.streamlines.ArraySequence
        Flipped streamlines.
    """

    flipped_streamlines = nib.streamlines.ArraySequence()
    flipped_streamlines._data = np.flip(streamlines.get_data(), 0)
    flipped_streamlines._lengths = np.flip(streamlines._lengths, 0)
    flipped_streamlines._offsets = (
        np.cumsum(flipped_streamlines._lengths) - flipped_streamlines._lengths
    )

    return flipped_streamlines


def flip_random_streamlines(streamlines, ratio=0.5):
    """Flips streamlines randomly by reversing the array ordering.

    Parameters
    ----------
    streamlines : nib.streamlines.ArraySequence
        Streamlines to be flipped.
    ratio : float, optional
        Ratio of the streamlines to be flipped. Must be in the range [0..1].

    Returns
    -------
    flipped_streamlines : nib.streamlines.ArraySequence
        Flipped streamlines.
    streamline_flip_indices : ndarray
        Flipped streamline indices.
    """

    if not is_normalized(ratio):
        raise ValueError(
            "`ratio` must be normalized (i.e. in the range "
            "[0..1].\nFound: {}".format(ratio)
        )

    flipped_streamlines = copy.deepcopy(streamlines)

    num_streamlines = len(flipped_streamlines._offsets)
    size = int(np.floor(num_streamlines * ratio))
    streamline_flip_indices = np.random.randint(0, num_streamlines, size)

    for i in streamline_flip_indices:
        num_points = flipped_streamlines._lengths[i]
        flipped_streamline = flip_streamlines(
            nib.streamlines.ArraySequence([streamlines[i]])
        )
        flipped_streamlines._data[
            flipped_streamlines._offsets[i] : flipped_streamlines._offsets[i]
            + num_points
        ] = flipped_streamline.get_data()

    return flipped_streamlines, streamline_flip_indices


def resample_streamlines(streamlines, num_points, arc_length=True):
    """Resamples streamlines to a number of points. If arc length
    parameterization is used, resampled streamlines will have equal length
    segments.

    Parameters
    ----------
    streamlines : nib.streamlines.ArraySequence
        Streamlines to be resampled.
    num_points : int
        Number of points for the resampled bundles.
    arc_length : bool, optional
        If True, use arc length parameterization resampling.

    Returns
    -------
    resampled_streamlines : nib.streamlines.ArraySequence
        Resampled streamlines.
    """

    num_streamlines = len(streamlines._lengths)
    max_streamlines = num_streamlines

    lengths = np.zeros(num_streamlines)
    for i in np.arange(num_streamlines):
        lengths[i] = length(streamlines[i])

    ind = list(range(0, num_streamlines))

    resampled_streamline_points = []
    min_length = 0.0
    max_length = 0.0

    while len(ind) > 0 and len(resampled_streamline_points) < max_streamlines:
        i = ind.pop()
        if lengths[i] >= min_length and (max_length <= 0.0 or lengths[i] <= max_length):
            if num_points:
                if arc_length:
                    line = set_number_of_points(streamlines[i], num_points)
                else:
                    line = downsample(streamlines[i], num_points)
                resampled_streamline_points.append(line)
            else:
                resampled_streamline_points.append(streamlines[i])

    resampled_streamlines = nib.streamlines.ArraySequence()
    resampled_streamlines._data = np.vstack(resampled_streamline_points)
    resampled_streamlines._lengths = np.array(
        [num_points for _ in range(num_streamlines)]
    )
    resampled_streamlines._offsets = np.array(
        [i * num_points for i in range(num_streamlines)]
    )

    return resampled_streamlines


def sft_voxel_transform(sft):
    sft.to_vox()
    sft.to_corner()
