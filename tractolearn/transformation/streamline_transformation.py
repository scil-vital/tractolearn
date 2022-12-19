# -*- coding: utf-8 -*-

import logging

import nibabel as nib
import numpy as np

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
