# -*- coding: utf-8 -*-

# ToDo
# Accept a Nifti1Image as well ?
# Accept an array that has not been formatted for visualization?
import numpy as np


def get_peak_count(peak_dirs):
    """Get the peak count per spatial location. All locations are expected to
    have the same peak count. Peaks are assumed to have 3 spatial coordinates
    (x, y, z).
    Parameters
    ----------
    peak_dirs : ndarray (X, Y, Z, Q)
        The peak directions. Q must be a multiple of 3.
    """

    assert len(peak_dirs.shape) == 4

    _, _, _, last_dim = peak_dirs.shape

    # Peaks are assumed to have 3 spatial coordinates (x,y,z)
    spatial_dims = 3

    assert last_dim % spatial_dims == 0

    peak_count = last_dim // spatial_dims

    return peak_count, spatial_dims


def reshape_peaks_for_computing(peak_dirs, peak_count, spatial_dims):
    """Reshape the peak dirs array so that the volume extension indices remain
    unchanged in the first three dimensions; the peak count indices are made
    the last but one dimension, and the peak orientation values are made the
    last dimension, typically (X, Y, Z, P, 3), where P is the number of peaks
    per voxel.
    See also dipy.direction.peaks.reshape_peaks_for_visualization
    Parameters
    ----------
    peak_dirs : ndarray (X, Y, Z, Q) (4D)
        Peak directions. Q must be a multiple of 3.
    peak_count : int
        Peak count.
    spatial_dims : int
        Spatial dimensions of peaks. Typically 3 (x, y, z).
    Returns
    -------
    ndarray
        Reshaped peak directions.
    """

    # Peak orientation values (spatial dimensions) are set to the last
    # dimension (..., P, 3)
    unfold_peak_dims = np.hstack(
        [peak_dirs.shape[:3], peak_count, spatial_dims]
    )

    return np.reshape(peak_dirs, unfold_peak_dims)
