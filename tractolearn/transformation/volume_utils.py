# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


def compute_isocenter(img):
    """Computes the iso-center of a volumetric image in RAS space (mm).

    Parameters
    ----------
    img : nibabel.nifti1.Nifti1Image
        NIfTI-1 Image.

    Returns
    -------
    isocenter : ndarray
        Iso-center of the input image.
    """

    voxel_isocenter = (np.array(img.get_fdata().shape) - 1) / 2.0
    isocenter = nib.affines.apply_affine(img.affine, voxel_isocenter)
    return isocenter


def compute_volume(img):
    """Computes the volume of an input image in mm^3.

    Parameters
    ----------
    img : nibabel.nifti1.Nifti1Image
        NIfTI-1 Image.

    Returns
    -------
    float
        Volume of the input image.
    """

    image_size = img.header.get_data_shape()
    voxel_size = img.header.get_zooms()

    return np.array(image_size) * np.array(voxel_size)


def interpolate_volume_at_coordinates(
    volume: np.ndarray,
    coords: np.ndarray,
    mode: str = "nearest",
    order: int = 3,
    cval=0.0,
) -> np.ndarray:
    """Evaluates a 3D or 4D volume data at the given coordinates by
    interpolation.

    Parameters
    ----------
    volume : 3D array or 4D array
        Data volume.
    coords : ndarray of shape (N, 3)
        3D coordinates where to evaluate the volume data.
    mode : str, optional
        Points outside the boundaries of the input are filled according to the
        given mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’).
        ('constant' uses 0.0 as a points outside the boundary)
    order : int, optional
        The order of the spline interpolation.
    cval : float, optional
        Value used to fill past-boundary coordinates.

    Returns
    -------
    output : 2D array
        Values from volume.
    """

    if volume.ndim <= 2 or volume.ndim >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.ndim == 3:
        return map_coordinates(volume, coords.T, order=order, mode=mode, cval=cval)

    if volume.ndim == 4:
        last_dim = volume.shape[-1]
        values_4d = np.zeros((coords.shape[0], last_dim))

        for i in range(volume.shape[-1]):
            values_4d[:, i] = map_coordinates(
                volume[..., i], coords.T, order=order, mode=mode, cval=cval
            )

        return values_4d
