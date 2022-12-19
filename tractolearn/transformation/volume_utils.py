# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np


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
