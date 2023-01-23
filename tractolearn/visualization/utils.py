# -*- coding: utf-8 -*-

import enum

import numpy as np
from dipy.tracking.streamline import transform_streamlines


class AnatomicalView(enum.Enum):
    AXIAL_INFERIOR = "axial_inferior"
    AXIAL_SUPERIOR = "axial_superior"
    CORONAL_ANTERIOR = "coronal_anterior"
    CORONAL_POSTERIOR = "coronal_posterior"
    SAGITTAL_LEFT = "sagittal_left"
    SAGITTAL_RIGHT = "sagittal_right"


def transform_streamlines_according_to_target(
    streamlines, target_template_img, transformation
):
    """."""

    transformed_streamlines = transform_streamlines(
        streamlines, np.linalg.inv(transformation)
    )

    return transformed_streamlines
