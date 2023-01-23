# -*- coding: utf-8 -*-

from enum import Enum


class BundlesAdditionalLabels(Enum):
    hdf5_invalid_class = 0
    invalid_connection_class = 100
    interpolated_class = 200
    reference_class = 125
    generic_streamline_class = 150
    nearest_neighbor_class = 175

    invalid_connection_class_name = "implausible"
    interpolated_class_name = "interpolated"

    reference_class_name = "reference"
    generic_streamline_class_name = "streamline"
    nearest_neighbor_class_name = "nearest_neighbor"
