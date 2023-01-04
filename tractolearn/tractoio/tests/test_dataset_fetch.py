#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from tractolearn.tractoio.dataset_fetch import Dataset, retrieve_dataset


def test_retrieve_dataset(tmp_path):

    dataset_keys = list(Dataset.__members__.keys())
    # Exclude attempting to download the HDF% file due to its size
    dataset_keys.remove(Dataset.TRACTOINFERNO_HCP_REF_TRACTOGRAPHY.name)

    for name in dataset_keys:
        files = retrieve_dataset(name, tmp_path)

        if isinstance(files, str):
            assert os.path.isfile(files)
        elif isinstance(files, list):
            for elem in files:
                assert os.path.isfile(elem)
        else:
            raise TypeError("Unexpected type found.")
