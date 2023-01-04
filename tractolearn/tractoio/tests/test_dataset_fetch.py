#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from tractolearn.tractoio.dataset_fetch import Dataset, retrieve_dataset


def test_retrieve_dataset(tmp_path):

    for name in Dataset.__members__.keys():
        file = retrieve_dataset(name, tmp_path)

        assert os.path.isfile(file)
