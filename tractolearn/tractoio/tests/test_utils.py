#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile

from tractolearn.tractoio.file_extensions import (
    DictDataExtensions,
    TractogramExtensions,
    fname_sep,
)
from tractolearn.tractoio.utils import (
    compose_filename,
    filter_filenames,
    identify_missing_bundle,
    identify_missing_tractogram,
    read_data_from_json_file,
    save_data_to_json_file,
)


def test_identify_missing_bundle(tmp_path):

    with tempfile.NamedTemporaryFile(
        suffix=fname_sep + DictDataExtensions.JSON.value, dir=tmp_path
    ) as f:

        # Target bundle names
        bundle_names = ["CC_Fr_1", "CST_L", "AC"]

        bundle_data = dict({"CC_Fr_1": 1.0, "CST_L": 2.0, "AC": 3.0})
        expected = sorted(set(bundle_names).difference(bundle_data.keys()))

        save_data_to_json_file(bundle_data, f.name)

        data = read_data_from_json_file(
            os.path.join(tmp_path, os.listdir(tmp_path)[0])
        )

        obtained = identify_missing_bundle(data, bundle_names)

        assert obtained == expected

    with tempfile.NamedTemporaryFile(
        suffix=fname_sep + DictDataExtensions.JSON.value, dir=tmp_path
    ) as f:

        # Target bundle names
        bundle_names = ["Cu", "PrCu"]

        bundle_data = dict({"Cu": 2.0})
        expected = sorted(set(bundle_names).difference(bundle_data.keys()))

        save_data_to_json_file(bundle_data, f.name)

        data = read_data_from_json_file(
            os.path.join(tmp_path, os.listdir(tmp_path)[0])
        )

        obtained = identify_missing_bundle(data, bundle_names)

        assert obtained == expected


def test_identify_missing_tractogram(tmp_path):

    # Target bundle names
    bundle_names = ["CC_Fr_1", "CST_L", "AC"]

    # Create some files in the temporary path
    file_rootnames = ["CC_Fr_1", "CC_Fr_2", "AC"]
    fnames = [
        compose_filename(tmp_path, val, TractogramExtensions.TRK.value)
        for val in file_rootnames
    ]
    [open(val, "w") for val in fnames]

    expected = sorted(set(bundle_names).difference(file_rootnames))

    obtained = identify_missing_tractogram(tmp_path, bundle_names)

    assert obtained == expected

    # Target bundle names
    bundle_names = ["Cu"]
    expected = [
        compose_filename(tmp_path, val, TractogramExtensions.TRK.value)
        for val in bundle_names
    ]

    # Create some files in the temporary path
    file_rootnames = ["Cu", "PrCu"]
    fnames = [
        compose_filename(tmp_path, val, TractogramExtensions.TRK.value)
        for val in file_rootnames
    ]
    [open(val, "w") for val in fnames]

    expected = sorted(set(bundle_names).difference(file_rootnames))

    obtained = identify_missing_tractogram(tmp_path, bundle_names)

    assert obtained == expected


def test_filter_fnames(tmp_path):

    # Target bundle names
    bundle_names = ["CC_Fr_1", "CST_L", "AC"]

    # Create some files in the temporary path
    file_rootnames = ["CC_Fr_1", "CC_Fr_2", "AC"]
    fnames = [
        compose_filename(tmp_path, val, TractogramExtensions.TRK.value)
        for val in file_rootnames
    ]
    [open(val, "w") for val in fnames]

    expected_rootnames = ["AC", "CC_Fr_1"]
    expected = [
        compose_filename(tmp_path, val, TractogramExtensions.TRK.value)
        for val in expected_rootnames
    ]

    obtained = filter_filenames(tmp_path, bundle_names)

    assert obtained == expected

    # Target bundle names
    bundle_names = ["Cu"]
    expected = [
        compose_filename(tmp_path, val, TractogramExtensions.TRK.value)
        for val in bundle_names
    ]

    # Create some files in the temporary path
    file_rootnames = ["Cu", "PrCu"]
    fnames = [
        compose_filename(tmp_path, val, TractogramExtensions.TRK.value)
        for val in file_rootnames
    ]
    [open(val, "w") for val in fnames]

    expected_rootnames = ["Cu"]
    expected = [
        compose_filename(tmp_path, val, TractogramExtensions.TRK.value)
        for val in expected_rootnames
    ]

    obtained = filter_filenames(tmp_path, bundle_names)

    assert obtained == expected
