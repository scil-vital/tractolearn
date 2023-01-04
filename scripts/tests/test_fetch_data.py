#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile
from os import listdir
from os.path import isfile, join


def test_help_option(script_runner):

    ret = script_runner.run(
        "fetch_data.py", "--help"
    )
    assert ret.success


def test_execution(script_runner):

    # Test the lightest datasets
    with tempfile.TemporaryDirectory() as tmp_dir:

        os.chdir(os.path.expanduser(tmp_dir))

        ret = script_runner.run(
            "fetch_data.py",
            "contrastive_autoencoder_weights",
            tmp_dir)

        assert ret.success

        files = [f for f in listdir(tmp_dir) if isfile(join(tmp_dir, f))]
        assert len(files) == 1

    with tempfile.TemporaryDirectory() as tmp_dir:

        os.chdir(os.path.expanduser(tmp_dir))

        ret = script_runner.run(
            "fetch_data.py",
            "mni2009cnonlinsymm_anat",
            tmp_dir)

        assert ret.success

        files = [f for f in listdir(tmp_dir) if isfile(join(tmp_dir, f))]
        assert len(files) == 1
