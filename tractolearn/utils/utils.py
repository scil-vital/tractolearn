# -*- coding: utf-8 -*-

import datetime
import os
from os.path import join as pjoin
from pathlib import Path
from uuid import uuid4


def make_run_dir(out_path=None):
    """Create a directory for this training run"""
    run_name = generate_uuid()
    if out_path is None:
        root_output_path = Path(os.environ.get("OUTPUT_PATH", "."))
    else:
        root_output_path = out_path
    run_dir = Path(pjoin(root_output_path, run_name))
    run_dir.mkdir()
    return run_dir


def generate_uuid():

    eventid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f-") + str(uuid4())
    return eventid
