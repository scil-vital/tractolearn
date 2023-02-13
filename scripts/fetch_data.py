#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from tractolearn.tractoio.dataset_fetch import (
    Dataset,
    provide_dataset_description,
    retrieve_dataset,
)


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description="Fetch tractolearn dataset.\n\nDetails (name: description: URL):\n\t"
        + "\t".join(provide_dataset_description()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dastaset_name",
        type=Dataset.argparse,
        choices=list(Dataset),
        help="Dataset name",
    )
    parser.add_argument(
        "out_path",
        type=Path,
        help="Output path",
    )

    return parser


def main():

    # Parse arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    _ = retrieve_dataset(Dataset(args.dastaset_name).name, args.out_path)


if __name__ == "__main__":
    main()
