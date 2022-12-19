#!/usr/bin/env python
"""distutils / setuptools helpers.
"""


class Bunch(object):
    def __init__(self, info):
        for key, name in info.items():
            if key.startswith("__"):
                continue
            self.__dict__[key] = name


def read_vars_from(info_file):
    """Read variables from Python text file.

    Parameters
    ----------
    info_file : str
        Filename of file to read.

    Returns
    -------
    info_vars : Bunch instance
        Bunch object where variables read from `info_file` appear as
        attributes.
    """

    # Use exec for compatibility with Python 3
    info = {}
    with open(info_file, "rt") as fobj:
        exec(fobj.read(), info)

    return Bunch(info)
