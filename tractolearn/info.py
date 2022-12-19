"""This file contains the parameters that are used to fill the settings in
tractolearn's `setup.py`, to provide tractolearn's top-level docstring, and to
build the docs.
"""

# tractolearn version information. An empty _version_extra corresponds to a
# full release. '.dev' as a _version_extra string means this is a development
# version
_version_major = 0
_version_minor = 1
_version_micro = 0
_version_extra = "dev0"
# _version_extra = ''

# Format expected by setup.py and doc/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s.%s" % (
    _version_major,
    _version_minor,
    _version_micro,
    _version_extra,
)

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Neuroimaging",
]

description = "Tractography learning."

keywords = "DL dMRI ML neuroimaging tractography"

# Main setup parameters
NAME = "tractolearn"
MAINTAINER = "jhlegarreta"
MAINTAINER_EMAIL = ""
DESCRIPTION = description
URL = "https://github.com/jhlegarreta/tractolearn"
DOWNLOAD_URL = ""
BUG_TRACKER = ("https://github.com/scil-vital/tractolearn/tractolearn/issues",)
DOCUMENTATION = ("https://tractolearn.readthedocs.io/en/latest/",)
SOURCE_CODE = ("https://github.com/scil-vital/tractolearn/tractolearn",)
LICENSE = ""
CLASSIFIERS = classifiers
KEYWORDS = keywords
AUTHOR = "jhlegarreta"
AUTHOR_EMAIL = "jon.haitz.legarreta@gmail.com"
PLATFORMS = ""
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
ISRELEASE = _version_extra == ""
VERSION = __version__
PROVIDES = ["tractolearn"]
REQUIRES = ["dipy", "nibabel", "h5py"]
