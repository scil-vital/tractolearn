# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import tomli
import pkg_resources

from datetime import datetime
# sys.path.insert(0, os.path.abspath('.'))

# Load the release info into a dict by explicit execution
with open(os.path.join("..", "pyproject.toml"), "rb") as f:
    info = tomli.load(f)

# -- Project information -----------------------------------------------------

project = info["project"]["name"]
_author = info["project"]["authors"][0]["name"]
_email = info["project"]["authors"][1]["email"]
copyright = f"2022-{datetime.now().year}, {_author}s <{_email}s>"
author = f"{_author}s"

_version = pkg_resources.require(project)[0].version
# The full version, including alpha/beta/rc tags
release = _version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Sources
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
