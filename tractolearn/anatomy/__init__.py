# -*- coding: utf-8 -*-

from enum import Enum


class BundleSettings(Enum):
    BUNDLE_DICTIONARIES = "bundles_dictionaries.json"


class Tissue(Enum):
    CSF = "CSF"
    GM = "GM"
    WM = "WM"
