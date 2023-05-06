# -*- coding: utf-8 -*-

import enum

fname_sep = "."


class DictDataExtensions(enum.Enum):
    JSON = "json"


class TractogramExtensions(enum.Enum):
    TCK = "tck"
    TRK = "trk"
