# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("root")


class OnTheFlyDataset(Dataset):
    def __init__(self, X: np.array, y: np.array, to_transpose=True):
        self.data = list(zip(X, y))
        self.to_transpose = to_transpose

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        X, y = self.data[item]

        X = X.astype(np.float32)
        y = y.astype(np.float64)

        if self.to_transpose:
            X = torch.from_numpy(X).transpose(1, 0)
        else:
            X = torch.from_numpy(X)
        y = torch.from_numpy(np.array(y))

        return X, y
