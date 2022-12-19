# -*- coding: utf-8 -*-

import numpy as np


class PredictWrapper:
    def __init__(self, fn):
        self.fn = fn

    def predict(self, x):
        # Note 1: We need to pass the result to the CPU so that the latent
        # distance informer class can deal with the NumPy array.
        # Note 2: Due to the Keras/PyTorch asymmetry, and the [0] index in
        # self._encoder.predict(
        #        streamline[np.newaxis, ])[0]
        # in TractographyLatentSpaceDistanceInformer.compute_distances
        # we need to add yet another axis here: [np.newaxis, ]
        return (
            self.fn(x)
            .cpu()
            .detach()
            .numpy()[
                np.newaxis,
            ]
        )
