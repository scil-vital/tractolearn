# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class LatentSpaceDistanceInformer:
    def __init__(self, encoder, ref_latent_space_samples, num_neighbors=1):
        self._encoder = encoder
        # The default metric is the `minkowski` distance, which is a
        # generalization of both the Euclidean distance (`p=2`) and the
        # Manhattan distance (`p=1`). The default value is `p=2`.
        self._nearest_classifier = NearestNeighbors(
            n_neighbors=num_neighbors, metric="minkowski", n_jobs=-1
        )
        self._nearest_classifier.fit(ref_latent_space_samples)

    def compute_distance_on_latent(self, projected_latent_space_sample):
        (
            sample_distances,
            sample_nearest_indices,
        ) = self._nearest_classifier.kneighbors(
            projected_latent_space_sample, return_distance=True
        )
        distances = sample_distances.squeeze()
        nearest_indices = sample_nearest_indices.squeeze()

        return (
            np.array(distances),
            np.array(nearest_indices),
        )

    def compute_distances(self, streamlines, batch=128):
        projected_latent_space_samples = []
        distances = []
        nearest_indices = []
        streamlines = torch.split(streamlines, batch)

        for streamline in tqdm(streamlines):
            projected_latent_space_sample = self._encoder.predict(streamline)[0]

            (
                sample_distances,
                sample_nearest_indices,
            ) = self._nearest_classifier.kneighbors(
                projected_latent_space_sample, return_distance=True
            )

            projected_latent_space_samples.append(
                projected_latent_space_sample.squeeze()
            )
            distances.append(sample_distances.squeeze())
            nearest_indices.append(sample_nearest_indices.squeeze())

        return (
            np.concatenate(projected_latent_space_samples),
            np.concatenate(distances),
            np.concatenate(nearest_indices),
        )
