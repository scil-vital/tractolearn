# -*- coding: utf-8 -*-
import logging
import pickle
import random

import h5py
import numpy as np
import torch
from dipy.tracking.streamlinespeed import set_number_of_points
from nibabel.streamlines import ArraySequence
from torch.utils.data import Dataset, IterableDataset
from tractolearn.transformation.streamline_transformation import flip_streamlines

from tractolearn.config.experiment import ExperimentKeys

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


class StreamlineClassificationDataset(IterableDataset):
    def __init__(self, experiment_dict: dict, set: str, seed: int):
        self.random_flip = experiment_dict[ExperimentKeys.RANDOM_FLIP]
        self.h5_set_info = experiment_dict["hdf5_path"], set
        self.h5_open = None

        with h5py.File(self.h5_set_info[0], mode="r") as f:
            self.classes = list(f[self.h5_set_info[1]].keys())
            self.n_in_class = [
                f[self.h5_set_info[1]][c]["streamline"].shape[0] for c in self.classes
            ]
            self.num_points = f[self.h5_set_info[1]][self.classes[0]][
                "streamline"
            ].shape[1]
            self.point_dims = f[self.h5_set_info[1]][self.classes[0]][
                "streamline"
            ].shape[2]

        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def get_random_streamline_from_class(self, class_idx):
        if self.h5_open is None:
            h5_set = h5py.File(self.h5_set_info[0], mode="r")[self.h5_set_info[1]]

        i = self.rng.integers(
            self.n_in_class[class_idx]
        )  # Pick one streamline in that class
        x = h5_set[self.classes[class_idx]]["streamline"][i]

        if self.random_flip:
            p = random.uniform(0, 1)

            if p > 0.5:
                x = flip_streamlines(ArraySequence([x])).get_data()
        x = torch.from_numpy(x).transpose(1, 0)

        if torch.any(torch.isnan(x)):
            print("hi")

        return x

    def __next__(self):
        class_idx = self.rng.integers(len(self.classes))  # Pick a random class
        x = self.get_random_streamline_from_class(class_idx)
        return x, torch.tensor(class_idx)


class ContrastiveDataset(IterableDataset):
    """This dataset returns batches, not items. Should be used with batch_size=None"""

    def __init__(self, experiment_dict: dict, type_set: str, seed, num_pairs):
        self.dataset = StreamlineClassificationDataset(experiment_dict, type_set, seed)
        self.num_pairs = num_pairs  # e.g. if this value is 4, there will be 4 positive pairs and 4 negative pairs

    @property
    def rng(self):
        return self.dataset.rng

    def set_seed(self, seed):
        self.dataset.rng = np.random.default_rng(seed)

    @property
    def num_points(self):
        return self.dataset.num_points

    @property
    def point_dims(self):
        return self.dataset.point_dims

    def __iter__(self):
        return self

    def __next__(self):
        """Returns a batch containing 4 consecutive equal length sections:

        1. The first elements of positive pairs
        2. The second elements of positive pairs
        3. The first elements of negative pairs
        4. The second elements of negative pairs

        If we write those as x_pos1, x_pos2, x_neg1, x_neg2; we will then be able to compute the pairwise distances
        using:
        z_* = encode(x_*)
        pos_distances = z_pos1 - z_pos2
        neg_distances = z_neg1 - z_neg2

        Since it's just a batch of streamlines, we can pass it through the encoder as-is.
        """

        # Positive pairs
        positive1, positive2 = [], []
        for _ in range(self.num_pairs):
            k = self.rng.choice(len(self.dataset.classes))
            x1 = self.dataset.get_random_streamline_from_class(k)
            x2 = self.dataset.get_random_streamline_from_class(k)
            positive1.append(x1)
            positive2.append(x2)
        positive = np.stack(positive1 + positive2)

        # Negative pairs
        negative1, negative2 = [], []
        for _ in range(self.num_pairs):
            k1, k2 = self.rng.choice(len(self.dataset.classes), size=2, replace=False)
            x1 = self.dataset.get_random_streamline_from_class(k1)
            x2 = self.dataset.get_random_streamline_from_class(k2)
            negative1.append(x1)
            negative2.append(x2)
        negative = np.stack(negative1 + negative2)

        samples = torch.from_numpy(
            np.concatenate((positive, negative))
        )  # (n_pos_pairs * 2 + n_neg_pairs * 2, 3, 256)
        return samples


class StreamlineClassificationDatasetTree(IterableDataset):
    def __init__(self, experiment_dict: dict, set: str, seed: int):
        self.random_flip = experiment_dict[ExperimentKeys.RANDOM_FLIP]

        with open(experiment_dict["pickle_path"], "rb") as f:
            data = pickle.load(f)
            self.point_dims = 3
            self.num_points = 256
            self.data = data[set]["clusters"]

        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def get_random_streamline_from_class_with_merge(self, class_idx):

        i = self.rng.integers(self.n_in_class[class_idx])
        j = self.rng.integers(
            self.n_in_class[class_idx]
        )  # Pick one streamline in that class

        merged_cluster = self.data["merged_cluster_map"]
        clusters = self.data["clusters"]
        merged_streamlines = merged_cluster[class_idx].indices

        anchor_id = merged_streamlines[i]
        anchor = merged_cluster[class_idx][i]
        positive = merged_cluster[class_idx][j]

        if self.random_flip:
            p = random.uniform(0, 1)

            if p > 0.5:
                anchor = flip_streamlines(ArraySequence([anchor])).get_data()

        if self.random_flip:
            p = random.uniform(0, 1)

            if p > 0.5:
                positive = flip_streamlines(ArraySequence([positive])).get_data()

        anchor = set_number_of_points(anchor, 256)
        positive = set_number_of_points(positive, 256)
        anchor = torch.from_numpy(anchor).transpose(1, 0)
        positive = torch.from_numpy(positive).transpose(1, 0)

        cluster = [
            idx
            for idx, i in enumerate(clusters.get_clusters(4))
            for j in i
            if j == anchor_id
        ][0]

        previous_parent = [i for i in merged_streamlines]
        parent = clusters.get_clusters(4)[cluster].parent
        ids = {}
        streamlines = []
        for i in reversed(range(1, 5)):
            parent_ids = parent.indices
            parent_ids = [i for i in parent_ids if i not in previous_parent]

            if len(parent_ids) == 0:
                ids[i] = None
                streamline = anchor

            else:
                current_id = self.rng.integers(len(parent_ids))
                ids[i] = parent_ids[current_id]
                current_id_parent = [
                    idx
                    for idx, i in enumerate(parent.indices)
                    if i == parent_ids[current_id]
                ][0]

                streamline = parent[current_id_parent]

                if self.random_flip:
                    p = random.uniform(0, 1)

                    if p > 0.5:
                        streamline = flip_streamlines(
                            ArraySequence([streamline])
                        ).get_data()

                streamline = set_number_of_points(streamline, 256)
                streamline = torch.from_numpy(streamline).transpose(1, 0)
            streamlines.append(streamline)
            previous_parent.extend(parent_ids)
            parent = parent.parent

        streamlines.insert(0, anchor)
        streamlines.insert(1, positive)

        streamlines = torch.stack(streamlines)

        return streamlines

    def get_random_streamline_from_class_without_merge(self):

        clusters = self.data

        indices = []
        child = clusters.get_clusters(0)[0].children
        is_leaf = False

        ids = {}

        i = 0

        while not is_leaf:
            possible_idx = list(range(len(child)))
            idx = self.rng.choice(possible_idx)

            indices.append(idx)
            is_leaf_negative = False
            del possible_idx[idx]
            negative_child = None
            n_id = None
            while not is_leaf_negative:
                if len(possible_idx) == 0:
                    n_id = None
                    break
                else:
                    n_id = "Negative ID not None"

                n_idx = self.rng.choice(possible_idx)
                if negative_child is None:
                    negative_child = child[n_idx].children
                else:
                    negative_child = negative_child[n_idx].children

                is_leaf_negative = negative_child[0].is_leaf
                possible_idx = list(range(len(negative_child)))

            if n_id is None:
                ids[i] = n_id
            else:
                n_i = self.rng.integers(len(negative_child))
                n_ii = self.rng.integers(len(negative_child[n_i].indices))
                ids[i] = negative_child[n_i].indices[n_ii]

            child = child[idx].children
            is_leaf = child[0].is_leaf

            i += 1

        possible_idx = list(range(len(child)))
        class_idx = self.rng.choice(possible_idx)
        del possible_idx[class_idx]

        i = self.rng.integers(len(child[class_idx]))
        j = self.rng.integers(len(child[class_idx]))

        anchor_id = child[class_idx][i]
        positive_id = child[class_idx][j]

        if len(possible_idx) == 0:
            less_positive_id = anchor_id
        else:
            negative_idx = self.rng.choice(possible_idx)
            k = self.rng.integers(len(child[negative_idx]))
            less_positive_id = child[negative_idx][k]

        anchor = clusters.refdata[anchor_id]
        positive = clusters.refdata[positive_id]
        less_positive = clusters.refdata[less_positive_id]

        streamlines = [anchor, positive, less_positive]

        for i in reversed(list(ids.keys())):
            if ids[i] is None:
                streamlines.append(anchor)
            else:
                streamlines.append(clusters.refdata[ids[i]])

        sample = []
        for s in streamlines:
            if self.random_flip:
                p = random.uniform(0, 1)

                if p > 0.5:
                    s = flip_streamlines(ArraySequence([s])).get_data()

            s = set_number_of_points(s, 256)
            s = torch.from_numpy(s).transpose(1, 0)
            sample.append(s)

        streamlines = torch.stack(sample)

        return streamlines

    def __next__(self):
        x = self.get_random_streamline_from_class_without_merge()
        return x[0], torch.tensor(0)


class HierarchicalDataset(IterableDataset):
    def __init__(self, experiment_dict: dict, type_set: str, seed: int, num_pairs: int):
        self.dataset = StreamlineClassificationDatasetTree(
            experiment_dict, type_set, seed
        )
        self.num_pairs = num_pairs  # e.g. if this value is 4, there will be 4 positive pairs and 4 negative pairs

    @property
    def rng(self):
        return self.dataset.rng

    def set_seed(self, seed):
        self.dataset.rng = np.random.default_rng(seed)

    @property
    def num_points(self):
        return self.dataset.num_points

    @property
    def point_dims(self):
        return self.dataset.point_dims

    def __iter__(self):
        return self

    def __next__(self):
        anchors, level1, level2, level3, level4, negatives = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for _ in range(self.num_pairs):
            # k = self.rng.choice(
            #     len(self.dataset.classes_clusters), replace=False
            # )
            streamlines = self.dataset.get_random_streamline_from_class_without_merge()

            anchors.append(streamlines[0])
            level1.append(streamlines[1])
            level2.append(streamlines[2])
            level3.append(streamlines[3])
            level4.append(streamlines[4])
            negatives.append(streamlines[5])
        samples = torch.from_numpy(
            np.stack(anchors + level1 + level2 + level3 + level4 + negatives)
        )  # (anchors + level1 + level2 + level3 + level4 + negatives, 3, 256)

        return samples


class TripletDataset(ContrastiveDataset):
    """This dataset returns batches, not items. Should be used with batch_size=None"""

    def __init__(self, experiment_dict: dict, type_set: str, seed, num_pairs):
        super().__init__(experiment_dict, type_set, seed, num_pairs)

    def __next__(self):
        """Returns a batch containing 3 consecutive equal length sections:

        1. Anchors
        2. Positive values
        3. Negative values

        """

        anchors, positives, negatives = [], [], []
        for _ in range(self.num_pairs):
            k1, k2 = self.rng.choice(len(self.dataset.classes), size=2, replace=False)
            a = self.dataset.get_random_streamline_from_class(k1)
            p = self.dataset.get_random_streamline_from_class(k1)
            n = self.dataset.get_random_streamline_from_class(k2)

            if (
                torch.any(torch.isnan(a))
                or torch.any(torch.isnan(p))
                or torch.any(torch.isnan(n))
            ):
                print("hi")

            anchors.append(a)
            positives.append(p)
            negatives.append(n)
        samples = torch.from_numpy(
            np.stack(anchors + positives + negatives)
        )  # (anchors + positives + negatives, 3, 256)

        return samples
