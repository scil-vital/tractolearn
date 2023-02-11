# -*- coding: utf-8 -*-

import torch

from tractolearn.config.experiment import DatasetTypes, ExperimentKeys
from tractolearn.learning.dataset import (
    ContrastiveDataset,
    HierarchicalDataset,
    StreamlineClassificationDataset,
    TripletDataset,
)


class DataManager:
    def __init__(self, experiment_dict: dict, seed: int):
        self.experiment_dict = experiment_dict
        self.seed = seed

        # Determine dataset class given the current task
        self.dataset = {
            DatasetTypes.hdf5dataset: StreamlineClassificationDataset,
            DatasetTypes.contrastive: ContrastiveDataset,
            DatasetTypes.triplet: TripletDataset,
            DatasetTypes.hierarchical: HierarchicalDataset,
        }[self.experiment_dict[ExperimentKeys.DATASET_TYPE]]

        (
            self.train_dataset,
            self.valid_dataset,
            self.test_dataset,
        ) = self.setup_dataset()

        self.num_points = (
            self.train_dataset.num_points
        )  # Num points per streamline
        self.point_dims = self.train_dataset.point_dims  # 3

    def setup_dataset(self):

        kwargs = {}
        if self.dataset in [
            ContrastiveDataset,
            TripletDataset,
            HierarchicalDataset,
        ]:
            kwargs.update(
                {"num_pairs": self.experiment_dict["contrastive_num_pairs"]}
            )

        # Note that the seed will be overwritten in the worker_init_fn, see `setup_data_loader`.
        train_dataset = self.dataset(
            self.experiment_dict, "train", self.seed, **kwargs
        )
        test_dataset = self.dataset(
            self.experiment_dict, "test", self.seed, **kwargs
        )
        valid_dataset = self.dataset(
            self.experiment_dict, "thres", self.seed, **kwargs
        )

        return train_dataset, valid_dataset, test_dataset

    def setup_data_loader(self):
        if self.dataset in [
            ContrastiveDataset,
            TripletDataset,
            HierarchicalDataset,
        ]:
            batch_size = None  # The dataset will output whole batches directly
            viz_dataset = self.valid_dataset.dataset
        else:
            batch_size = self.experiment_dict["batch_size"]
            viz_dataset = self.valid_dataset

        # This will ensure that different dataloader workers will sample different streamlines
        def worker_init_fn(worker_id):
            worker_info = torch.utils.data.get_worker_info()
            worker_info.dataset.set_seed(worker_id)

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=self.experiment_dict["num_workers"],
            worker_init_fn=worker_init_fn,
        )

        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            num_workers=self.experiment_dict["num_workers"],
            worker_init_fn=worker_init_fn,
        )

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=self.experiment_dict["num_workers"],
            worker_init_fn=worker_init_fn,
        )

        # For contrastive learning, it does not make sense to use contrastive batches when generating the viz
        viz_loader = torch.utils.data.DataLoader(
            viz_dataset,
            batch_size=self.experiment_dict["batch_size"],
            num_workers=self.experiment_dict["num_workers"],
            worker_init_fn=worker_init_fn,
        )

        return train_loader, valid_loader, test_loader, viz_loader
