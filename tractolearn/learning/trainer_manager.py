# -*- coding: utf-8 -*-
import itertools
import logging
import sys
from os.path import join as pjoin
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from comet_ml import Experiment
from dipy.tracking.streamlinespeed import Streamlines
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from tractolearn.logger import LoggerKeys
from tractolearn.config.experiment import DatasetTypes, ExperimentKeys
from tractolearn.models.forward import make_forward
from tractolearn.models.model_performance_history import LossHistory
from tractolearn.models.model_pool import get_model
from tractolearn.tractoio.utils import save_loss_history
from tractolearn.utils.processing_utils import postprocess
from tractolearn.visualization.tractogram_visualization import (
    plot_tractogram_with_anat_ref,
)

logger = logging.getLogger("root")


class Trainer:
    best_model_fname = "best_model.pt"

    def __init__(
        self,
        experiment_dict: dict,
        experiment_dir: str,
        device: torch.device,
        data: Tuple[DataLoader, DataLoader, DataLoader],
        input_size: Tuple[int, int],
        isocenter: np.array,
        volume: np.array,
        experiment_recorder: Experiment,
    ):

        self._device = device
        self._model_name = experiment_dict["model_name"]
        self._data = data
        self._latent_space_dims = experiment_dict["latent_space_dims"]
        self._batch_size = experiment_dict["batch_size"]
        self._epochs = experiment_dict["epochs"]
        self._showing_results_epoch = self._epochs - 1
        self._log_interval = experiment_dict["log_interval"]
        self.experiment_dict = experiment_dict  # TODO discuss if this is clean?

        self._input_size = input_size
        self._normalize = experiment_dict["normalize"]
        self._isocenter = isocenter
        self._volume = volume
        self._ref_anat_fname = experiment_dict["ref_anat_fname"]
        self._experiment_recoder = experiment_recorder

        self._lowest_loss = sys.float_info.max

        self._model = None
        self._optimizer = None
        self._loss_function = None
        self._lr_scheduler = None
        self._forward_pass = None

        self.run_dir = experiment_dir

        self.build_model()

        self._train_loss_recorder = LossHistory()
        self._test_loss_recorder = LossHistory()
        self._valid_loss_recorder = LossHistory()

    def build_model(self):
        self._model = get_model(self._model_name, self._latent_space_dims, self._device)

        self._model_name = self._model.__class__.__name__

        self._forward_pass = make_forward(
            self._model, self._device, self.experiment_dict
        )

        summary(self._model, self._input_size, batch_size=self._batch_size)

        self._optimizer = optim.Adam(
            self._model.parameters(), lr=6.68e-4, weight_decay=0.13
        )

        if self.experiment_dict[ExperimentKeys.DATASET_TYPE] in [
            DatasetTypes.triplet,
        ]:
            # TODO investigate what would be a good lr schedule for contrastive learning
            # The following should be equivalent to training with a fixed LR
            self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self._optimizer, int(1e20)
            )
        else:
            self._lr_scheduler = OneCycleLR(
                self._optimizer,
                max_lr=0.001,
                total_steps=None,
                epochs=self._epochs,
                steps_per_epoch=self.experiment_dict["num_steps_per_train_epoch"],
                pct_start=0.3,
                anneal_strategy="cos",
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=25.0,
                final_div_factor=10000.0,
                last_epoch=-1,
                verbose=False,
            )

    @property
    def model(self):
        return self._model

    @property
    def normalize(self):
        return self._normalize

    @property
    def model_name(self):
        return self._model_name

    @property
    def train_loader(self):
        return self._data[0]

    @property
    def test_loader(self):
        return self._data[2]

    @property
    def valid_loader(self):
        return self._data[1]

    @property
    def lowest_loss(self):
        return self._lowest_loss

    @property
    def experiment(self):
        return self._experiment_recoder

    @property
    def train_loss_recorder(self):
        return self._train_loss_recorder

    @property
    def valid_loss_recorder(self):
        return self._valid_loss_recorder

    @property
    def test_loss_recorder(self):
        return self._test_loss_recorder

    @property
    def best_checkpoint(self):
        return self.run_dir / self.best_model_fname

    def save_checkpoint(self, state, fname=best_model_fname):
        torch.save(state, self.run_dir / fname)

    def load_checkpoint(self, fname, device):
        checkpoint = torch.load(fname, map_location=device)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._lowest_loss = checkpoint["lowest_loss"]
        return checkpoint["epoch"]

    def get_batch_iterator(self, dataloader, phase):
        steps_per_epoch = self.experiment_dict[f"num_steps_per_{phase}_epoch"]
        batch_iterator = itertools.islice(dataloader, steps_per_epoch)
        return batch_iterator, steps_per_epoch

    def train(self, epoch):
        self._model.train()
        train_loss = 0

        batch_iterator, steps_per_epoch = self.get_batch_iterator(
            self.train_loader, "train"
        )
        for batch_idx, batch in enumerate(batch_iterator):
            # PyTorch uses the channel first convention so we need to reshape
            # the data
            for param_group in self._optimizer.param_groups:
                lr = param_group["lr"]

            self._optimizer.zero_grad()

            loss = self._forward_pass(batch)

            loss.backward()
            train_loss += loss.item()
            self._optimizer.step()
            self._lr_scheduler.step()

            if batch_idx % self._log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {:.9f}".format(
                        epoch,
                        batch_idx,
                        steps_per_epoch,
                        100.0 * batch_idx / steps_per_epoch,
                        loss.item() / len(batch),
                        lr,
                    )
                )

        loss_dataset_avg = train_loss / (steps_per_epoch * len(batch))
        # self._lr_scheduler.step(train_loss)
        logger.info(
            "====> Epoch: {} Average loss: {:.4f}".format(epoch, loss_dataset_avg)
        )
        self.experiment.log_metric("loss", loss_dataset_avg, epoch=epoch)
        self._train_loss_recorder.update(loss_dataset_avg)
        self._train_loss_recorder.end_epoch()

    def valid(self, epoch):
        self._model.eval()
        valid_loss = 0

        batch_iterator, steps_per_epoch = self.get_batch_iterator(
            self.valid_loader, "valid"
        )
        with torch.no_grad():
            for i, batch in enumerate(tqdm(batch_iterator, total=steps_per_epoch)):
                loss = self._forward_pass(batch)
                valid_loss += loss.item()

        valid_loss /= steps_per_epoch * len(batch)
        logger.info("====> Validation set loss: {:.4f}".format(valid_loss))

        self.experiment.log_metric("loss", valid_loss, epoch=epoch)
        self._valid_loss_recorder.update(valid_loss)
        self._valid_loss_recorder.end_epoch()

        # Save model and weights if loss is lower
        if valid_loss < self._lowest_loss:
            logger.info(
                "Current validation loss ({}) is lower than recorded "
                "lowest loss ({}): saving network...".format(
                    valid_loss, self._lowest_loss
                )
            )
            self._lowest_loss = valid_loss
            state = {
                "epoch": epoch,
                "state_dict": self._model.state_dict(),
                "lowest_loss": self._lowest_loss,
                "optimizer": self._optimizer.state_dict(),
            }
            self.save_checkpoint(state)
            logger.info("Network saved.")

    def save_loss_history(self):
        train_loss_history = self.train_loss_recorder.epochs
        valid_loss_history = self.valid_loss_recorder.epochs

        train_loss_data_fname = pjoin(
            self.run_dir, LoggerKeys.train_loss_data_file_basename.value
        )
        save_loss_history(train_loss_history, train_loss_data_fname)

        valid_loss_data_fname = pjoin(
            self.run_dir, LoggerKeys.valid_loss_data_file_basename.value
        )
        save_loss_history(valid_loss_history, valid_loss_data_fname)

        return train_loss_data_fname, valid_loss_data_fname

    def plot_results(self, data, fname: str):
        if self._normalize:
            data = postprocess(data, self._isocenter, self._volume)

        filename = pjoin(self.run_dir, fname)
        anat_ref_img = nib.load(self._ref_anat_fname)
        streamlines = Streamlines(data)

        plot_tractogram_with_anat_ref(
            streamlines,
            anat_ref_img,
            filename=filename,
            interactive=False,
        )
