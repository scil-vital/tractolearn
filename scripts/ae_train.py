#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for the inference and training of FINTA-multibundle
"""
import argparse
import logging
import os
import random
from os.path import join as pjoin

import comet_ml
import nibabel as nib
import numpy as np
import torch
from scilpy.io.utils import add_overwrite_arg, add_verbose_arg

from tractolearn.config.experiment import ExperimentFormatter
from tractolearn.filtering.latent_space_featuring import plot_latent_space
from tractolearn.learning.data_manager import DataManager
from tractolearn.learning.trainer_manager import Trainer
from tractolearn.logger import LoggerKeys, _set_up_logger
from tractolearn.models.autoencoding_utils import encode_data
from tractolearn.transformation.volume_utils import (
    compute_isocenter,
    compute_volume,
)
from tractolearn.visualization.plot_utils import plot_loss_history

torch.set_flush_denormal(True)

logger = logging.getLogger("root")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "config_file",
        help="Configuration YAML file",
    )

    parser.add_argument(
        "ref_anat_fname",
        help="Reference anatomical filename (usually a t1.nii.gz or wm.nii.gz) [ *.nii/.nii.gz ]",
    )

    parser.add_argument(
        "hdf5_dataset_path",
        help="Path of the hdf5 dataset path [ *.hdf5 ].",
    )

    parser.add_argument(
        "bundle_keys",
        help="Path of the bundle keys json file [ *.json ].",
    )

    parser.add_argument(
        "output_path",
        help="Output path to save experiment.",
    )

    add_overwrite_arg(parser)
    add_verbose_arg(parser)

    return parser.parse_args()


def main():
    args = _build_arg_parser()

    # Call Experiment module

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO if args.verbose == 1 else logging.DEBUG
        )

    experiment = ExperimentFormatter(
        args.config_file,
        args.hdf5_dataset_path,
        args.bundle_keys,
        args.output_path,
        args.ref_anat_fname,
    )

    logger_fname = pjoin(
        experiment.experiment_dir, LoggerKeys.logger_file_basename.value
    )
    _set_up_logger(logger_fname)

    logger.info("Starting experiment {}...".format(experiment.experiment_dir))
    logger.info("Reading experiment parameters...")

    experiment_dict = experiment.setup_experiment()

    logger.info("Finished reading experiment parameters.")

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Find a better way to import API key (eventually remove comet.ml)
    logger.info(comet_ml.get_comet_version())
    experiment_recorder = experiment.record_experiment(
        api_key=os.environ["COMETML"]
    )

    ref_anat_img = nib.load(experiment_dict["ref_anat_fname"])
    isocenter = compute_isocenter(ref_anat_img)
    volume = compute_volume(ref_anat_img)

    # Load tractograms
    logger.info("Loading tractograms...")

    data_manager = DataManager(experiment_dict, seed)

    logger.info("Finished loading tractograms.")

    (
        train_loader,
        valid_loader,
        test_loader,
        viz_loader,
    ) = data_manager.setup_data_loader()

    data_loaders = (train_loader, valid_loader, test_loader)

    logger.info("Building model and trainer...")

    trainer = Trainer(
        experiment_dict,
        experiment.experiment_dir,
        device,
        data_loaders,
        (data_manager.point_dims, data_manager.num_points),
        isocenter,
        volume,
        experiment_recorder,
    )

    logger.info("Finished building model and trainer.")

    # Start training run
    logger.info("Starting training...")
    for epoch in range(1, experiment_dict["epochs"] + 1):
        with experiment_recorder.train():
            trainer.train(epoch)
        with experiment_recorder.validate():
            trainer.valid(epoch)

        # Project the valid set
        val_latent_vecs, val_classes = encode_data(
            viz_loader,
            device,
            trainer.model,
            limit_batch=experiment_dict["viz_num_batches"],
        )
        # Save the vectors (and the class of each streamline for easy plotting)
        torch.save(
            {"vecs": val_latent_vecs, "classes": val_classes},
            f"{experiment.experiment_dir}/prediction_valid.pt",
        )
        # Plot the latent space
        latent_plot_filename = plot_latent_space(
            val_latent_vecs,
            val_classes,
            val_latent_vecs.shape[1],
            f"{experiment.experiment_dir}/latent_valid_epoch{epoch}",
            experiment_dict["rbx_classes"],
        )
        # Log the latent space plot to Comet
        experiment_recorder.log_image(
            latent_plot_filename, name="latent_umap", step=epoch
        )

    logger.info("Finished training.")
    torch.cuda.empty_cache()
    lowest_loss_epoch = trainer.load_checkpoint(
        trainer.best_checkpoint, device
    )

    trainer.model.eval()
    train_loss_data_fname, valid_loss_data_fname = trainer.save_loss_history()

    loss_plot_fname = pjoin(
        trainer.run_dir, LoggerKeys.loss_plot_file_basename.value
    )
    plot_loss_history(
        train_loss_data_fname, valid_loss_data_fname, loss_plot_fname
    )

    logger.info(
        "Loaded saved best model: loss {} at epoch: {}".format(
            trainer.lowest_loss, lowest_loss_epoch
        )
    )

    logger.info("Finished experiment.")


if __name__ == "__main__":
    main()
