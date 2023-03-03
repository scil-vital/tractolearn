# -*- coding: utf-8 -*-

from enum import Enum

from tractolearn.utils.logging_setup import set_up


class LoggerKeys(Enum):
    logger_file_basename = "logfile.log"

    TOTAL_LOSS = "loss"

    underscore = "_"
    fname_extension_sep = "."
    json_extension = "json"
    plot_extension = "png"
    trk_extension = "trk"

    latent_plot_fname_label = "latent"
    latent_distance_plot_fname_label = "latent_distance"
    interpolated_plot_fname_label = "latent_interpolated"
    test_data_fname_label = "test_data"
    valid_data_fname_label = "valid_data"
    generic_streamline_data_fname_label = "generic_streamline"

    test_data_batch_input_file_basename = (
        test_data_fname_label + "_batch_input.png"
    )
    test_data_reconst_batch_file_basename = (
        test_data_fname_label + "_reconst_batch.png"
    )
    test_data_reconst_tractogram_fname_root = (
        test_data_fname_label + "_reconst"
    )

    train_loss_data_file_basename = "train_loss.json"
    valid_loss_data_file_basename = "valid_loss.json"
    test_loss_data_file_basename = "test_loss.json"
    loss_plot_file_basename = "loss.png"


def _set_up_logger(log_fname):
    set_up(log_fname)
