# -*- coding: utf-8 -*-
from enum import Enum


class LatentSpaceKeys(Enum):
    balanced_accuracy_label = "balanced_accuracy"

    latent_space_fname_label = "latent_space"
    latent_space_histogram_fname_label = "histogram"
    latent_space_histogram_cont_fname_label = "histogram_cont"
    latent_space_roc_fname_label = "roc"
    latent_space_stats_fname_label = "stats"
    latent_space_stats_raincloud_fname_label = "stats_raincloud"
    tractogram_fname_label = "tractogram"
    colorbar_fname_label = "colorbar"
    nearest_neighbor_fname_label = "nearest_neighbor"
    threshold_fname_label = "thr"

    plaus_streamlines_tractogram_latent_distances_cbar_file_basename = (
        "plausible_streamlines_tractogram_latent_space_distances_colorbar.png"
    )
    implaus_streamlines_tractogram_latent_distances_cbar_file_basename = "implausible_streamlines_tractogram_latent_space_distances_colorbar.png"

    plausible_streamlines_latent_distances_file_basename = (
        "plausible_streamlines_latent_distances.pickle"
    )
    implausible_streamlines_latent_distances_file_basename = (
        "implausible_streamlines_latent_distances.pickle"
    )
    roc_data_file_basename = "roc_data.pickle"
    filtering_stats_file_basename = "filtering_stats.json"

    confusion_matrix_fname_root = "confusion_matrix"

    test_data_reconst_plaus_tractogram_fname_root = (
        "test_data_reconst_plausibles"
    )
    test_data_reconst_implaus_tractogram_fname_root = (
        "test_data_reconst_implausibles"
    )

    test_data_tp_tractogram_fname_root = "test_data_tp"
    test_data_fp_tractogram_fname_root = "test_data_fp"
    test_data_tn_tractogram_fname_root = "test_data_tn"
    test_data_fn_tractogram_fname_root = "test_data_fn"
