# -*- coding: utf-8 -*-

import logging
import os
import shutil
from os.path import join as pjoin
from pathlib import Path

import yaml
from comet_ml import Experiment

from tractolearn.utils.utils import generate_uuid

logger = logging.getLogger("root")


class ExperimentKeys:
    TASK = "task"
    DATASET_TYPE = "dataset_type"
    LOSS_FN = "loss"
    MODEL_NAME = "model_name"
    TRACTOGRAM_GT_DATA_PATH = "tractogram_gt_data_path"
    TRACTOGRAM_DET_PLAUS_TRACK_DATA_PATH = (
        "tractogram_det_plaus_track_data_path"
    )
    TRACTOGRAM_DET_IMPLAUS_TRACK_DATA_PATH = (
        "tractogram_det_implaus_track_data_path"
    )
    TRACTOGRAM_PROB_PLAUS_TRACK_DATA_PATH = (
        "tractogram_prob_plaus_track_data_path"
    )
    TRACTOGRAM_PROB_IMPLAUS_TRACK_DATA_PATH = (
        "tractogram_prob_implaus_track_data_path"
    )
    TRACTOGRAM_FILTERING_DISTANCE_TEST_PLAUS_TRACK_DATA_PATH = (
        "tractogram_filtering_distance_test_plaus_track_data_path"
    )
    TRACTOGRAM_FILTERING_DISTANCE_TEST_IMPLAUS_TRACK_DATA_PATH = (
        "tractogram_filtering_distance_test_implaus_track_data_path"
    )
    TRACTOGRAM_TRAIN_PLAUS_DATA_PATH = "tractogram_train_plausibles_data_path"
    TRACTOGRAM_TRAIN_IMPLAUS_DATA_PATH = (
        "tractogram_train_implausibles_data_path"
    )
    TRACTOGRAM_VALID_PLAUS_DATA_PATH = "tractogram_valid_plausibles_data_path"
    TRACTOGRAM_VALID_IMPLAUS_DATA_PATH = (
        "tractogram_valid_implausibles_data_path"
    )
    TRACTOGRAM_TEST_PLAUS_DATA_PATH = "tractogram_test_plausibles_data_path"
    TRACTOGRAM_TEST_IMPLAUS_DATA_PATH = (
        "tractogram_test_implausibles_data_path"
    )
    REF_ANAT_FNAME = "ref_anat_fname"
    DATASET_NAME = "dataset_name"
    NUM_POINTS = "num_points"
    RANDOM_FLIP = "random_flip"
    NORMALIZE = "normalize"
    BATCH_SIZE = "batch_size"
    EPOCHS = "epochs"
    LATENT_SPACE_DIMS = "latent_space_dims"
    LOG_INTERVAL = "log_interval"
    WEIGHTS = "weights"
    VIZ = "viz"
    ARBITRARY_STREAMLINES_FNAMES = "arbitrary_streamlines_fnames"
    STREAMLINE_COUNT_SUBSAMPLE_FACTOR = "streamline_count_subsample_factor"
    OUT_PATH = "out_path"
    EXCLUDE_LABELS_TRACTOGRAM_FILE_ROOTNAMES = (
        "exclude_labels_tractogram_file_rootnames"
    )
    INCLUDE_LABELS_TRACTOGRAM_FILE_ROOTNAMES = (
        "include_labels_tractogram_file_rootnames"
    )
    HDF5_PATH = "hdf5_path"
    TRK_PATH = "trk_path"
    PKL_PATH = "pickle_path"
    BUNDLE_KEYS = "bundle_keys"
    DATA_IN_MEMORY = "data_in_memory"
    NUM_WORKERS = "num_workers"
    DISTANCE_FUNCTION = "distance_function"
    TO_SWAP = "to_swap"
    LOG_TO_COMET = "log_to_comet"


class ThresholdTestKeys:
    MODEL_TYPE = "model_type"
    LATENT_DIMS = "latent_dims"

    VALID_BUNDLE_PATH = "valid_bundle_path"
    INVALID_BUNDLE_FILE = "invalid_bundle_file"
    ATLAS_PATH = "atlas_path"
    REFERENCE = "reference"
    MODEL = "model"
    OUTPUT = "output"
    MAX_IMPLAUSIBLE = "max_implausible"
    MAX_PLAUSIBLE = "max_plausible"
    VIZ = "viz"
    NO_THRESHOLD = "no_threshold"
    THRESHOLDS_FILE = "thresholds_file"
    VALID_BUNDLE_PATH_TEST = "valid_bundle_path_test"
    INVALID_BUNDLE_FILE_TEST = "invalid_bundle_file_test"

    STREAMLINE_CLASSES = "streamline_classes"
    STREAMLINE_LENGTH = "streamline_length"

    DEVICE = "device"


class LearningTask:
    ae = "ae"
    contrastive_lecun_classes = "contrastive_lecun_classes"
    ae_contrastive_lecun_classes = "ae_contrastive_lecun_classes"
    ae_triplet_classes = "ae_triplet_classes"
    ae_triplet_hierarchical_classes = "ae_triplet_hierarchical_classes"


class DatasetTypes:
    hdf5dataset = "hdf5dataset"
    contrastive = "contrastive"
    triplet = "triplet"
    hierarchical = "hierarchichal"


class LossFunctionTypes:
    ae = "ae"
    contrastive_lecun_classes = "contrastive_lecun_classes"
    ae_contrastive_lecun_classes = "ae_contrastive_lecun_classes"
    ae_triplet_classes = "ae_triplet_classes"
    ae_triplet_hierarchical_classes = "ae_triplet_hierarchical_classes"


class ExperimentFormatter:
    def __init__(
        self,
        config: str,
        hdf5_path: str,
        bundle_keys_path: str,
        output_path: str,
        ref_anat_fname: str,
    ):
        """Experiment formatter.

        Parameters
        ----------
        config : str
            Config file.
        hdf5_path : str
            HDF5 filename.
        bundle_keys_path : str
            Bundle keys path.
        output_path : str
            Output path.
        ref_anat_fname : str
            Reference anatomical T1 image filename.
        """

        with open(config) as f:
            self.config = yaml.safe_load(f.read())

        self.config[ExperimentKeys.HDF5_PATH] = hdf5_path
        self.config[ExperimentKeys.BUNDLE_KEYS] = bundle_keys_path
        self.config[ExperimentKeys.OUT_PATH] = output_path
        self.config[ExperimentKeys.REF_ANAT_FNAME] = ref_anat_fname

        self.setup_task()

        self.experiment_dir = make_run_dir(
            out_path=self.config[ExperimentKeys.OUT_PATH]
        )
        # Copy the YAML configuration file to the experiment directory
        shutil.copy(config, self.experiment_dir)

    @property
    def log_to_comet(self):
        return self.config[ExperimentKeys.LOG_TO_COMET]

    def setup_experiment(self):

        return self.config

    def setup_task(self):
        task_configs = {
            LearningTask.ae: {
                ExperimentKeys.DATASET_TYPE: DatasetTypes.hdf5dataset,
                ExperimentKeys.LOSS_FN: LossFunctionTypes.ae,
            },
            LearningTask.contrastive_lecun_classes: {
                ExperimentKeys.DATASET_TYPE: DatasetTypes.contrastive,
                ExperimentKeys.LOSS_FN: LossFunctionTypes.contrastive_lecun_classes,
            },
            LearningTask.ae_contrastive_lecun_classes: {
                ExperimentKeys.DATASET_TYPE: DatasetTypes.contrastive,
                ExperimentKeys.LOSS_FN: LossFunctionTypes.ae_contrastive_lecun_classes,
            },
            LearningTask.ae_triplet_classes: {
                ExperimentKeys.DATASET_TYPE: DatasetTypes.triplet,
                ExperimentKeys.LOSS_FN: LossFunctionTypes.ae_triplet_classes,
            },
            LearningTask.ae_triplet_hierarchical_classes: {
                ExperimentKeys.DATASET_TYPE: DatasetTypes.hierarchical,
                ExperimentKeys.LOSS_FN: LossFunctionTypes.ae_triplet_hierarchical_classes,
            },
        }
        current_task = self.config[ExperimentKeys.TASK]
        self.config.update(task_configs[current_task])

    def record_experiment(self, api_key: str):
        experiment_recoder = Experiment(
            api_key=api_key,
            project_name="vital_scil_learning",
            log_code=False,
            auto_output_logging=False,
        )
        experiment_recoder.disabled_monkey_patching = True

        experiment_recoder.log_parameter(
            "experiment_dir", str(self.experiment_dir)
        )
        logger.info("Experiment id in comet: {}".format(experiment_recoder.id))

        experiment_recoder.log_parameters(self.config)

        return experiment_recoder


def path_str(x):
    return os.path.expandvars(x)


def make_run_dir(out_path=None):
    """Create a directory for this training run"""
    run_name = generate_uuid()
    if out_path is None:
        root_output_path = Path(os.environ.get("OUTPUT_PATH", "."))
    else:
        root_output_path = out_path
    run_dir = Path(pjoin(root_output_path, run_name))
    run_dir.mkdir(parents=True)
    return run_dir
