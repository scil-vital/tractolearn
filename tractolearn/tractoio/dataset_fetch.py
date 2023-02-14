# -*- coding: utf-8 -*-

import contextlib
import enum
import logging
import os
import tarfile
import zipfile
from hashlib import md5
from os.path import join as pjoin
from shutil import copyfileobj
from urllib.request import urlopen

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

TRACTOLEARN_DATASETS_URL = "https://zenodo.org/record/"

key_separator = ","


class Dataset(enum.Enum):
    """Datasets for tractography learning."""

    BUNDLE_LABEL_CONFIG = "bundle_label_config"
    CONTRASTIVE_AUTOENCODER_WEIGHTS = "contrastive_ae_weights"
    MNI2009CNONLINSYMM_ANAT = "mni2009cnonlinsymm_anat"
    GENERATIVE_LOA_CONE_CONFIG = "generative_loa_cone_config"
    GENERATIVE_SEED_STRML_RATIO_CONFIG = (
        "generative_seed_streamline_ratio_config"
    )
    GENERATIVE_STRML_MAX_COUNT_CONFIG = (
        "generative_streamline_max_count_config"
    )
    GENERATIVE_STRML_RQ_COUNT_CONFIG = "generative_streamline_req_count_config"
    GENERATIVE_WM_TISSUE_CRITERION_CONFIG = (
        "generative_wm_tisue_criterion_config"
    )
    RECOBUNDLESX_ATLAS = "recobundlesx_atlas"
    RECOBUNDLESX_CONFIG = "recobundlesx_config"
    TRACTOINFERNO_HCP_CONTRASTIVE_THR_CONFIG = (
        "tractoinferno_hcp_contrastive_thr_config"
    )
    TRACTOINFERNO_HCP_REF_TRACTOGRAPHY = "tractoinferno_hcp_ref_tractography"

    # Methods for argparse compatibility
    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return Dataset[s.upper()]
        except KeyError:
            return s


class FetcherError(Exception):
    pass


class DatasetError(Exception):
    pass


def _check_known_dataset(name):
    """Raise a DatasetError if the dataset is unknown.

    Parameters
    ----------
    name : string
        Dataset name.
    """

    if name not in Dataset.__members__.keys():
        raise DatasetError(_unknown_dataset_msg(name))


def _exclude_dataset_use_permission_files(fnames, permission_fname):
    """Exclude dataset use permission files from the data filenames.

    Parameters
    ----------
    fnames : list
        Filenames.

    Returns
    -------
    key : string
        Key value.
    """

    return [f for f in fnames if permission_fname not in f]


def copyfileobj_withprogress(fsrc, fdst, total_length, length=16 * 1024):

    for _ in tqdm(range(0, int(total_length), length), unit=" MB"):
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)


def _already_there_msg(folder):
    """Print a message indicating that dataset is already in place."""

    msg = "Dataset is already in place.\nIf you want to fetch it again, "
    msg += f"please first remove the file at issue in folder\n{folder}"
    logger.info(msg)


def _unknown_dataset_msg(name):
    """Build a message indicating that dataset is not known.

    Parameters
    ----------
    name : string
        Dataset name.

    Returns
    -------
    msg : string
        Message.
    """

    msg = f"Unknown dataset.\nProvided: {name}; Available: {Dataset.__members__.keys()}"
    return msg


def _get_file_hash(filename):
    """Generate an MD5 hash for the entire file in blocks of 128.

    Parameters
    ----------
    filename : str
        The path to the file whose MD5 hash is to be generated.

    Returns
    -------
    hash256_data : str
        The computed MD5 hash from the input file.
    """

    hash_data = md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * hash_data.block_size), b""):
            hash_data.update(chunk)
    return hash_data.hexdigest()


def check_hash(filename, stored_hash=None):
    """Check that the hash of the given filename equals the stored one.

    Parameters
    ----------
    filename : str
        The path to the file whose hash is to be compared.
    stored_hash : str, optional
        Used to verify the generated hash.
        Default: None, checking is skipped.
    """

    if stored_hash is not None:
        computed_hash = _get_file_hash(filename)
        if stored_hash.lower() != computed_hash:
            msg = (
                f"The downloaded file\n{filename}\ndoes not have the expected hash "
                f"value of {stored_hash}.\nInstead, the hash value was {computed_hash}.\nThis could "
                "mean that something is wrong with the file or that the "
                "upstream file has been updated.\nYou can try downloading "
                f"file again or updating to the newest version of {__name__.split('.')[0]}"
            )
            raise FetcherError(msg)


def _get_file_data(fname, url):

    with contextlib.closing(urlopen(url)) as opener:
        try:
            response_size = opener.headers["content-length"]
        except KeyError:
            response_size = None

        with open(fname, "wb") as data:
            if response_size is None:
                copyfileobj(opener, data)
            else:
                copyfileobj_withprogress(opener, data, response_size)


def fetch_data(files, folder, data_size=None):
    """Download files to folder and checks their hashes.

    Parameters
    ----------
    files : dictionary
        For each file in ``files`` the value should be (url, hash). The file
        will be downloaded from url if the file does not already exist or if
        the file exists but the hash does not match.
    folder : str
        The directory where to save the file, the directory will be created if
        it does not already exist.
    data_size : str, optional
        A string describing the size of the data (e.g. "91 MB") to be logged to
        the screen. Default does not produce any information about data size.

    Raises
    ------
    FetcherError
        Raises if the hash of the file does not match the expected value. The
        downloaded file is not deleted when this error is raised.
    """

    if not os.path.exists(folder):
        logger.info(f"Creating new folder\n{folder}")
        os.makedirs(folder)

    if data_size is not None:
        logger.info(f"Data size is approximately {data_size}")

    all_skip = True
    for f in files:
        url, _hash = files[f]
        fullpath = pjoin(folder, f)
        if os.path.exists(fullpath) and (
            _get_file_hash(fullpath) == _hash.lower()
        ):
            continue
        all_skip = False
        logger.info(f"Downloading\n{f}\nto\n{folder}")
        _get_file_data(fullpath, url)
        check_hash(fullpath, _hash)
    if all_skip:
        _already_there_msg(folder)
    else:
        logger.info(f"\nFiles successfully downloaded to\n{folder}")


def _make_fetcher(
    folder,
    name,
    baseurl,
    remote_fnames,
    local_fnames,
    hash_list=None,
    doc="",
    data_size=None,
    msg=None,
    unzip=False,
):
    """Create a new fetcher.

    Parameters
    ----------
    folder : str
        The full path to the folder in which the files would be placed locally.
    name : str
        The name of the fetcher function.
    baseurl : str
        The URL from which this fetcher reads files.
    remote_fnames : list of strings
        The names of the files in the baseurl location.
    local_fnames : list of strings
        The names of the files to be saved on the local filesystem.
    hash_list : list of strings, optional
        The hash values of the files. Used to verify the content of the files.
        Default: None, skipping checking hash.
    doc : str, optional.
        Documentation of the fetcher.
    data_size : str, optional.
        If provided, is sent as a message to the user before downloading
        starts.
    msg : str, optional
        A message to print to screen when fetching takes place. Default (None)
        is to print nothing.
    unzip : bool, optional
        Whether to unzip the file(s) after downloading them. Supports zip, gz,
        and tar.gz files.

    Returns
    -------
    fetcher : function
        A function that, when called, fetches data according to the designated
        inputs

    """

    def fetcher():
        files = {}
        for (
            i,
            (f, n),
        ) in enumerate(zip(remote_fnames, local_fnames)):
            files[n] = (
                baseurl + f,
                hash_list[i] if hash_list is not None else None,
            )
        fetch_data(files, folder, data_size)

        if msg is not None:
            logger.info(msg)
        if unzip:
            for f in local_fnames:
                split_ext = os.path.splitext(f)
                if split_ext[-1] == ".gz" or split_ext[-1] == ".bz2":
                    if os.path.splitext(split_ext[0])[-1] == ".tar":
                        ar = tarfile.open(pjoin(folder, f))
                        ar.extractall(path=folder)
                        ar.close()
                    else:
                        raise ValueError("File extension is not recognized")
                elif split_ext[-1] == ".zip":
                    z = zipfile.ZipFile(pjoin(folder, f), "r")
                    files[f] += (tuple(z.namelist()),)
                    z.extractall(folder)
                    z.close()
                else:
                    raise ValueError("File extension is not recognized")

        return files, folder

    fetcher.__name__ = name
    fetcher.__doc__ = doc
    return fetcher


fetch_bundle_label_config = (
    "fetch_bundle_label_config",
    TRACTOLEARN_DATASETS_URL + "/7562790/files/",
    ["rbx_atlas_v10.json"],
    ["rbx_atlas_v10.json"],
    ["0edde5be1b3e32d12a5f02d77b46d32b"],
    "Bundle labels",
    "581B",
    "",
    False,
)

fetch_contrastive_ae_weights = (
    "fetch_contrastive_ae_weights",
    TRACTOLEARN_DATASETS_URL + "7562790/files/",
    ["best_model_contrastive_tractoinferno_hcp.pt"],
    ["best_model_contrastive_tractoinferno_hcp.pt"],
    ["2181aa950d8110b89f5b4bf7ebbb9aff"],
    "Download contrastive-loss trained tractolearn autoencoder weights",
    "56.7MB",
    "",
    False,
)

fetch_mni2009cnonlinsymm_anat = (
    "fetch_mni2009cnonlinsymm_anat",
    TRACTOLEARN_DATASETS_URL + "7562790/files/",
    ["mni_masked.nii.gz"],
    ["mni_masked.nii.gz"],
    ["ea6c119442d23a25033de19b55c607d3"],
    "Download MNI ICBM 2009c Nonlinear Symmetric 1×1x1mm template dataset",
    "4.9MB",
    "",
    False,
)

fetch_generative_loa_cone_config = (
    "fetch_generative_loa_cone_config",
    TRACTOLEARN_DATASETS_URL + "/7562790/files/",
    ["degree.json"],
    ["degree.json"],
    ["9b97737ac0f9f362f9936792028de934"],
    "Bundle-wise local orientation angle cone in degrees for generative streamline assessment",
    "670B",
    "",
    False,
)

fetch_generative_seed_streamline_ratio_config = (
    "fetch_generative_seed_streamline_ratio_config",
    TRACTOLEARN_DATASETS_URL + "/7562790/files/",
    ["ratio.json"],
    ["ratio.json"],
    ["69d14a2100d9a948f63489767586a939"],
    "Bundle-wise (subject | atlas) seed streamline ratio",
    "978B",
    "",
    False,
)

fetch_generative_streamline_max_count_config = (
    "fetch_generative_streamline_max_count_config",
    TRACTOLEARN_DATASETS_URL + "/7562790/files/",
    ["max_total_sampling.json"],
    ["max_total_sampling.json"],
    ["33f46357fd791d2c6f6b7da473fd8bbc"],
    "Maximum number of generative bundle-wise streamline count",
    "787B",
    "",
    False,
)

fetch_generative_streamline_req_count_config = (
    "fetch_generative_streamline_req_count_config",
    TRACTOLEARN_DATASETS_URL + "/7562790/files/",
    ["number_rejection_sampling.json"],
    ["number_rejection_sampling.json"],
    ["46acbb64aa3b2ba846c727cb8554566d"],
    "Requested number of generative bundle-wise streamline count",
    "783MB",
    "",
    False,
)

fetch_generative_wm_tisue_criterion_config = (
    "fetch_generative_wm_tisue_criterion_config",
    TRACTOLEARN_DATASETS_URL + "/7562790/files/",
    ["white_matter_mask.json"],
    ["white_matter_mask.json"],
    ["79b7605ec13e0b2bcb6a38c15f159381"],
    "Bundle-wise WM tissue criterion (WM mask | thresholded FA) for generative streamline assessment",
    "748B",
    "",
    False,
)

fetch_recobundlesx_atlas = (
    "fetch_recobundlesx_atlas",
    TRACTOLEARN_DATASETS_URL + "7562635/files/",
    ["atlas.zip"],
    ["atlas.zip"],
    ["0d2857efa7cfda6f57e5abcad4717c2a"],
    "Download RecoBundlesX population average and centroid tractograms",
    "159.0MB",
    "",
    True,
)

fetch_recobundlesx_config = (
    "fetch_recobundlesx_config",
    TRACTOLEARN_DATASETS_URL + "7562635/files/",
    ["config.zip"],
    ["config.zip"],
    ["439e2488597243455872ec3dcb50eda7"],
    "Download RecoBundlesX clustering parameter values",
    "3.6KB",
    "",
    True,
)

fetch_tractoinferno_hcp_contrastive_threshold_config = (
    "fetch_tractoinferno_hcp_contrastive_threshold_config",
    TRACTOLEARN_DATASETS_URL + "/7562790/files/",
    ["thresholds_contrastive_tractoinferno_hcp.json"],
    ["thresholds_contrastive_tractoinferno_hcp.json"],
    ["beecb87d73d53fa4f4ed6714af420cfd"],
    "Bundle-wise bundling latent space distance threshold values",
    "824B",
    "",
    False,
)

fetch_tractoinferno_hcp_ref_tractography = (
    "fetch_tractoinferno_hcp_ref_tractography",
    TRACTOLEARN_DATASETS_URL + "/7562790/files/",
    ["data_tractoinferno_hcp_qbx.hdf5"],
    ["data_tractoinferno_hcp_qbx.hdf5"],
    ["4803d36278d1575a40e9048a7380aa10"],
    "Download TractoInferno-HCP reference tractography dataset",
    "74.0GB",
    "",
    False,
)


def get_fetcher_method(name):
    """Provide the fetcher method corresponding to the method name.

    Returns
    -------
    callable
        Fetcher method.
    """

    if name == Dataset.BUNDLE_LABEL_CONFIG.name:
        return fetch_bundle_label_config
    elif name == Dataset.CONTRASTIVE_AUTOENCODER_WEIGHTS.name:
        return fetch_contrastive_ae_weights
    elif name == Dataset.MNI2009CNONLINSYMM_ANAT.name:
        return fetch_mni2009cnonlinsymm_anat
    elif name == Dataset.GENERATIVE_LOA_CONE_CONFIG.name:
        return fetch_generative_loa_cone_config
    elif name == Dataset.GENERATIVE_SEED_STRML_RATIO_CONFIG.name:
        return fetch_generative_seed_streamline_ratio_config
    elif name == Dataset.GENERATIVE_STRML_MAX_COUNT_CONFIG.name:
        return fetch_generative_streamline_max_count_config
    elif name == Dataset.GENERATIVE_STRML_RQ_COUNT_CONFIG.name:
        return fetch_generative_streamline_req_count_config
    elif name == Dataset.GENERATIVE_WM_TISSUE_CRITERION_CONFIG.name:
        return fetch_generative_wm_tisue_criterion_config
    elif name == Dataset.RECOBUNDLESX_ATLAS.name:
        return fetch_recobundlesx_atlas
    elif name == Dataset.RECOBUNDLESX_CONFIG.name:
        return fetch_recobundlesx_config
    elif name == Dataset.TRACTOINFERNO_HCP_CONTRASTIVE_THR_CONFIG.name:
        return fetch_tractoinferno_hcp_contrastive_threshold_config
    elif name == Dataset.TRACTOINFERNO_HCP_REF_TRACTOGRAPHY.name:
        return fetch_tractoinferno_hcp_ref_tractography
    else:
        raise DatasetError(_unknown_dataset_msg(name))


def provide_dataset_description():
    """Provide the description of the available datasets.

    Returns
    -------
    descr : list
        Dataset value, description and URL tuples.
    """

    url_idx = 1
    descr_idx = 5
    descr = list()

    for elem in list(Dataset):
        params = get_fetcher_method(elem.name)
        descr.append(
            elem.value
            + ": "
            + params[descr_idx]
            + ": "
            + params[url_idx]
            + "\n"
        )

    return descr


def retrieve_dataset(name, path):
    """Retrieve the given dataset to the provided path.

    Parameters
    ----------
    name : string
        Dataset name.
    path : string
        Destination path.

    Returns
    -------
    fnames : string or list
        Filenames for dataset.
    """

    logger.info(f"\nDataset: {name}")

    params = get_fetcher_method(name)
    files, folder = _make_fetcher(path, *params)()

    file_basename = list(files.keys())[0]

    # Check if the file is a ZIP file
    if zipfile.is_zipfile(pjoin(folder, file_basename)):
        fnames = files[file_basename][1]
        return sorted(
            [
                pjoin(folder, f)
                for f in fnames
                if os.path.isfile(pjoin(folder, f))
            ]
        )
    else:
        return pjoin(folder, file_basename)
