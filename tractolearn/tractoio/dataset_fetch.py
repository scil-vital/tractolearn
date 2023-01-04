# -*- coding: utf-8 -*-

import contextlib
import enum
import os
import subprocess
import sys
import tarfile
import zipfile
from hashlib import md5
from os.path import join as pjoin
from shutil import copyfileobj
from urllib.request import urlopen


TRACTOLEARN_DATASETS_URL = "https://osf.io/"

key_separator = ","


class Dataset(enum.Enum):
    CONTRASTIVE_AUTOENCODER_WEIGHTS = "contrastive_ae_weights"
    MNI2009CNONLINSYMM_ANAT = "mni2009cnonlinsymm_anat"
    TRACTOINFERNO_HCP_REF_TRACTOGRAPHY = "tractoinferno_hcp_ref_tractography"


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


def update_progressbar(progress, total_length):
    """Show progressbar.

    Takes a number between 0 and 1 to indicate progress from 0 to 100%.
    """

    # Try to set the bar_length according to the console size
    # noinspection PyBroadException
    try:
        columns = subprocess.Popen("tput cols", "r").read()
        bar_length = int(columns) - 46
        if bar_length < 1:
            bar_length = 20
    except Exception:
        # Default value if determination of console size fails
        bar_length = 20
    block = int(round(bar_length * progress))
    size_string = f"{float(total_length) / (1024 * 1024):.2f} MB"
    text = rf"Download Progress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}%  of {size_string}"
    sys.stdout.write(text)
    sys.stdout.flush()


def copyfileobj_withprogress(fsrc, fdst, total_length, length=16 * 1024):

    copied = 0
    while True:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
        copied += len(buf)
        progress = float(copied) / float(total_length)
        update_progressbar(progress, total_length)


def _already_there_msg(folder):
    """Print a message indicating that dataset is already in place."""

    msg = "Dataset is already in place.\nIf you want to fetch it again, "
    msg += f"please first remove the file at issue in folder\n{folder}"
    print(msg)


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
        print(f"Creating new folder\n{folder}")
        os.makedirs(folder)

    if data_size is not None:
        print(f"Data size is approximately {data_size}")

    all_skip = True
    for f in files:
        url, _hash = files[f]
        fullpath = pjoin(folder, f)
        if os.path.exists(fullpath) and (
            _get_file_hash(fullpath) == _hash.lower()
        ):
            continue
        all_skip = False
        print(f"Downloading\n{f}\nto\n{folder}")
        _get_file_data(fullpath, url)
        check_hash(fullpath, _hash)
    if all_skip:
        _already_there_msg(folder)
    else:
        print(f"\nFiles successfully downloaded to\n{folder}")


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
            print(msg)
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


fetch_contrastive_ae_weights = (
    "fetch_contrastive_ae_weights",
    TRACTOLEARN_DATASETS_URL + "2xmgw/",
    ["download"],
    ["weights.pt"],
    ["7170d0192fa00b5ef069f8e7c274950c"],
    "Download contrastive-loss trained autoencoder weights",
    "543B",
    False,
)

fetch_mni2009cnonlinsymm_anat = (
    "fetch_mni2009cnonlinsymm_anat",
    TRACTOLEARN_DATASETS_URL + "br4ds/",
    ["download"],
    ["sub01-dwi.zip"],
    ["705396981f1bcda51de12098db968390"],
    "Download MNI ICBM 2009c Nonlinear Symmetric 1×1x1mm template dataset",
    "4.2MB",
    False,
)

fetch_tractoinferno_hcp_ref_tractography = (
    "fetch_tractoinferno_hcp_ref_tractography",
    TRACTOLEARN_DATASETS_URL + "br4ds/",
    ["download"],
    ["tractoinferno_hcp_ref_tractography.hdf5"],
    ["705396981f1bcda51de12098db968390"],
    "Download TractoInferno-HCP reference tractography dataset",
    "0.39MB",
    False,
)


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

    print(f"\nDataset: {name}")

    if name == Dataset.CONTRASTIVE_AUTOENCODER_WEIGHTS.name:
        params = fetch_contrastive_ae_weights
        files, folder = _make_fetcher(path, *params)()
        return pjoin(folder, list(files.keys())[0])
    elif name == Dataset.MNI2009CNONLINSYMM_ANAT.name:
        params = fetch_mni2009cnonlinsymm_anat
        files, folder = _make_fetcher(path, *params)()
        return pjoin(folder, list(files.keys())[0])
    elif name == Dataset.TRACTOINFERNO_HCP_REF_TRACTOGRAPHY.name:
        params = fetch_tractoinferno_hcp_ref_tractography
        files, folder = _make_fetcher(path, *params)()
        return pjoin(folder, list(files.keys())[0])
    else:
        raise DatasetError(_unknown_dataset_msg(name))
