# -*- coding: utf-8 -*-

import logging
from os.path import join as pjoin
from time import time
from typing import Tuple

import numpy as np
import torch
import umap
from matplotlib import pyplot as plt
from numpy.random import SeedSequence
from pathos.multiprocessing import Pool
from scipy.stats import multivariate_normal
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from tractolearn.learning.dataset import OnTheFlyDataset
from tractolearn.models.autoencoding_utils import encode_data

torch.set_flush_denormal(True)
logger = logging.getLogger("root")


class RejectionSampler:
    """Implements the rejection sampling algorithm using an abstract interface
    that only requires data from the distribution we want to sample from.
    """

    def __init__(
        self,
        data: np.ndarray,
        kde_bw: float = None,
        kde_bw_factor: float = 1,
        kernel="gaussian",
        proposal_distribution_params: dict = None,
        scaling_mode: str = "max",
        allow_singular=False,
        kde_bw_auto_estimation: str = "Cross-Validation",
        proposal_distribution_name: str = "multivariate_normal",
        bundle_name=None,
        output=None,
        cluster_estimation: str = "silhouette",
    ):
        """Initializes the inner distributions used by the rejection sampling
        algorithm.
        Parameters
        ----------
        data : ndarray
            N x D array where N is the number of data points and D is the
            dimensionality of the data.
        kde_bw : float
            The bandwidth of the kernel density estimator. If no bandwidth is
            given, it will be determined by cross-validation over `data`.
        proposal_distribution_params : dict
            Only with proposal_distribution = multivariate_normal
            The `mean` and `cov` parameters to use for the Gaussian proposal
            distribution. If no params are given, the proposal distribution is
            inferred from the mean and covariance computed on `data`.
            OR
            The `n_components` parameters to use for the GMM proposal
            distribution. If no params are given, the proposal distribution is
            inferred with 2 components.
        scaling_mode : str
            Algorithm to use to compute the scaling factor between the proposal
            distribution and the KDE estimation of the real distribution.
            Valid options are ['max', '3rd_quartile'].
        kde_bw_auto_estimation: str
            Automated method to use to estimate the optimal bandwidth
            Valid options are ["Cross-Validation", "Silverman1986"]
        proposal_distribution: str
            Proposal distribution
            Valid options are ["multivariate_normal", "GMM"]
        """

        if kde_bw_factor <= 0:
            raise ValueError(
                "KDE bandwidth factor cannot be negative or 0.\n"
                f"Found: {kde_bw_factor}"
            )

        self.data = data
        self.proposal_distribution_name = proposal_distribution_name

        # Initialize kernel density estimate
        if kde_bw:
            self.kde = KernelDensity(bandwidth=kde_bw, kernel=kernel).fit(
                self.data
            )
        else:
            if kde_bw_auto_estimation == "Cross-Validation":
                logger.info(
                    "Cross-validating bandwidth of kernel density estimate..."
                )
                grid = GridSearchCV(
                    KernelDensity(kernel=kernel),
                    {"bandwidth": np.e ** np.linspace(-1, 1, 100)},
                    cv=ShuffleSplit(),
                    verbose=1,
                    n_jobs=-1,
                )
                grid.fit(self.data)
                self.kde = grid.best_estimator_
            elif kde_bw_auto_estimation == "Silverman1986":
                sigma = np.std(self.data, ddof=1)
                IQR = (
                    np.percentile(self.data, q=75)
                    - np.percentile(self.data, q=25)
                ) / 1.3489795003921634
                m = np.minimum(sigma, IQR)
                b = 0.9 * m / len(self.data) ** 0.2
                self.kde = KernelDensity(bandwidth=b, kernel=kernel).fit(
                    self.data
                )
            else:
                raise NotImplementedError(
                    "kde_bw_auto_estimation valid option are ['Cross-Validation', 'Silverman1986'], "
                    f"Got {kde_bw_auto_estimation}. "
                )

        # Multiply the optimal bandwidth with a factor to make the generative
        # process more/less strict/permissive
        bw = self.kde.__getattribute__("bandwidth")

        self.kde.set_params(**{"bandwidth": kde_bw_factor * bw})

        # Get the optimal bandwidth found using cross validation

        logger.info(f"Bandwidth: {self.kde.__getattribute__('bandwidth')}")

        # Init proposal distribution
        if self.proposal_distribution_name == "multivariate_normal":
            if proposal_distribution_params:
                mean = np.full(
                    data.shape[1], proposal_distribution_params["mean"]
                )
                cov = proposal_distribution_params["cov"]
            else:
                mean = np.mean(self.data, axis=0)
                cov = np.cov(self.data, rowvar=False)
            self.proposal_distribution = multivariate_normal(
                mean=mean, cov=cov, allow_singular=allow_singular
            )

            likelihood = self.proposal_distribution.pdf(self.data)
        elif self.proposal_distribution_name == "GMM":
            if proposal_distribution_params:
                n_components = proposal_distribution_params["n_components"]
                cov = "full"
            else:
                if cluster_estimation == "silhouette":
                    assert bundle_name is not None
                    assert output is not None
                    silhouette_avg = []
                    range_of_clusters = list(range(2, 15))
                    for num_clusters in range_of_clusters:
                        # initialise kmeans
                        gmm = GaussianMixture(n_components=num_clusters)
                        gmm.fit(self.data)
                        cluster_labels = gmm.predict(self.data)

                        # silhouette score
                        silhouette_avg.append(
                            silhouette_score(self.data, cluster_labels)
                        )
                    plt.plot(range_of_clusters, silhouette_avg, "bx-")
                    plt.xlabel("Values of K")
                    plt.ylabel("Silhouette score")
                    plt.title(
                        f"{bundle_name} Silhouette analysis For Optimal k"
                    )
                    plt.savefig(pjoin(output, f"{bundle_name}_silhouette.png"))

                    n_components = range_of_clusters[np.argmax(silhouette_avg)]
                    cov = "full"

                    logger.info(
                        f"Optimal GMM parameters: n_components = {n_components}, cov = {cov} "
                    )
                elif cluster_estimation == "BIC":
                    n_components = range(2, 16)
                    covariance_type = ["spherical", "tied", "diag", "full"]
                    score = []
                    for cov in covariance_type:
                        for n_comp in tqdm(n_components):
                            gmm = GaussianMixture(
                                n_components=n_comp, covariance_type=cov
                            )
                            gmm.fit(self.data)
                            score.append((cov, n_comp, gmm.bic(self.data)))

                    lowest_bic = score[np.argmin([s for _, _, s in score])]
                    n_components = lowest_bic[1]
                    cov = lowest_bic[0]

                    logger.info(
                        f"Optimal GMM parameters: n_components = {n_components}, cov = {cov} "
                    )

                else:
                    raise NotImplementedError(
                        "cluster_estimation valid option are ['silhouette', 'BIC'], "
                        f"Got {cluster_estimation}. "
                    )

            self.proposal_distribution = GaussianMixture(
                n_components=n_components, verbose=True, covariance_type=cov
            ).fit(self.data)

            likelihood = np.e ** self.proposal_distribution.score_samples(
                self.data
            )

        else:
            raise NotImplementedError(
                "proposal_distribution valid option are ['multivariate_normal', 'GMM']"
                f"Got {self.proposal_distribution_name}"
            )

        # Initialize scaling factor
        factors_between_data_and_proposal_distribution = (
            np.e ** self.kde.score_samples(self.data)
        ) / likelihood

        # 'max' factor is used when the initial samples fit in a sensible
        # distribution. Should be preferred to other algorithms whenever it is
        # applicable.
        # '3rd_quartile' factor is used when outliers in the initial samples
        # skew the ratio and cause an impossibly high scaling factor.
        if scaling_mode == "max":
            self.scaling_factor = np.max(
                factors_between_data_and_proposal_distribution
            )
        else:  # scaling_mode == '3rd_quartile'
            self.scaling_factor = np.percentile(
                factors_between_data_and_proposal_distribution, 75
            )

    def _sample(
        self, nb_samples: int, rng: np.random.Generator = None
    ) -> Tuple[np.ndarray, int]:
        """Performs rejection sampling to sample M samples that fit the visible
        distribution of ``data``. `self._sample` performs the sampling in
        itself, as opposed to `self.sample` which is a public wrapper to
        coordinate sampling multiple batches in parallel.
        Parameters
        ----------
        nb_samples : int
            Number of samples to sample from the data distribution.
        rng : np.Generator
            Random Number Generator to use to draw from both the proposal and
            uniform distributions.
        Returns
        -------
        accepted_samples : ndarray
            M x D array where M equals `num_samples` and D is the
            dimensionality of the sampled data.
        nb_trials : int
            Number of draws (rejected or accepted) it took to reach M accepted
            samples. This is mainly useful to evaluate the efficiency of the
            rejection sampling.
        """

        if rng is None:
            rng = np.random.default_rng()

        accepted_samples = []
        nb_trials = 0
        while len(accepted_samples) < nb_samples:
            if self.proposal_distribution_name == "multivariate_normal":
                sample = self.proposal_distribution.rvs(
                    size=1, random_state=rng
                )
                rand_likelihood_threshold = rng.uniform(
                    0,
                    self.scaling_factor
                    * self.proposal_distribution.pdf(sample),
                )
            elif self.proposal_distribution_name == "GMM":
                self.proposal_distribution.set_params(
                    random_state=rng.integers(0, 2**32 - 1)
                )
                sample, _ = self.proposal_distribution.sample(n_samples=1)
                rand_likelihood_threshold = rng.uniform(
                    0,
                    self.scaling_factor
                    * (
                        np.e
                        ** self.proposal_distribution.score_samples(sample)
                    ),
                )
                sample = sample.squeeze()
            else:
                raise NotImplementedError(
                    "proposal_distribution valid option are ['multivariate_normal', 'GMM']"
                    f"Got {self.proposal_distribution_name}"
                )

            if rand_likelihood_threshold <= (
                np.e ** self.kde.score_samples(sample[np.newaxis, :])
            ):
                accepted_samples.append(sample)

            nb_trials += 1

        return np.array(accepted_samples), nb_trials

    def sample(
        self, nb_samples: int, batch_size: int = None, entropy: int = 1234
    ) -> tuple:
        """Performs rejection sampling to sample N samples that fit the
        visible distribution of `data`.
        Parameters
        ----------
        nb_samples : int
            The number of samples to sample from the data distribution.
        batch_size : int
            Number of samples to generate in each batch. If ``None``, defaults
            to ``nb_samples / 100``.
        entropy : int
            Entropy for the seed generator.
        Returns
        -------
        M x D array where M equals `nb_samples` and D is the dimensionality
        of the sampled data.
        """

        # Determine the size and number of batches (possibly including a final
        # irregular batch)
        if batch_size is None:
            batch_size = nb_samples // 100
            logger.info(
                f"No `batch_size` provided. Defaulted to use a `batch_size` of {batch_size}."
            )

            if batch_size == 0:
                batch_size = 1

        batches = [batch_size] * (nb_samples // batch_size)
        if last_batch := nb_samples % batch_size:
            batches.append(last_batch)

        # Prepare different seeds for each batch
        ss = SeedSequence(entropy=entropy)
        # Log this so that the entropy can be re-used for reproducibility
        logger.info(
            f"Entropy of root `SeedSequence` used to spawn generators: {ss.entropy}."
        )
        rngs = [np.random.default_rng(seed) for seed in ss.spawn(len(batches))]

        # Sample batches in parallel using a pool of processes
        start = time()
        with Pool() as pool:
            sampling_result = tqdm(
                pool.imap(
                    lambda args: self._sample(*args), zip(batches, rngs)
                ),
                total=len(batches),
                desc="Sampling from observed data distribution with rejection sampling",
                unit="batch",
            )
            samples, nb_trials = zip(*sampling_result)

        elapsed = time() - start

        # Merge batches of samples in a single array
        samples = np.vstack(samples)
        # Sum over the number of points sampled to get each batch
        nb_trials = sum(nb_trials)

        logger.info(
            "Percentage of generated samples accepted by rejection "
            f"sampling: {round(samples.shape[0] / nb_trials * 100, 2)}\n"
        )

        return samples, nb_trials, elapsed


def generate_points(
    output: str,
    name: str,
    device: str,
    model,
    bundle: np.array,
    num_generate_points: int = 1000,
    atlas_bundle: np.array = None,
    max_seeds: int = None,
    composition: Tuple[int, int] = (1, 0),
    bandwidth: float = None,
    plot_seeds_generated: bool = False,
    use_rs: bool = False,
    optimization="composition",
    gmm_n_component: int = 11,
    random_seed: int = 1234,
):
    """Generate new streamlines from an AE model and seed streamlines.

    Parameters
    ----------
    output : str
        Output path.
    name : str
        Bundle name.
    device : str
        cpu or cuda.
    model : str
        AE model for streamline compression.
    bundle : ndarray
        Bundle array (N x 256).
    num_generate_points : int
        Number of streamline to generate.
    atlas_bundle : ndarray
        Atlas bundle array (N x 256)
    max_seeds: int
        Maximum number of seed streamlines.
    composition: Tuple[int, int]
        Composition of seeds (subject bundle|atlas bundle).
    bandwidth : float
        Kernel Density bandwidth.
    plot_seeds_generated : bool
        Flag to plot umap streamlines in latent space.
    use_rs : bool
        If true will use RS instead of gaussian sampling.
    optimization : str
        Possible options are ['composition', 'max_seeds'].
    gmm_n_component : str
        Number of GMM components for RS proposal distribution.
    random_seed : int
        Random seed.

    Returns
    -------
    ndarray
    Generated streamlines.
    """

    logger.info(f"Output: {output}")
    logger.info(f"Name: {name}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {model.__class__.__name__}")

    bundle_shape = bundle.shape if isinstance(bundle, np.ndarray) else None
    logger.info(f"Bundle shape: {bundle_shape}")
    logger.info(f"num_generate_points: {num_generate_points}")

    atlas_bundle_shape = (
        atlas_bundle.shape if isinstance(atlas_bundle, np.ndarray) else None
    )
    logger.info(f"atlas_bundle shape: {atlas_bundle_shape}")
    logger.info(f"max_seeds: {max_seeds}")
    logger.info(f"Composition: {composition}")
    logger.info(f"bandwidth: {bandwidth}")
    logger.info(f"plot_seeds_generated: {plot_seeds_generated}")
    logger.info(f"use_rs: {use_rs}")
    logger.info(f"Optimization: {optimization}")
    logger.info(f"gmm_n_component: {gmm_n_component}")

    assert device in ["cuda", "cpu"]
    if max_seeds:
        assert (
            max_seeds >= 1
        ), f"The maximum number of seeds must be a positive integer. Got {max_seeds}"
    assert (
        len(composition) == 2
    ), f"Length composition must equal 2. Got {len(composition)}"
    assert (
        sum(list(composition)) == 1
    ), f"Composition sum must equal 1. Got {composition}"

    assert (bundle is not None) or (atlas_bundle is not None)

    seeds = None

    if bundle is not None:
        y_bundle = np.arange(0, len(bundle))
        dataset_bundle = OnTheFlyDataset(bundle, y_bundle)
        logger.debug(bundle.shape)

        dataloader_bundle = torch.utils.data.DataLoader(
            dataset_bundle, batch_size=128, shuffle=False
        )
        latent_bundle, y_latent_bundle = encode_data(
            dataloader_bundle, device, model
        )

        assert np.all(y_bundle == y_latent_bundle)

        if seeds is None:
            seeds = np.empty((0, latent_bundle.shape[1]))

    else:
        latent_bundle = None

    if atlas_bundle is not None:
        y_atlas_bundle = np.arange(0, len(atlas_bundle))
        dataset_atlas_bundle = OnTheFlyDataset(atlas_bundle, y_atlas_bundle)
        logger.debug(atlas_bundle.shape)

        dataloader_atlas_bundle = torch.utils.data.DataLoader(
            dataset_atlas_bundle, batch_size=128, shuffle=False
        )
        latent_atlas_bundle, y_latent_atlas_bundle = encode_data(
            dataloader_atlas_bundle, device, model
        )

        assert np.all(y_atlas_bundle == y_latent_atlas_bundle)

        if seeds is None:
            seeds = np.empty((0, latent_atlas_bundle.shape[1]))
    else:
        latent_atlas_bundle = None

    num_bundle_seeds = 0

    if bundle is not None:
        if composition[0] > 0:
            if max_seeds is not None:
                num_bundle_seeds = int(np.round(max_seeds * composition[0]))

                if optimization == "composition":
                    if composition[1] != 0 and latent_atlas_bundle is not None:
                        maximum_allowed_bundle_seeds = int(
                            np.round(
                                len(latent_atlas_bundle)
                                * composition[0]
                                / composition[1]
                            )
                        )

                        if num_bundle_seeds > maximum_allowed_bundle_seeds:
                            num_bundle_seeds = maximum_allowed_bundle_seeds

                elif optimization == "max_seeds":
                    pass

                else:
                    raise NotImplementedError(
                        f"Allowed optimizations are ['composition', 'max_seeds']"
                        f"Got {optimization}. "
                    )

                if num_bundle_seeds > len(latent_bundle):
                    bundle_seeds = latent_bundle
                else:
                    bundle_idx = np.random.choice(
                        len(latent_bundle),
                        size=num_bundle_seeds,
                        replace=False,
                    )
                    bundle_seeds = latent_bundle[bundle_idx]
            else:
                num_bundle_seeds = len(latent_bundle)
                if composition[1] != 0 and latent_atlas_bundle is not None:
                    num_bundle_seeds = int(
                        np.round(
                            len(latent_atlas_bundle)
                            * composition[0]
                            / composition[1]
                        )
                    )

                    if num_bundle_seeds > len(latent_bundle):
                        num_bundle_seeds = len(latent_bundle)

                bundle_idx = np.random.choice(
                    len(latent_bundle), size=num_bundle_seeds, replace=False
                )
                bundle_seeds = latent_bundle[bundle_idx]

            seeds = np.vstack((seeds, bundle_seeds))
            num_bundle_seeds = len(bundle_seeds)

        elif composition[0] == 0 and latent_atlas_bundle is None:
            if max_seeds is not None:
                num_bundle_seeds = max_seeds

                if num_bundle_seeds > len(latent_bundle):
                    bundle_seeds = latent_bundle
                else:
                    bundle_idx = np.random.choice(
                        len(latent_bundle),
                        size=num_bundle_seeds,
                        replace=False,
                    )
                    bundle_seeds = latent_bundle[bundle_idx]

            else:
                bundle_seeds = latent_bundle

            seeds = np.vstack((seeds, bundle_seeds))
            num_bundle_seeds = len(bundle_seeds)

    num_atlas_seeds = 0

    if atlas_bundle is not None:
        if composition[1] > 0:
            if max_seeds is not None:
                num_atlas_seeds = int(np.round(max_seeds * composition[1]))

                if optimization == "composition":
                    if composition[0] != 0 and latent_bundle is not None:
                        maximum_allowed_atlas_seeds = int(
                            np.round(
                                len(latent_bundle)
                                * composition[1]
                                / composition[0]
                            )
                        )

                        if num_atlas_seeds > maximum_allowed_atlas_seeds:
                            num_atlas_seeds = maximum_allowed_atlas_seeds

                elif optimization == "max_seeds":
                    pass

                else:
                    raise NotImplementedError(
                        f"Allowed optimizations are ['composition', 'max_seeds']"
                        f"Got {optimization}. "
                    )

                if num_atlas_seeds > len(latent_atlas_bundle):
                    atlas_seeds = latent_atlas_bundle
                else:
                    atlas_idx = np.random.choice(
                        len(latent_atlas_bundle),
                        size=num_atlas_seeds,
                        replace=False,
                    )
                    atlas_seeds = latent_atlas_bundle[atlas_idx]
            else:
                num_atlas_seeds = len(latent_atlas_bundle)
                if composition[0] != 0 and latent_bundle is not None:
                    num_atlas_seeds = int(
                        np.round(
                            len(latent_bundle)
                            * composition[1]
                            / composition[0]
                        )
                    )

                    if num_atlas_seeds > len(latent_atlas_bundle):
                        num_atlas_seeds = len(latent_atlas_bundle)

                atlas_idx = np.random.choice(
                    len(latent_atlas_bundle),
                    size=num_atlas_seeds,
                    replace=False,
                )
                atlas_seeds = latent_atlas_bundle[atlas_idx]

            seeds = np.vstack((seeds, atlas_seeds))
            num_atlas_seeds = len(atlas_seeds)

        elif composition[1] == 0 and latent_bundle is None:
            if max_seeds is not None:
                num_atlas_seeds = max_seeds

                if num_atlas_seeds > len(latent_atlas_bundle):
                    atlas_seeds = latent_atlas_bundle
                else:
                    atlas_idx = np.random.choice(
                        len(latent_atlas_bundle),
                        size=num_atlas_seeds,
                        replace=False,
                    )
                    atlas_seeds = latent_atlas_bundle[atlas_idx]

            else:
                atlas_seeds = latent_atlas_bundle

            seeds = np.vstack((seeds, atlas_seeds))
            num_atlas_seeds = len(atlas_seeds)

    total_seeds = len(seeds)
    atlas_seeds_ratio = num_atlas_seeds / total_seeds
    bundle_seeds_ratio = num_bundle_seeds / total_seeds

    logger.info(
        f"Seeds composition absolute values: Bundle: {num_bundle_seeds}, Atlas: {num_atlas_seeds}"
    )
    logger.info(
        "Seeds composition ratio (Bundle/Atlas): {:.3f}/{:.3f}".format(
            bundle_seeds_ratio, atlas_seeds_ratio
        )
    )

    if use_rs:
        assert len(seeds) > gmm_n_component
        samples_kde, _, _ = RejectionSampler(
            data=seeds,
            kde_bw=bandwidth,
            kde_bw_factor=1,
            kernel="gaussian",
            scaling_mode="3rd_quartile",
            kde_bw_auto_estimation="Silverman1986",
            proposal_distribution_name="GMM",
            proposal_distribution_params={"n_components": gmm_n_component},
            bundle_name=name,
            output=output,
            cluster_estimation="BIC",
        ).sample(num_generate_points, entropy=random_seed)
    else:

        if bandwidth is None:
            bandwidth = 0.025
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(seeds)
        samples_kde = kde.sample(
            n_samples=num_generate_points, random_state=42
        )

    if plot_seeds_generated:
        reducer = umap.UMAP(
            random_state=0,
            min_dist=0.9,
            n_neighbors=100,
            n_epochs=1000,
            verbose=True,
        )
        umap_results_clusters = reducer.fit_transform(
            np.vstack((seeds, samples_kde))
        )
        plt.figure(figsize=(12, 10))

        plt.title(
            f"Bundle name: {name}   Epochs: 1000   min_dist: 0.9   n_neigh: 100 "
        )

        if num_atlas_seeds > 0:
            plt.scatter(
                umap_results_clusters[0 : len(atlas_seeds), 0],
                umap_results_clusters[0 : len(atlas_seeds), 1],
                label="Atlas seeds",
            )
            plt.scatter(
                umap_results_clusters[len(atlas_seeds) : len(seeds), 0],
                umap_results_clusters[len(atlas_seeds) : len(seeds), 1],
                label=f"{name} seeds",
            )
        else:
            plt.scatter(
                umap_results_clusters[: len(seeds), 0],
                umap_results_clusters[: len(seeds), 1],
                label=f"{name} seeds",
            )
        plt.legend()
        plt.savefig(pjoin(output, f"{name}_seeds.png"))

        plt.scatter(
            umap_results_clusters[len(seeds) :, 0],
            umap_results_clusters[len(seeds) :, 1],
            label="Generated",
        )

        plt.legend()
        plt.savefig(pjoin(output, f"{name}_generated.png"))

    dataset = OnTheFlyDataset(
        samples_kde, np.repeat(1, len(samples_kde)), to_transpose=False
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2048)
    samples = []
    for i, (data, _) in enumerate(tqdm(dataloader)):
        data = data.to(device)

        sample = model.decode(data).cpu().detach().numpy()

        samples.append(sample)

    X_decoded = np.vstack(samples)

    return X_decoded.transpose((0, 2, 1))
