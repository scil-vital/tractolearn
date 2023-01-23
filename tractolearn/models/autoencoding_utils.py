# -*- coding: utf-8 -*-


from typing import Tuple

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader


def encode_data(
    latent_space_loader: DataLoader,
    device: torch.device,
    model: Module,
    limit_batch: int = None,
) -> Tuple[np.array, np.array]:
    """Encode streamlines in a Dataloader object in a smaller latent space.

    Parameters
    ----------
    latent_space_loader : DataLoader
        Dataloader containing streamlines to encode.
    device : torch.device
        Device to use.
    model : Module
        Deep learning model.
    limit_batch : int
        Encode a limit number of batch from the latent_space_loader.

    Returns
    -------
    Tuple[np.array, np.array]
        Latent space and streamline bundle classes.
    """

    latent_space_samples = []
    latent_space_samples_classes = []
    for i, (data, data_classes) in enumerate(latent_space_loader):

        if i == limit_batch:
            break

        data = data.to(device)

        latent_sample = model.encode(data).cpu().detach().numpy()

        latent_space_samples.append(latent_sample)
        latent_space_samples_classes.append(
            data_classes.cpu().detach().numpy().astype(int)
        )

    X_latent = np.vstack(latent_space_samples)
    y_latent = np.hstack(latent_space_samples_classes)

    return X_latent, y_latent
