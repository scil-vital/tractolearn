import torch

from tractolearn.config.experiment import (
    LossFunctionTypes,
    ExperimentKeys,
    LearningTask,
)
from tractolearn.utils.losses import (
    loss_function_ae,
    loss_contrastive_lecun_classes,
    loss_triplet_classes,
    loss_triplet_hierarchical_classes,
)


def forward_ae(model, loss_fn, device, batch):
    """Take a labeled batch from HDF5Dataset, encode it, decode it, and compute the appropriate loss."""
    x, y = batch
    x = x.to(device, dtype=torch.float32)
    x_recon = model.forward(x)
    loss = loss_fn(x_recon, x)
    return loss


def forward_contrastive(model, loss_fn, device, streamline_batch):
    """Take a batch of streamlines from ContrastiveDataset, encode it, and compute a contrastive loss."""
    streamline_batch = streamline_batch.to(device, dtype=torch.float32)
    z = model.encode(streamline_batch)
    loss = loss_fn(z)
    return loss


def forward_ae_contrastive(model, loss_fn, device, streamline_batch):
    """Take a batch of streamlines from ContrastiveDataset, encode it, decode it, and compute the supplied loss fn.

    The loss fn takes in x, x_recon and z; which allows to compute both a reconstruction loss and a constrastive loss.
    """
    x = streamline_batch.to(device, dtype=torch.float32)
    z = model.encode(x)
    x_recon = model.decode(z)
    loss = loss_fn(x, x_recon, z)
    return loss


def make_forward(model, device, experiment_dict):
    """Make a forward pass function by combining a model execution and a loss function computation."""

    loss_fn = {
        LossFunctionTypes.ae: loss_function_ae,
        LossFunctionTypes.contrastive_lecun_classes: loss_contrastive_lecun_classes,
        LossFunctionTypes.ae_contrastive_lecun_classes: (
            loss_function_ae,
            loss_contrastive_lecun_classes,
        ),
        LossFunctionTypes.ae_triplet_classes: (
            loss_function_ae,
            loss_triplet_classes,
        ),
        LossFunctionTypes.ae_triplet_hierarchical_classes: (
            loss_function_ae,
            loss_triplet_hierarchical_classes,
        ),
    }[experiment_dict[ExperimentKeys.LOSS_FN]]

    if experiment_dict[ExperimentKeys.TASK] == LearningTask.ae:

        def forward(batch):
            return forward_ae(model, loss_fn, device, batch)

    elif experiment_dict[ExperimentKeys.TASK] == LearningTask.contrastive_lecun_classes:

        def loss_fn_configured(z):
            return loss_fn(
                z,
                margin=experiment_dict["contrastive_margin"],
            )

        def forward(batch):
            return forward_contrastive(model, loss_fn_configured, device, batch)

    elif experiment_dict[ExperimentKeys.TASK] in [
        LearningTask.ae_contrastive_lecun_classes,
        LearningTask.ae_triplet_classes,
        LearningTask.ae_triplet_hierarchical_classes,
    ]:

        def loss_fn_configured(x, x_recon, z):
            loss_fn_ae, loss_fn_contrastive = loss_fn
            loss_value_ae = loss_fn_ae(x_recon, x)

            if (
                experiment_dict[ExperimentKeys.TASK]
                == LearningTask.ae_contrastive_lecun_classes
            ):
                loss_value_contrastive = loss_fn_contrastive(
                    z, margin=experiment_dict["contrastive_margin"]
                )
            else:
                loss_value_contrastive = loss_fn_contrastive(
                    z,
                    margin=experiment_dict["contrastive_margin"],
                    metric=experiment_dict[ExperimentKeys.DISTANCE_FUNCTION],
                    swap=experiment_dict[ExperimentKeys.TO_SWAP],
                )
            return (
                loss_value_ae
                + experiment_dict["contrastive_loss_weight"] * loss_value_contrastive
            )

        def forward(batch):
            return forward_ae_contrastive(model, loss_fn_configured, device, batch)

    else:
        raise NotImplementedError

    return forward
