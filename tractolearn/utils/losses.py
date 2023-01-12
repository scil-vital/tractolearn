# -*- coding: utf-8 -*-
from typing import Optional, Callable, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import _reduction as _Reduction

reconstruction_loss = nn.MSELoss(reduction="sum")


def loss_function_vae(recon_x, x, mu, logvar):
    mse = reconstruction_loss(recon_x, x)

    # See Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # kld = torch.sum(kld_element).__mul__(-0.5)

    return mse + kld


def loss_function_ae(recon_x, x):
    mse = reconstruction_loss(recon_x, x)

    return mse


def loss_contrastive_lecun_classes(z, margin):
    """Attract pairs of latent vectors of the same class, repulse pairs of different classes

    This is the contrastive loss as defined by Hadsell, Chopra and LeCun, 2006. However, in their paper, they don't
    use class information.

    Parameters:
        z       Tensor of size (num_pos_pairs * 2 + num_neg_pairs * 2, latent_size). This is the batch format output by
                ContrastiveDataset.
        margin  The margin hyperparameter
    """

    # Decompose the batch to recover the positive and negative pairs
    quarter_batch_size = z.shape[0] // 4
    pos1 = z[:quarter_batch_size]
    pos2 = z[quarter_batch_size : quarter_batch_size * 2]
    neg1 = z[quarter_batch_size * 2 : quarter_batch_size * 3]
    neg2 = z[quarter_batch_size * 3 :]

    # Compute loss for positive pairs
    loss_pos = (pos1 - pos2) ** 2.0

    # Compute loss for negative pairs
    loss_neg = torch.maximum(margin - (neg1 - neg2).abs(), torch.tensor(0.0)) ** 2

    return torch.sum(loss_pos + loss_neg)


def loss_triplet_classes(z, margin, metric="l2", swap=False):
    """Anchors attract positive values and repulse negatives values

    Parameters:
        z       Tensor of size (num_pos_pairs * 2 + num_neg_pairs * 2, latent_size). This is the batch format output by
                TripletDataset.
        margin  The margin hyperparameter
    """

    # Decompose the batch to recover the positive and negative pairs
    third_batch_size = z.shape[0] // 3
    anchors = z[:third_batch_size]
    positives = z[third_batch_size : third_batch_size * 2]
    negatives = z[third_batch_size * 2 :]

    if metric == "l2":
        distance_function = nn.PairwiseDistance()

    elif metric == "cosine_similarity":

        def distance_function(x, y):
            return 1.0 - F.cosine_similarity(x, y)

    else:
        raise NotImplementedError

    return nn.TripletMarginWithDistanceLoss(
        distance_function=distance_function,
        margin=margin,
        reduction="sum",
        swap=swap,
    )(anchors, positives, negatives)


def loss_triplet_hierarchical_classes(z, margin, metric="l2", swap=False):
    if metric == "l2":
        distance_function = nn.PairwiseDistance()

    elif metric == "cosine_similarity":

        def distance_function(x, y):
            return 1.0 - F.cosine_similarity(x, y)

    else:
        raise NotImplementedError

    sixth_batch_size = z.shape[0] // 6

    anchor = z[:sixth_batch_size]
    positives = [
        z[sixth_batch_size * (i + 1) : sixth_batch_size * (i + 2)] for i in range(4)
    ]
    negative = z[-sixth_batch_size:]

    return triplet_margin_with_distance_loss_hierarchical(
        anchor,
        positives=positives,
        negative=negative,
        distance_function=distance_function,
        margin=margin,
        reduction="sum",
    )


def triplet_margin_with_distance_loss_hierarchical(
    anchor: Tensor,
    positives: List[Tensor],
    negative: Tensor,
    distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    margin: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    r"""
    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.
    """

    negative_dist = distance_function(anchor, negative)

    cumulative_loss = None

    for i, l in enumerate(positives):
        positive_dist = distance_function(anchor, l)

        if cumulative_loss is None:
            cumulative_loss = torch.exp(1 / (torch.tensor(i) + 1)) * torch.clamp(
                positive_dist - negative_dist + (len(positives) - i) * margin,
                min=0.0,
            )
        else:
            cumulative_loss += torch.exp(1 / (torch.tensor(i) + 1)) * torch.clamp(
                positive_dist - negative_dist + (len(positives) - i) * margin,
                min=0.0,
            )

    cumulative_loss /= len(positives)

    reduction_enum = _Reduction.get_enum(reduction)
    if reduction_enum == 1:
        return cumulative_loss.mean()
    elif reduction_enum == 2:
        return cumulative_loss.sum()
    else:
        return cumulative_loss
