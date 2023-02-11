# -*- coding: utf-8 -*-

from tractolearn.models.track_ae_cnn1d_incr_feat_strided_conv_fc_upsamp_reflect_pad_pytorch import (
    IncrFeatStridedConvFCUpsampReflectPadAE,
)


def get_model(model_name, latent_space_dims, device):
    """Autoencoders (AE) [1]_; Variational Autoencoders (VAE) [2]_;
    Reparamaterization trick [2]_.

    References
    ----------
    .. [1] Hinton, G.E., and Salakhutdinov R. R. Reducing the Dimensionality of
           Data with Neural Networks. Science 28 Jul 2006: 313(5786),
           pp. 504-507 DOI: 10.1126/science.1127647
    .. [2] Kingma, D.P., and Welling, M. Auto-Encoding Variational Bayes.
           Proceedings of the 2nd International Conference on Learning
           Representations (ICLR). 2014
    """

    if model_name == "IncrFeatStridedConvFCUpsampReflectPadAE":
        model = IncrFeatStridedConvFCUpsampReflectPadAE(latent_space_dims).to(
            device
        )
    else:
        print("Model not implemented")
        model = None

    return model
