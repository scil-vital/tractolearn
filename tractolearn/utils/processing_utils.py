# -*- coding: utf-8 -*-


def postprocess(x_reconstructed, isocenter, volume):

    x_reconstructed_unnorm = x_reconstructed * volume
    x_reconstructed_shifted_unnorm = x_reconstructed_unnorm + isocenter

    return x_reconstructed_shifted_unnorm
