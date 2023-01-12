# -*- coding: utf-8 -*-
import numpy as np


def is_normalized(n):
    """Checks if a number is a normalized.

    Parameters
    ----------
    n : float
        Number to be checked.

    Returns
    -------
    bool
      True if the input number is normalized; False otherwise.
    """

    return 0.0 <= n <= 1.0


def unit_vector(vector):
    """Returns the unit vector of the vector.
    NaN values might be returned for null values.
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        return vector / np.sqrt(np.sum(vector**2, axis=-1, keepdims=True))


def compute_angle(u, v, use_smallest=True, nan_to_zero=True):
    """Compute the angle between the input vectors.
    If both u and v have the same shape, angles are computed in a pairwise
    fashion. Otherwise, P angles are computed for each vector in `u` if P
    vectors are provided in `v`. The former case can be found, for example,
    when computing the angle between streamlined endpoint dirs and their
    matched surface vertex normal vectors; the latter case can be found when
    computing the angle for N streamline segment directions and the local
    orientation (e.g. fODF) peaks, having P peaks at each streamline segment.
    Note that although a vector is supposed to have a direction, in the context
    of tractography, the directionality is not implied by the local
    orientation. In such cases, the smallest angle between the lines might be
    desirable.
    Parameters
    ----------
    u : ndarray (N, 3)
        Container of `N` 3D vectors.
    v : ndarray (N, 3) or (P, N, 3)
        Container of `N` 3D vectors or `P` times `N` 3D vectors.
    use_smallest : bool, optional
        True to get the smallest value between the computed angle and its
        supplementary. This might be especially relevant if vectors are assumed
        to be undirected, such as local orientation (e.g. fODF) peaks.
    Returns
    -------
    angles : ndarray (N,) or (P, N)
        The angle between the vectors.
    """

    if not u.size or not v.size:
        return np.empty((0,))

    # Normalize the vectors in case
    u = unit_vector(u)
    v = unit_vector(v)

    # Given that `unit_vector` causes NaNs when vectors are 0, set such NaNs to
    # 0
    if nan_to_zero:
        u = np.nan_to_num(u)
        v = np.nan_to_num(v)

    u_shape = u.shape
    v_shape = v.shape

    if u_shape == v_shape:
        subscripts = "ij,ij->i"
    elif u_shape == (v_shape[1], v_shape[2]):
        # Subscripts to compute the dot/inner product between all vectors
        # in u (e.g. streamline segment direction) and each vector in v (e.g.
        # fODF peaks)
        subscripts = "ik,jik->ij"
    else:
        raise ValueError(
            "Unexpected shapes. Available implementations expect input data "
            "to have either same shape or `v` to contain multiple vectors for "
            "each vector in `u`. "
            f"Found: shape `u`: {u_shape}; shape `v`: {v_shape}"
        )

    dot = np.einsum(subscripts, u, v)
    # Due to numerical instabilities, np.einsum can yield > 1 or < -1
    # values, so clip them to the [-1, 1] range.
    dot = np.clip(dot, -1, 1)

    # We should not have NaNs, but check in case
    assert not np.isnan(dot).any(), dot

    if use_smallest:
        # Take the absolute value of the alignment. Allows to get the smallest
        # value between the computed angle and its supplementary by always
        # using positive arguments (i.e. values in first quadrants in practice
        # here) for the arccos.
        # This is especially relevant when dealing with local orientation (e.g.
        # fODF) peaks, since peaks are non-directed.
        # Also, note that np.rad2deg(np.arccos(np.abs(0))) == 90.
        dot = np.abs(dot)

    # ToDo
    # Not sure why but this looks like it computes the complementary to
    # 90 degrees that we will be setting as a limit
    # Get the angle in degrees (expeted range is [0, 180/360[)
    # angles = np.arccos(dot[0, :] / (np.sqrt(dot[1, :]) * np.sqrt(dot[2, :])))
    angles = np.rad2deg(np.arccos(dot))

    return angles.T
