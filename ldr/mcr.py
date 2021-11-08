"""Helper functions for computing coding rates."""

from typing import Optional

import jax
from jax import numpy as jnp


def logdet_hermitian(A: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-determinant of a Hermitian matrix."""
    M, N = A.shape
    assert M == N
    return 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(A))))


def coding_rate(Z: jnp.ndarray, epsilon_sq: float) -> jnp.ndarray:
    """Compute the rate distortion function.

    This can be interpreted as a ball packing problem: given the volume defined by some
    set of vectors Z, how many balls of radius epsilon_sq fit inside? The log of this
    number will be proportional to the bit count `R`."""
    ZZ_T = jnp.einsum("ni,nj->nij", Z, Z)
    return coding_rate_from_autocorrelations(ZZ_T, epsilon_sq=epsilon_sq, mask=None)


def coding_rate_from_autocorrelations(
    ZZ_T: jnp.ndarray,
    epsilon_sq: float,
    mask: Optional[jnp.ndarray] = None,  # Pi in LDR paper.
) -> jnp.ndarray:
    """Compute the rate distortion function. Same as `coding_rate()`, but operates on a
    set of sample autocorrelation matrices (each of which should be PSD and rank 1)."""

    N, D, D_ = ZZ_T.shape
    assert D == D_
    assert mask is None or mask.shape == (N,)

    if mask is not None:
        N = jnp.sum(mask)
        mask = mask[:, None, None]  # For broadcasting.

    alpha = D / (N * epsilon_sq)
    cov = jnp.eye(D) + alpha * jnp.sum(ZZ_T, axis=0, where=mask)
    assert cov.shape == (D, D)

    return logdet_hermitian(cov)


def coding_rate_distance_from_autocorrelations(
    ZZ_T_0: jnp.ndarray,
    ZZ_T_1: jnp.ndarray,
    epsilon_sq: float,
    mask_0: Optional[jnp.ndarray] = None,
    mask_1: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute a coding rate "distance" metric between two sets."""

    N1, D, D_ = ZZ_T_0.shape
    N2, D__, D___ = ZZ_T_1.shape
    assert D == D_ == D__ == D___
    assert (mask_0 is None) == (mask_1 is None)

    ZZ_T_union = jnp.concatenate([ZZ_T_0, ZZ_T_1], axis=0)
    if mask_0 is not None:
        mask_union = jnp.concatenate([mask_0, mask_1], axis=0)
    else:
        mask_union = None

    return coding_rate_from_autocorrelations(ZZ_T_union, epsilon_sq, mask_union) - 0.5 * (
        coding_rate_from_autocorrelations(ZZ_T_0, epsilon_sq, mask_0)
        + coding_rate_from_autocorrelations(ZZ_T_1, epsilon_sq, mask_1)
    )


def multiclass_coding_rate_from_autocorrelations(
    ZZ_T: jnp.ndarray,
    one_hot_labels: jnp.ndarray,
    epsilon_sq: float,
) -> jnp.ndarray:
    """The difference between the coding rate for the whole volume and the coding rates
    for each subset.

    Note that to produce highly incoherent subspaces, we want to perform coding rate
    _reduction_, that is, this quantity should generally be *maximized*."""

    # Check shapes.
    N, K = one_hot_labels.shape
    N_, D, D_ = ZZ_T.shape
    assert N == N_
    assert D == D_

    # Coding rate for the whole minibatch.
    coding_rate_whole = coding_rate_from_autocorrelations(ZZ_T, epsilon_sq)

    # Sum across coding rate for each subset.
    class_ratios = jnp.sum(one_hot_labels, axis=0) / N
    coding_rate_per_class = jax.vmap(
        lambda mask: coding_rate_from_autocorrelations(ZZ_T, epsilon_sq, mask)
    )(
        one_hot_labels.T,  # (K, N)
    )
    assert coding_rate_per_class.shape == (K,)
    coding_rate_per_class = jnp.sum(class_ratios * coding_rate_per_class)

    return (coding_rate_whole - coding_rate_per_class) / 2.0
