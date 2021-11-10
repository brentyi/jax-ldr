"""Helper functions for computing coding rates."""

from typing import Optional

import jax
from jax import numpy as jnp

_count = 0


def logdet_hermitian(A: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-determinant of a Hermitian matrix."""
    M, N = A.shape
    assert M == N
    global _count
    _count = _count + 1
    print(_count)
    return 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(A))))


def coding_rate(Z: jnp.ndarray, epsilon_sq: float) -> jnp.ndarray:
    """Compute the rate distortion function.

    This can be interpreted as a ball packing problem: given the volume defined by some
    set of vectors Z, how many balls of radius epsilon fit inside? The log of this
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
    precomputed_ZZ_T_0_coding_rate: Optional[jnp.ndarray] = None,
    precomputed_ZZ_T_1_coding_rate: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute a coding rate "distance" metric between two sets.

    Optionally accepts precomputed coding rates for each set."""
    N1, D, D_ = ZZ_T_0.shape
    N2, D__, D___ = ZZ_T_1.shape
    assert D == D_ == D__ == D___
    assert (mask_0 is None) == (mask_1 is None)

    ZZ_T_union = jnp.concatenate([ZZ_T_0, ZZ_T_1], axis=0)
    if mask_0 is not None:
        mask_union = jnp.concatenate([mask_0, mask_1], axis=0)
    else:
        mask_union = None

    return coding_rate_from_autocorrelations(
        ZZ_T_union, epsilon_sq, mask_union
    ) - 0.5 * (
        (
            coding_rate_from_autocorrelations(ZZ_T_0, epsilon_sq, mask_0)
            if precomputed_ZZ_T_0_coding_rate is None
            else precomputed_ZZ_T_0_coding_rate
        )
        + (
            coding_rate_from_autocorrelations(ZZ_T_1, epsilon_sq, mask_1)
            if precomputed_ZZ_T_1_coding_rate is None
            else precomputed_ZZ_T_1_coding_rate
        )
    )


def coding_rate_per_class(
    ZZ_T: jnp.ndarray,
    one_hot_labels: jnp.ndarray,
    epsilon_sq: float,
) -> jnp.ndarray:
    """Compute a coding rate vector of shape (K,); one for each class."""
    N, K = one_hot_labels.shape
    out = jax.vmap(
        lambda mask: coding_rate_from_autocorrelations(ZZ_T, epsilon_sq, mask)
    )(
        one_hot_labels.T,  # (K, N)
    )
    assert out.shape == (K,)
    return out


def multiclass_coding_rate_delta_from_autocorrelations(
    ZZ_T: jnp.ndarray,
    one_hot_labels: jnp.ndarray,
    epsilon_sq: float,
    precomputed_coding_rate_per_class: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """The difference between the coding rate for the whole volume and the coding rates
    for each subset.

    Note that to produce highly incoherent subspaces, we want to perform coding rate
    _reduction_, that is, this quantity should generally be *maximized*."""
    # Check shapes.
    N, D, D_ = ZZ_T.shape
    N_, K = one_hot_labels.shape
    assert D == D_
    assert N == N_
    if precomputed_coding_rate_per_class is not None:
        assert precomputed_coding_rate_per_class.shape == (K,)

    # Coding rate for the whole minibatch.
    coding_rate_whole = coding_rate_from_autocorrelations(ZZ_T, epsilon_sq)

    # Sum across coding rate for each subset.
    class_ratios = jnp.sum(one_hot_labels, axis=0) / N
    sum_across_classes_term = jnp.sum(
        class_ratios
        * (
            coding_rate_per_class(ZZ_T, one_hot_labels, epsilon_sq)
            if precomputed_coding_rate_per_class is None
            else precomputed_coding_rate_per_class
        )
    )

    return (coding_rate_whole - sum_across_classes_term) / 2.0
