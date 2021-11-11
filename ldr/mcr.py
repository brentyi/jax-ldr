"""Helper functions for computing coding rates."""

from typing import Optional

import jax
from jax import numpy as jnp

_count = 0


def logdet_hermitian(A: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-determinant of a Hermitian matrix."""
    M, N = A.shape
    assert M == N
    return 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(A))))


def coding_rate_from_ZTZ(ZTZ: jnp.ndarray, N: int, epsilon_sq: float) -> jnp.ndarray:
    """Compute the rate distortion function, from `Z^T @ Z`.

    This can be interpreted as a ball packing problem: given the volume defined by some
    set of vectors Z, how many balls of radius epsilon fit inside? The log of this
    number will be proportional to the bit count `R`."""
    D, D_ = ZTZ.shape
    assert D == D_
    alpha = D / (N * epsilon_sq)
    A = jnp.eye(D) + alpha * ZTZ
    assert A.shape == (D, D)
    return logdet_hermitian(A)


def coding_rate(Z: jnp.ndarray, epsilon_sq: float) -> jnp.ndarray:
    """Compute the rate distortion function.

    This can be interpreted as a ball packing problem: given the volume defined by some
    set of vectors Z, how many balls of radius epsilon fit inside? The log of this
    number will be proportional to the bit count `R`."""
    N, D = Z.shape
    return coding_rate_from_ZTZ(Z.T @ Z, N, epsilon_sq)


def multiclass_coding_rate_delta(
    Z: jnp.ndarray,
    coding_rate_per_class: jnp.ndarray,
    one_hot_labels: jnp.ndarray,
    epsilon_sq: float,
) -> jnp.ndarray:
    """The difference between the coding rate for the whole volume and the coding rates
    for each subset.

    Note that to produce highly incoherent subspaces, we want to perform coding rate
    _reduction_, that is, this quantity should generally be *maximized*."""
    # Check shapes.
    N, D = Z.shape
    N_, K = one_hot_labels.shape
    assert N == N_
    assert coding_rate_per_class.shape == (K,)

    # Coding rate for the whole minibatch.
    coding_rate_whole = coding_rate(Z, epsilon_sq)

    # Sum across coding rate for each subset.
    class_ratios = jnp.sum(one_hot_labels, axis=0) / N
    sum_across_classes_term = jnp.sum(class_ratios * coding_rate_per_class)

    return (coding_rate_whole - sum_across_classes_term) / 2.0
