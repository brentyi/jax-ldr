from typing import Any, Optional, Protocol

import jax
from jax import numpy as jnp

from .protocols import Decoder, Encoder

Pytree = Any


def minimaxing_ldr_score(
    X: jnp.ndarray,
    one_hot_labels: jnp.ndarray,
    f: Encoder,
    theta: Pytree,  # Encoder parameters
    g: Decoder,
    eta: Pytree,  # Decoder parameters
    epsilon: float,
):
    """Score for minimax game proposed in `Closed-Loop Data Transcription to an LDR via
    Minimaxing Rate Reduction`, by Anonymous et al."""

    N, K = one_hot_labels.shape

    # Compute codes: from original and transcribed.
    Z = f(X, theta)
    Z_hat = f(g(f(X, theta), eta), theta)

    # Compute Gram matrices.
    ZZ_T = jnp.einsum(Z, Z, "ni,nj->nij")
    ZZ_T_hat = jnp.einsum(Z_hat, Z_hat, "ni,nj->nij")

    # Our score is made of 3 parts, consisting of coding rates that correspond to: an
    # expansive encoding score, compressive decoding score, and a sum over
    # per-class contrastive/contractive terms.
    expansive_encode = multiclass_coding_rate_from_gram(ZZ_T, one_hot_labels, epsilon)
    compressive_decode = multiclass_coding_rate_from_gram(
        ZZ_T_hat, one_hot_labels, epsilon
    )
    contrastive_contractive_terms = jax.vmap(
        lambda mask: coding_rate_distance_from_gram(
            ZZ_T_1=ZZ_T,
            ZZ_T_2=ZZ_T_hat,
            epsilon=epsilon,
            mask_1=mask,
            mask_2=mask,
        )
    )(
        one_hot_labels.T  # (K, N)
    )
    assert contrastive_contractive_terms.shape == (K,)

    score = (
        expansive_encode + compressive_decode + jnp.sum(contrastive_contractive_terms)
    )
    return score


def coding_rate_from_gram(
    ZZ_T: jnp.ndarray,
    epsilon: float,
    mask: Optional[jnp.ndarray] = None,  # Pi in paper
) -> jnp.ndarray:
    """Same as coding_rate(), but with a (possibly cached) Gram matrix."""

    N, D, D_ = ZZ_T.shape
    assert D == D_
    assert mask is None or mask.shape == (N,)

    if mask is not None:
        N = jnp.sum(mask)

    alpha = D / (N * epsilon ** 2)
    cov = jnp.eye(D) + alpha * jnp.sum(ZZ_T, axis=0, where=mask)
    assert cov.shape == (D, D)

    return jnp.log(jnp.linalg.det(cov))


def coding_rate_distance_from_gram(
    ZZ_T_1: jnp.ndarray,
    ZZ_T_2: jnp.ndarray,
    epsilon: float,
    mask_1: Optional[jnp.ndarray] = None,
    mask_2: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute a coding rate "distance" metric between two sets."""

    N1, D, D_ = ZZ_T_1.shape
    N2, D__, D___ = ZZ_T_2.shape
    assert D == D_ == D__ == D___
    assert mask_1 is None or mask_2 is None

    ZZ_T_union = jnp.concatenate([ZZ_T_1, ZZ_T_2], axis=0)
    if mask_1 is not None:
        mask_union = jnp.concatenate([mask_1, mask_2], axis=0)
    else:
        mask_union = None

    return coding_rate_from_gram(ZZ_T_union, epsilon, mask_union) - 0.5 * (
        coding_rate_from_gram(ZZ_T_1, epsilon, mask_1)
        + coding_rate_from_gram(ZZ_T_2, epsilon, mask_2)
    )


def multiclass_coding_rate_from_gram(
    ZZ_T: jnp.ndarray,
    one_hot_labels: jnp.ndarray,
    epsilon: float,
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
    coding_rate_whole = coding_rate_from_gram(ZZ_T, epsilon)

    # Sum across coding rate for each subset.
    subsets_terms = jax.vmap(lambda mask: coding_rate_from_gram(ZZ_T, epsilon, mask))(
        one_hot_labels.T,  # (K, N)
    )
    assert subsets_terms.shape == (K,)
    coding_rate_subsets = jnp.sum(subsets_terms)

    return coding_rate_whole - coding_rate_subsets
