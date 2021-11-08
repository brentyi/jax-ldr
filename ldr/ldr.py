from typing import Any, Tuple

import jax
from jax import numpy as jnp

from . import mcr

Pytree = Any


def ldr_score(
    Z,  # f(x)
    Z_hat,  # f(g(f(X)))
    one_hot_labels: jnp.ndarray,
    epsilon_sq: float,
) -> jnp.ndarray:
    """Score for the minimax game proposed in `Closed-Loop Data Transcription to an LDR
    via Minimaxing Rate Reduction`, by Anonymous et al."""
    a, b, c = ldr_score_terms(Z, Z_hat, one_hot_labels, epsilon_sq)
    return a + b + c


def ldr_score_terms(
    Z,  # f(x)
    Z_hat,  # f(g(f(X)))
    one_hot_labels: jnp.ndarray,
    epsilon_sq: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the three terms used as a score for the minimax game proposed in
    `Closed-Loop Data Transcription to an LDR via Minimaxing Rate Reduction`, by
    Anonymous et al.

    The three terms, returned in order, are:
    - An expansive encoding term
    - A compressive decoding term
    - A contrastive-contractive term

    This is the same as `ldr_score()`, but doesn't sum the three terms. Might be useful
    for logging/inspection/debugging."""

    N, K = one_hot_labels.shape

    # Compute autocorrelation matrices.
    ZZ_T = jnp.einsum("ni,nj->nij", Z, Z)
    ZZ_T_hat = jnp.einsum("ni,nj->nij", Z_hat, Z_hat)

    # Our score is made of 3 parts, consisting of coding rates that correspond to: an
    # expansive encoding score, compressive decoding score, and a sum over per-class
    # contrastive/contractive terms.
    expansive_encode = mcr.multiclass_coding_rate_from_autocorrelations(
        ZZ_T, one_hot_labels, epsilon_sq
    )
    compressive_decode = mcr.multiclass_coding_rate_from_autocorrelations(
        ZZ_T_hat, one_hot_labels, epsilon_sq
    )
    contrastive_contractive_terms = jax.vmap(
        lambda mask: mcr.coding_rate_distance_from_autocorrelations(
            ZZ_T_0=ZZ_T,
            ZZ_T_1=ZZ_T_hat,
            epsilon_sq=epsilon_sq,
            mask_0=mask,
            mask_1=mask,
        )
    )(
        one_hot_labels.T  # (K, N)
    )
    assert contrastive_contractive_terms.shape == (K,)

    return expansive_encode, compressive_decode, jnp.sum(contrastive_contractive_terms)
