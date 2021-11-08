from typing import Any, Optional

import jax
from jax import numpy as jnp

from . import mcr, protocols

Pytree = Any


def ldr_score(
    Z,  # f(x)
    Z_hat,  # f(g(f(X)))
    one_hot_labels: jnp.ndarray,
    epsilon_sq: float,
) -> jnp.ndarray:
    """Score for the minimax game proposed in `Closed-Loop Data Transcription to an LDR
    via Minimaxing Rate Reduction`, by Anonymous et al."""

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

    score = (
        expansive_encode + compressive_decode + jnp.sum(contrastive_contractive_terms)
    )
    return score

def ldr_score_parts(
    Z,  # f(x)
    Z_hat,  # f(g(f(X)))
    one_hot_labels: jnp.ndarray,
    epsilon_sq: float,
) -> jnp.ndarray:
    """Score for the minimax game proposed in `Closed-Loop Data Transcription to an LDR
    via Minimaxing Rate Reduction`, by Anonymous et al."""

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

    return expansive_encode , compressive_decode , jnp.sum(contrastive_contractive_terms)
