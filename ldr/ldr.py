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
    N_, D = Z.shape
    assert N == N_ and Z_hat.shape == Z.shape

    def _masked_ZTZ(Z: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        assert Z.shape == (N, D)
        assert mask.shape == (N,)
        Z_masked = jnp.where(mask[:, None], Z, 0.0)
        ZTZ = Z_masked.T @ Z_masked
        assert ZTZ.shape == (D, D)
        return ZTZ

    # Compute a `Z.T @ Z` matrix for each class.
    encoded_ZTZ_per_class = jax.vmap(lambda mask: _masked_ZTZ(Z, mask))(
        one_hot_labels.T  # (K, N)
    )
    transcribed_ZTZ_per_class = jax.vmap(lambda mask: _masked_ZTZ(Z_hat, mask))(
        one_hot_labels.T  # (K, N)
    )
    assert encoded_ZTZ_per_class.shape == transcribed_ZTZ_per_class.shape == (K, D, D)

    # Compute per-class counts.
    count_per_class = jnp.sum(one_hot_labels, axis=0)
    assert count_per_class.shape == (K,)

    # Compute per-class coding rates.
    encoded_coding_rate_per_class = jax.vmap(
        lambda ZTZ, count: mcr.coding_rate_from_ZTZ(ZTZ, count, epsilon_sq)
    )(encoded_ZTZ_per_class, count_per_class)
    transcribed_coding_rate_per_class = jax.vmap(
        lambda ZTZ, count: mcr.coding_rate_from_ZTZ(ZTZ, count, epsilon_sq)
    )(transcribed_ZTZ_per_class, count_per_class)
    assert (
        encoded_coding_rate_per_class.shape
        == transcribed_coding_rate_per_class.shape
        == (K,)
    )

    # Our score is made of 3 parts, consisting of coding rates that correspond to: an
    # expansive encoding score, compressive decoding score, and a sum over per-class
    # contrastive/contractive terms.
    expansive_encode = mcr.multiclass_coding_rate_delta(
        Z,
        encoded_coding_rate_per_class,
        one_hot_labels,
        epsilon_sq,
    )
    compressive_decode = mcr.multiclass_coding_rate_delta(
        Z_hat,
        transcribed_coding_rate_per_class,
        one_hot_labels,
        epsilon_sq,
    )
    contrastive_contractive_terms = jax.vmap(
        # Coding rate distances.
        lambda union_ZTZ, encoded_coding_rate, transcribed_coding_rate, count: mcr.coding_rate_from_ZTZ(
            union_ZTZ, count * 2, epsilon_sq
        )
        - 0.5 * (encoded_coding_rate + transcribed_coding_rate)
    )(
        encoded_ZTZ_per_class + transcribed_ZTZ_per_class,
        encoded_coding_rate_per_class,
        transcribed_coding_rate_per_class,
        count_per_class,
    )
    assert contrastive_contractive_terms.shape == (K,)

    return expansive_encode, compressive_decode, jnp.sum(contrastive_contractive_terms)
