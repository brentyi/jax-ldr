from typing import Any, Protocol

from jax import numpy as jnp

Pytree = Any


class Encoder(Protocol):
    """Encoders take in inputs X and parameters theta, then output a latent vector Z."""

    def __call__(self, x: jnp.ndarray, theta: Pytree) -> jnp.ndarray:
        ...


class Decoder(Protocol):
    """Decoders take in latent features Z and parameters eta, then output a
    reconstructed X."""

    def __call__(self, z: jnp.ndarray, eta: Pytree) -> jnp.ndarray:
        ...
