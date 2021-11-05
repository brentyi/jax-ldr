"""DCGAN-style networks.

Loosely adapted from: https://github.com/bilal2vec/jax-dcgan/blob/main/dcgan.ipynb"""

from typing import TypedDict

import flax
from flax import linen as nn
from jax import numpy as jnp

# From the first DCGAN paper.
conv_kernel_init = nn.initializers.normal(stddev=0.02)
batchnorm_scale_init = lambda *a, **kw: 1.0 + conv_kernel_init(*a, **kw)  # type: ignore


class MnistModelState(TypedDict):
    """Type hint for encoder and decoder states."""

    params: flax.core.FrozenDict
    batch_stats: flax.core.FrozenDict


class MnistEncoder(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:  # type: ignore
        batch_size = x.shape[0]
        ndf = 64
        nz = 128
        x = nn.Conv(
            features=ndf,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            kernel_init=conv_kernel_init,
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(
            features=ndf * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            kernel_init=conv_kernel_init,
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            scale_init=batchnorm_scale_init,
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(
            features=ndf * 4,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            kernel_init=conv_kernel_init,
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            scale_init=batchnorm_scale_init,
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(
            features=nz,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding="VALID",
            use_bias=False,
            kernel_init=conv_kernel_init,
        )(x)
        x = x.reshape((x.shape[0], -1))
        assert x.shape == (batch_size, nz)

        # Normalize
        z = x / jnp.linalg.norm(x, axis=1, keepdims=True)

        return z


class MnistDecoder(nn.Module):
    @nn.compact
    def __call__(self, z: jnp.ndarray, train: bool) -> jnp.ndarray:  # type: ignore
        batch_size = z.shape[0]
        ndf = 64
        nz = 128
        x = z.reshape((batch_size, 1, 1, nz))
        x = nn.ConvTranspose(
            features=ndf * 8,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding="VALID",
            use_bias=False,
            kernel_init=conv_kernel_init,
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            scale_init=batchnorm_scale_init,
        )(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(
            features=ndf * 4,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            kernel_init=conv_kernel_init,
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            scale_init=batchnorm_scale_init,
        )(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(
            features=ndf * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            kernel_init=conv_kernel_init,
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            scale_init=batchnorm_scale_init,
        )(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(
            features=1,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            kernel_init=conv_kernel_init,
        )(x)
        assert x.shape == (batch_size, 32, 32, 1)

        return jnp.tanh(x)
