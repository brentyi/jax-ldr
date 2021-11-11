import timeit

import jax
import numpy as onp
from jax import numpy as jnp

N = 2048
D = 128

X = onp.random.randn(N, D)


@jax.jit
def simple(X):
    return X.T @ X


@jax.jit
def einsum(X):
    return jnp.einsum("ni,nj->ij", X, X)


@jax.jit
def einsum_then_sum(X):
    return jnp.sum(jnp.einsum("ni,nj->nij", X, X), axis=0)


onp.testing.assert_allclose(simple(X), einsum(X), rtol=1e-5, atol=1e-5)
onp.testing.assert_allclose(simple(X), einsum_then_sum(X), rtol=1e-5, atol=1e-5)

print(timeit.timeit(lambda: simple(X), number=10000))
print(timeit.timeit(lambda: einsum(X), number=10000))
print(timeit.timeit(lambda: einsum_then_sum(X), number=10000))
