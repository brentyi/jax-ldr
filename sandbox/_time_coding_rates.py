import timeit

import jax
import numpy as onp
from jax import numpy as jnp


def logdet_hermitian(A: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-determinant of a Hermitian matrix."""
    M, N = A.shape
    assert M == N
    return 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(A))))


def coding_rate(A: jnp.ndarray, epsilon_sq: float = 0.5) -> jnp.ndarray:
    M, N = A.shape
    assert M == N
    I = jnp.eye(M)
    alpha = D / (N * epsilon_sq)
    return logdet_hermitian(I + alpha * A)


N = 2048
D = 128
X = onp.random.randn(N, D)

coding_rate_funcs = {
    "simple": lambda X: coding_rate(X.T @ X),
    "einsum": lambda X: coding_rate(jnp.einsum("ni,nj->ij", X, X)),
    "einsum_then_sum": lambda X: coding_rate(
        jnp.sum(jnp.einsum("ni,nj->nij", X, X), axis=0)
    ),
}

print("FORWARD PASSES:")
for name, f in coding_rate_funcs.items():
    jit_f = jax.jit(f)
    jit_f(X)
    print(f"{name}\t", timeit.timeit(lambda: jit_f(X), number=1000))

print("BACKWARD PASSES:")
for name, f in coding_rate_funcs.items():
    jit_grad_f = jax.jit(jax.grad(f))
    jit_grad_f(X)
    print(f"{name}\t", timeit.timeit(lambda: jit_grad_f(X), number=1000))
