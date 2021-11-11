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


def _apply_mask(X: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(mask[:, None], X, 0.0)


class_count = 20
masks = onp.random.randint(low=0, high=2, size=(class_count, N)) == 1


def simple_zeroed(X):
    total = 0
    for i in range(class_count):
        X_masked = _apply_mask(X, mask=masks[i])
        total = total + coding_rate(X_masked.T @ X_masked)
    return total


def einsum_zeroed(X):
    total = 0
    for i in range(class_count):
        X_masked = _apply_mask(X, mask=masks[i])
        total = total + coding_rate(jnp.einsum("ni,nj->ij", X_masked, X_masked))
    return total


def einsum_then_sum_where(X):
    XX_T = jnp.einsum("ni,nj->nij", X, X)
    total = 0
    for i in range(class_count):
        total = total + coding_rate(
            jnp.sum(XX_T, axis=0, where=masks[i, :, None, None])
        )
    return total


coding_rate_funcs = {
    "simple_zeroed": simple_zeroed,
    "einsum_zeroed": einsum_zeroed,
    "einsum_then_sum_where": einsum_then_sum_where,
}

print("FORWARD PASSES:")
for name, f in coding_rate_funcs.items():
    jit_f = jax.jit(f)
    jit_f(X)
    print(f"{name}\t", timeit.timeit(lambda: jit_f(X), number=500))

print()
print("BACKWARD PASSES:")
for name, f in coding_rate_funcs.items():
    jit_grad_f = jax.jit(jax.grad(f))
    jit_grad_f(X)
    print(f"{name}\t", timeit.timeit(lambda: jit_grad_f(X), number=500))
