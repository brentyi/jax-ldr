import math
import timeit
from typing import Callable, TypeVar

import jax
import numpy as onp
from jax import numpy as jnp

K = 100  # Number of classes. One coding rate computed for each class.
N = 2048  # Batch size.
D = 128  # Latent dimension.


def logdet_hermitian(A: jnp.ndarray) -> jnp.ndarray:
    """Compute the log-determinant of a Hermitian matrix."""
    M, N = A.shape
    assert M == N
    return 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(A))))


def coding_rate(A: jnp.ndarray, count: int, epsilon_sq: float = 0.5) -> jnp.ndarray:
    M, N = A.shape
    assert M == N
    I = jnp.eye(M)
    alpha = D / (count * epsilon_sq)
    return logdet_hermitian(I + alpha * A)


def make_one_hot(labels: onp.ndarray, num_classes: int) -> onp.ndarray:
    """Convert integer labels to one-hot. Supports arbitrary batch axes."""
    batch_axes = labels.shape
    out_flat = onp.zeros((math.prod(batch_axes), num_classes), dtype=onp.uint8)
    out_flat[onp.arange(out_flat.shape[0]), labels.flatten()] = 1
    return out_flat.reshape(labels.shape + (num_classes,))


CallableType = TypeVar("CallableType", bound=Callable)


def _looped_vmap(func: CallableType) -> CallableType:
    """Drop-in replacement for `jax.vmap`, which instead uses a for loop."""
    # print("Looping!")
    def looped_func(*args, **kwargs):
        batch_count = None
        for leaf in jax.tree_leaves((args, kwargs)):
            if batch_count is None:
                batch_count = leaf.shape[0]
            else:
                assert batch_count == leaf.shape[0]

        output = []
        for i in range(batch_count):
            a, kw = jax.tree_map(lambda x: x[i], (args, kwargs))
            output.append(func(*a, **kw))
        return jax.tree_map(lambda *x: jnp.stack(x, axis=0), *output)

    return looped_func  # type: ignore


def _apply_mask_where(X: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(mask[:, None], X, 0.0)


def _apply_mask_multiply(X: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return mask[:, None] * X


def make_simple_zeroed(apply_mask, vmap):
    def simple_zeroed(X, one_hot_labels):
        def _masked_ZTZ(Z: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
            assert Z.shape == (N, D)
            assert mask.shape == (N,)
            Z_masked = jnp.where(mask[:, None], Z, 0.0)
            ZTZ = Z_masked.T @ Z_masked
            assert ZTZ.shape == (D, D)
            return ZTZ

        ZZT_by_class = vmap(lambda mask: _masked_ZTZ(X, mask))(one_hot_labels.T)
        assert ZZT_by_class.shape == (K, D, D)
        count_by_class = jnp.sum(one_hot_labels, axis=0)
        coding_rates = vmap(coding_rate)(ZZT_by_class, count_by_class)
        assert coding_rates.shape == (K,)
        return jnp.sum(coding_rates)

    return simple_zeroed


def make_einsum_zeroed(apply_mask, vmap):
    def einsum_zeroed(X, one_hot_labels):
        def f(mask):
            X_masked = apply_mask(X, mask=mask)
            return jnp.einsum("ni,nj->ij", X_masked, X_masked)

        ZZT_by_class = vmap(f)(one_hot_labels.T)
        assert ZZT_by_class.shape == (K, D, D)
        count_by_class = jnp.sum(one_hot_labels, axis=0)
        coding_rates = vmap(coding_rate)(ZZT_by_class, count_by_class)
        assert coding_rates.shape == (K,)
        return jnp.sum(coding_rates)

    return einsum_zeroed


def make_einsum_then_sum_where(vmap):
    def einsum_then_sum_where(X, one_hot_labels):
        ZZ_T = jnp.einsum("ni,nj->nij", X, X)

        def f(mask):
            return jnp.mask(ZZ_T, axis=0, where=mask[:, None, None])

        ZZT_by_class = vmap(f)(one_hot_labels.T)
        assert ZZT_by_class.shape == (K, D, D)
        count_by_class = jnp.sum(one_hot_labels, axis=0)
        coding_rates = vmap(coding_rate)(ZZT_by_class, count_by_class)
        assert coding_rates.shape == (K,)
        return jnp.sum(coding_rates)

    return einsum_then_sum_where


def main():
    Z = onp.random.randn(N, D)
    Z = Z / onp.linalg.norm(Z, axis=1, keepdims=True)
    one_hot_labels = make_one_hot(
        onp.random.randint(low=0, high=K, size=(N,)), num_classes=K
    )
    assert one_hot_labels.shape == (N, K)

    coding_rate_funcs = {
        # Vectorized
        "simple_zeroed_where_mask": make_simple_zeroed(_apply_mask_where, jax.vmap),
        "einsum_zeroed_where_mask": make_einsum_zeroed(_apply_mask_where, jax.vmap),
        # "simple_zeroed_multiply_mask": make_simple_zeroed(
        #     _apply_mask_multiply, jax.vmap
        # ),
        # "einsum_zeroed_multiply_mask": make_einsum_zeroed(
        #     _apply_mask_multiply, jax.vmap
        # ),
        # "einsum_then_sum_where": make_einsum_then_sum_where(jax.vmap),
        # Looped
        "looped_simple_zeroed_where_mask": make_simple_zeroed(
            _apply_mask_where, _looped_vmap
        ),
        "looped_einsum_zeroed_where_mask": make_einsum_zeroed(
            _apply_mask_where, _looped_vmap
        ),
        # "looped_simple_zeroed_multiply_mask": make_simple_zeroed(
        #     _apply_mask_multiply, _looped_vmap
        # ),
        # "looped_einsum_zeroed_multiply_mask": make_einsum_zeroed(
        #     _apply_mask_multiply, _looped_vmap
        # ),
        # "looped_einsum_then_sum_where": make_einsum_then_sum_where(_looped_vmap),
    }

    print("FORWARD PASSES:")
    expected_out = None
    for name, f in coding_rate_funcs.items():
        jit_f = jax.jit(f)
        out_f = jit_f(Z, one_hot_labels)
        if expected_out is not None:
            onp.testing.assert_allclose(out_f, expected_out, rtol=1e-3, atol=1e-3)
        else:
            expected_out = out_f
        print(f"{name}\t", timeit.timeit(lambda: jit_f(Z, one_hot_labels), number=200))

    print()
    print("BACKWARD PASSES:")
    for name, f in coding_rate_funcs.items():
        jit_grad_f = jax.jit(jax.grad(f))
        jit_grad_f(Z, one_hot_labels)
        print(
            f"{name}\t",
            timeit.timeit(lambda: jit_grad_f(Z, one_hot_labels), number=200),
        )


if __name__ == "__main__":
    main()
