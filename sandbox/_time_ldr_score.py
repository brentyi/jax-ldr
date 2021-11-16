import sys
from tqdm.auto import tqdm

sys.path.append("..")

import functools
import math
import timeit

import jax
import numpy as onp
from ldr import ldr
from matplotlib import pyplot as plt


def make_one_hot(labels: onp.ndarray, num_classes: int) -> onp.ndarray:
    """Convert integer labels to one-hot. Supports arbitrary batch axes."""
    batch_axes = labels.shape
    out_flat = onp.zeros((math.prod(batch_axes), num_classes), dtype=onp.uint8)
    out_flat[onp.arange(out_flat.shape[0]), labels.flatten()] = 1
    return out_flat.reshape(labels.shape + (num_classes,))


def f(theta, Z, one_hot_labels, vectorize):
    Z = Z * theta
    Z_hat = Z + theta
    a, b, c = ldr.ldr_score_terms(
        Z,
        Z_hat,
        one_hot_labels=one_hot_labels,
        epsilon_sq=0.5,
        vectorize_over_classes=vectorize,
    )
    return a + b + c


f_vmap = functools.partial(f, vectorize=True)
f_loop = functools.partial(f, vectorize=False)


funcs = {
    "vmap": f_vmap,
    "loop": f_loop,
}

K: int  # Number of classes. One coding rate computed for each class.
for name, f in funcs.items():
    Ks = range(5, 51, 5)
    runtimes = []
    for K in tqdm(Ks):
        N = 2048  # Batch size.
        D = 128  # Latent dimension.

        Z = onp.random.randn(N, D)
        one_hot_labels = make_one_hot(
            onp.random.randint(low=0, high=K, size=(N,)), num_classes=K
        )
        assert one_hot_labels.shape == (N, K)
        # print("FORWARD PASS")
        time = []
        theta = onp.random.randn(D)
        jit_f = jax.jit(jax.grad(f))
        jit_f(theta, Z, one_hot_labels)
        runtimes.append(timeit.timeit(lambda: jit_f(theta, Z, one_hot_labels), number=100))
    plt.plot(Ks, runtimes, label=name)

plt.xlabel("K (number of classes)")
plt.ylabel("seconds for 100 runs")
plt.legend()
plt.ylim(bottom=0)
plt.title("LDR score backward pass time")
plt.show()


# print()
# print("BACKWARD PASS")
# for name, f in funcs.items():
#     theta = onp.random.randn(128)
#     jit_f = jax.jit(jax.grad(f))  # type: ignore
#     jit_f(theta, Z, one_hot_labels)
#     print(name, timeit.timeit(lambda: jit_f(theta, Z, one_hot_labels), number=100))
