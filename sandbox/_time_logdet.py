import timeit

import jax
import numpy as onp
from jax import numpy as jnp
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


def logdet_hermitian(A: jnp.ndarray) -> float:
    return 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(A))))


jax_logdets = {
    "standard logdet": lambda A: jnp.linalg.slogdet(A)[1],
    "cholesky logdet": lambda A: 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(A)))),
}
# jax_logdets = {k: jax.jit(v) for k, v in jax_logdets.items()}

number: int = 1000

fig, axes = plt.subplots(1, 2)

for name, f_raw in jax_logdets.items():
    for f, ax in zip((jax.jit(f_raw), jax.jit(jax.grad(f_raw))), axes):
        dims = range(1, 200, 10)
        times = []
        for n in tqdm(dims):
            A = onp.random.randn(n, n)
            A = A @ A.T
            f(A)  # type: ignore
            times.append(timeit.timeit(lambda: f(A), number=number))  # type: ignore
        ax.plot(dims, times, label=name)

ax0, ax1 = axes
ax0.set_ylim(bottom=0)
ax0.legend()
ax0.set_title("Forward pass logdet runtimes")
ax0.set_xlabel("N (matrix dim)")
ax0.set_ylabel(f"seconds for {number} calls")

ax1.set_ylim(bottom=0)
ax1.legend()
ax1.set_title("Backward pass logdet runtimes")
ax1.set_xlabel("N (matrix dim)")
ax1.set_ylabel(f"seconds for {number} calls")

plt.show()
