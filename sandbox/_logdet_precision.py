from typing import Dict, List

import jax
import numpy as onp
from jax import numpy as jnp
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

jax_logdets = {
    "slogdet": lambda A: jnp.linalg.slogdet(A)[1],
    "cholesky": lambda A: 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(A)))),
    "eigh": lambda A: jnp.sum(jnp.log(jnp.linalg.eigvalsh(A))),
}
jax_logdets = {k: jax.jit(v) for k, v in jax_logdets.items()}


stds = 10.0 ** (-onp.arange(0.0, 10.0, 0.1))
delta_means: Dict[str, List[float]] = {}
delta_std_errors: Dict[str, List[float]] = {}
for method, logdet in jax_logdets.items():
    delta_means[method] = []
    delta_std_errors[method] = []
    for std in stds:
        delta_list = []
        for _ in tqdm(range(5000)):
            X = onp.random.normal(scale=std, size=(3, 3))
            A = X.T @ X
            _, logdet_64 = onp.linalg.slogdet(A)
            # logdet_64 = 2 * onp.sum(onp.log(onp.diag(onp.linalg.cholesky(A))))
            # logdet_64 = logdet_64 ** 2
            assert logdet_64.dtype == onp.float64
            logdet_32 = onp.array(logdet(A), dtype=onp.float64)
            delta_list.append(onp.abs(logdet_64 - logdet_32))

        delta_array = onp.array(delta_list)
        delta_array = delta_array[onp.logical_not(onp.isnan(delta_list))]
        delta_array = delta_array[onp.logical_not(onp.isinf(delta_list))]
        delta_array = delta_array[onp.logical_not(onp.isneginf(delta_list))]
        delta_means[method].append(onp.mean(delta_array))
        delta_std_errors[method].append(
            onp.std(delta_list) / onp.sqrt(len(delta_array))  # Standard error
        )


for method, delta_list in delta_means.items():
    assert len(stds) == len(delta_list)
    delta_array = onp.array(delta_list)
    delta_std_error_array = onp.array(delta_std_errors[method])

    plt.plot(stds, delta_list, label=method)
    plt.fill_between(
        stds,
        y1=delta_array - delta_std_error_array,
        y2=delta_array + delta_std_error_array,
        alpha=0.1,
    )
    print(delta_list)

plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.show()
