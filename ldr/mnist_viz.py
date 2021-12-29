import io
from typing import cast

import jax
import numpy as onp
import PIL
from jax import numpy as jnp
from matplotlib import pyplot as plt

from . import mnist_data, mnist_training


@jax.jit
def _encode_then_decode(
    train_state: mnist_training.TrainState,
    X: jnp.ndarray,  # These should be images.
) -> jnp.ndarray:
    batch_size = X.shape[0]
    assert X.shape == (batch_size, 32, 32, 1)

    Z = cast(
        jnp.ndarray,
        train_state.f_model.apply(train_state.f_state, X, train=False),
    )
    X_hat = cast(
        jnp.ndarray,
        train_state.g_model.apply(train_state.g_state, Z, train=False),
    )

    assert X_hat.shape == X.shape
    return X_hat


def visualize_encode_decode(
    train_state: mnist_training.TrainState, minibatch: mnist_data.MnistStruct
) -> onp.ndarray:
    """Generate a plot to visualize some MNIST encoding/decoding pairs. Returned as a
    numpy array."""
    (batch_size,) = minibatch.get_batch_axes()
    assert batch_size >= 8, "Visualization assume >=8 images."
    X = minibatch.image[:8]
    X_hat = _encode_then_decode(train_state, X)

    fig, axs = plt.subplots(4, 4, figsize=(3, 3))
    axs = axs.flatten()
    for i in range(16 // 2):
        axs[i * 2].imshow(X[i])
        axs[i * 2 + 1].imshow(X_hat[i])
        if i < 2:
            axs[i * 2].set_title("X")
            axs[i * 2 + 1].set_title("g(f(X))")

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    return onp.array(PIL.Image.open(buf))
