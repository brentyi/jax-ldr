import math
from typing import Literal

import datasets
import jax
import jax_dataclasses as jdc
import numpy as onp
import PIL.Image
from jax import numpy as jnp
from typing_extensions import Annotated


@jdc.pytree_dataclass
class MnistStruct(jdc.EnforcedAnnotationsMixin):
    image: Annotated[
        jnp.ndarray,
        (32, 32, 1),
        jnp.floating,
    ]
    label: Annotated[jnp.ndarray, (10,), jnp.integer]


def make_one_hot(labels: onp.ndarray, num_classes: int) -> onp.ndarray:
    """Convert integer labels to one-hot. Supports arbitrary batch axes."""
    batch_axes = labels.shape
    out_flat = onp.zeros((math.prod(batch_axes), num_classes), dtype=onp.uint8)
    out_flat[onp.arange(out_flat.shape[0]), labels.flatten()] = 1
    return out_flat.reshape(labels.shape + (num_classes,))


def load_mnist_dataset(split: Literal["train", "test"]) -> "MnistStruct":
    """Load entire MNIST dataset into an `MnistStruct` container."""
    d = datasets.load_dataset("mnist")
    d.set_format("numpy")  # type: ignore
    images = []

    im: PIL.Image
    for im in d[split]["image"]:  # type: ignore
        images.append(onp.array(im.resize((32, 32), PIL.Image.BILINEAR)))

    return MnistStruct(
        image=2.0 * (onp.array(images, dtype=onp.float32)[:, :, :, None] / 255.0 - 0.5),  # type: ignore
        label=make_one_hot(d[split]["label"], num_classes=10),  # type: ignore
    )


class MnistDataset:
    """MNIST as a torch-style dataset."""

    def __init__(self, split: Literal["train", "test"]):
        self.data = load_mnist_dataset(split)

    def __getitem__(self, index: int) -> MnistStruct:
        # import time
        # import random
        # time.sleep(random.random()/100.0)
        return jax.tree_map(lambda x: x[index], self.data)

    def __len__(self) -> int:
        (length,) = self.data.get_batch_axes()
        return length
        #  return 64 # TODO: remove
