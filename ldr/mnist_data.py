import math
from typing import Iterable, Literal, TypeVar, cast

import datasets
import fifteen
import jax
import jax_dataclasses
import numpy as onp
import PIL.Image
from jax import numpy as jnp
from typing_extensions import Annotated


@jax_dataclasses.pytree_dataclass
class MnistStruct(jax_dataclasses.EnforcedAnnotationsMixin):
    image: Annotated[
        jnp.ndarray,
        (32, 32, 1),
        jnp.floating,
    ]
    label: Annotated[jnp.ndarray, (10,), jnp.integer]

    def normalize(self) -> "MnistStruct":
        with jax_dataclasses.copy_and_mutate(self) as out:
            out.image = (self.image - onp.mean(self.image)) / onp.std(
                self.image
            ) * 0.5 + 0.5
        return out


def make_one_hot(labels: onp.ndarray, num_classes: int) -> onp.ndarray:
    """Convert integer labels to one-hot. Supports arbitrary batch axes."""
    batch_axes = labels.shape
    out_flat = onp.zeros((math.prod(batch_axes), num_classes), dtype=onp.uint8)
    out_flat[onp.arange(out_flat.shape[0]), labels.flatten()] = 1
    return out_flat.reshape(labels.shape + (num_classes,))


def load_mnist_dataset(split: Literal["train", "test"]) -> "MnistStruct":
    """Load entire MNIST dataset into an `MnistStruct` container."""
    d = datasets.load_dataset("mnist")
    d.set_format("numpy")

    images = []
    for im in d[split]["image"]:
        images.append(
            onp.array(PIL.Image.fromarray(im).resize((32, 32), PIL.Image.BILINEAR))[
                :, :, None
            ]
        )

    return MnistStruct(
        image=onp.array(images, dtype=onp.float32),
        label=make_one_hot(d[split]["label"], num_classes=10),
    ).normalize()


class MnistDataset:
    """MNIST as a torch-style dataset."""

    def __init__(self, split: Literal["train", "test"]):
        self.dataset = load_mnist_dataset(split)

    def __getitem__(self, index: int) -> MnistStruct:
        # import time
        # import random
        # time.sleep(random.random()/100.0)
        return jax.tree_map(lambda x: x[index], self.dataset)

    def __len__(self) -> int:
        (length,) = self.dataset.get_batch_axes()
        return length
