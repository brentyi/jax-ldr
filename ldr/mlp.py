import dataclasses
import enum
from functools import partial

from flax import linen as nn
from jax import numpy as jnp

relu_layer_init = nn.initializers.kaiming_normal()  # variance = 2.0 / fan_in
linear_layer_init = nn.initializers.lecun_normal()  # variance = 1.0 / fan_in


class Activation(enum.Enum):
    # Enum attributes must be objects, not method definitions
    RELU = enum.auto()
    SIGMOID = enum.auto()


@dataclasses.dataclass
class MLPConfig:
    units: int = 32
    layers: int = 4
    use_bias: bool = True
    output_dim: int = 1
    activation: Activation = Activation.RELU


class MLP(nn.Module):
    config: MLPConfig

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs

        for i in range(self.config.layers):
            x = nn.Dense(
                self.config.units,
                kernel_init={
                    Activation.RELU: relu_layer_init,
                    Activation.SIGMOID: linear_layer_init,
                }[self.config.activation],
                use_bias=self.config.use_bias,
            )(x)
            x = {Activation.RELU: nn.relu, Activation.SIGMOID: nn.sigmoid}[
                self.config.activation
            ](x)

        x = nn.Dense(
            self.config.output_dim,
            kernel_init=linear_layer_init,
            use_bias=self.config.use_bias,
        )(x)
        return x
