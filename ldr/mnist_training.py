import dataclasses
from typing import Any, Optional, Tuple, cast

import fifteen
import flax
import jax
import jax_dataclasses
import numpy as onp
import optax
from jax import numpy as jnp
from typing_extensions import Literal

from . import ldr, mnist_data, mnist_networks, protocols

Pytree = Any


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 1.5e-4
    scheduler: Literal["linear", "none"] = "linear"
    adam_beta1: float = 0.0
    adam_beta2: float = 0.9


@dataclasses.dataclass
class TrainConfig:
    ldr_epsilon_sq: float = 0.5
    optimizer: OptimizerConfig = OptimizerConfig()


@jax_dataclasses.pytree_dataclass
class Optimizer:
    tx: optax.GradientTransformation = jax_dataclasses.static_field()
    state: optax.OptState

    @staticmethod
    def setup(config: OptimizerConfig, params: Pytree) -> "Optimizer":
        tx = optax.adam(
            learning_rate=optax.linear_schedule(
                init_value=config.learning_rate,
                end_value=0.0,
                transition_steps=45000,
            )
            if config.scheduler == "linear"
            else config.learning_rate,
            b1=config.adam_beta1,
            b2=config.adam_beta2,
        )
        state = tx.init(params)
        return Optimizer(tx=tx, state=state)


@jax_dataclasses.pytree_dataclass
class TrainState:
    config: TrainConfig = jax_dataclasses.static_field()

    # f and theta in paper.
    f_model: mnist_networks.MnistEncoder = jax_dataclasses.static_field()
    f_state: flax.core.FrozenDict
    f_optimizer: Optimizer

    # g and eta in paper.
    g_model: mnist_networks.MnistDecoder = jax_dataclasses.static_field()
    g_state: flax.core.FrozenDict
    g_optimizer: Optimizer

    steps: int

    @staticmethod
    def setup(config: TrainConfig, seed: int) -> "TrainState":

        prng_key_0, prng_key_1 = jax.random.split(jax.random.PRNGKey(seed))

        nz: int = 128
        dummy_x = onp.zeros((1, 32, 32, 1), dtype=onp.float32)
        dummy_z = onp.zeros((1, 1, 1, nz), dtype=onp.float32)

        f_model = mnist_networks.MnistEncoder()
        f_state = f_model.init(prng_key_0, dummy_x, train=True)

        g_model = mnist_networks.MnistDecoder()
        g_state = g_model.init(prng_key_1, dummy_z, train=True)

        return TrainState(
            config=config,
            f_model=f_model,
            f_state=f_state,
            f_optimizer=Optimizer.setup(
                config=config.optimizer, params=f_state["params"]
            ),
            g_model=g_model,
            g_state=g_state,
            g_optimizer=Optimizer.setup(
                config=config.optimizer, params=g_state["params"]
            ),
            steps=0,
        )

    def _compute_score(
        self,
        minibatch: mnist_data.MnistStruct,
        f_params: Optional[flax.core.FrozenDict] = None,
        g_params: Optional[flax.core.FrozenDict] = None,
        negate_score: bool = False,
    ) -> Tuple[
        jnp.ndarray,
        Tuple[
            flax.core.FrozenDict,
            flax.core.FrozenDict,
            jnp.ndarray,
            jnp.ndarray,
        ],
    ]:
        """Score computation helper for train time.

        Args:
            minibatch (mnist_data.MnistStruct): Minibatch.
            f_params (Optional[flax.core.FrozenDict]): Encoder params override. Optional,
                but useful for gradients.
            g_params (Optional[flax.core.FrozenDict]): Decoder params override. Optional,
                but useful for gradients.
            negate_score (bool): Set to True to negate the score; useful for maximizing
                instead of minimizing. Optional.

        Returns:
            Score,
        """
        if f_params is None:
            f_params = self.f_state["params"]
        if g_params is None:
            g_params = self.g_state["params"]

        # We'll be updating our states with batch statistics.
        f_state = self.f_state
        g_state = self.g_state

        X = minibatch.image
        Z, f_state = self.f_model.apply(
            {"params": f_params, "batch_stats": f_state["batch_stats"]},
            X,
            train=True,
            mutable=["batch_stats"],
        )
        X_hat, g_state = self.g_model.apply(
            {"params": g_params, "batch_stats": g_state["batch_stats"]},
            Z,
            train=True,
            mutable=["batch_stats"],
        )
        Z_hat, f_state = self.f_model.apply(
            {"params": f_params, "batch_stats": f_state["batch_stats"]},
            X_hat,
            train=True,
            mutable=["batch_stats"],
        )
        assert Z.shape == Z_hat.shape == (minibatch.get_batch_axes()[0], 128)
        score = ldr.ldr_score(
            Z=Z,
            Z_hat=Z_hat,
            one_hot_labels=minibatch.label,
            epsilon_sq=self.config.ldr_epsilon_sq,
        )
        if negate_score:
            score = -score

        Z_dot_Z_hat = jnp.einsum("ni,ni->n", Z, Z_hat)
        return score, (f_state, g_state, X, X_hat, Z_dot_Z_hat)

    @jax.jit
    def min_step(
        self,
        minibatch: mnist_data.MnistStruct,
    ) -> Tuple["TrainState", fifteen.experiments.TensorboardLogData]:
        """Run one LDR score minimization step. Updates decoder parameters."""
        (batch_size,) = minibatch.get_batch_axes()

        (
            score,
            (
                f_state_updated_batch_stats,
                g_state_updated_batch_stats,
                _X,
                _X_hat,
                _Z_dot_Z_hat,
            ),
        ), grads = jax.value_and_grad(
            lambda g_params: self._compute_score(
                minibatch, g_params=g_params, negate_score=False
            ),
            has_aux=True,
        )(
            self.g_state["params"]
        )
        g_params_updates, g_optimizer_state_new = self.g_optimizer.tx.update(
            grads, self.g_optimizer.state, self.g_state["params"]
        )
        with jax_dataclasses.copy_and_mutate(self) as out:
            out.g_optimizer.state = g_optimizer_state_new
            out.f_state = flax.core.FrozenDict(
                params=self.f_state["params"],
                batch_stats=f_state_updated_batch_stats["batch_stats"],
            )
            out.g_state = flax.core.FrozenDict(
                params=optax.apply_updates(self.g_state["params"], g_params_updates),
                batch_stats=g_state_updated_batch_stats["batch_stats"],
            )
            out.steps = self.steps + 1
        return out, fifteen.experiments.TensorboardLogData(
            scalars={
                "min/global_norm": optax.global_norm(grads),
                "min/score": score,
            }
        )

    @jax.jit
    def max_step(
        self,
        minibatch: mnist_data.MnistStruct,
    ) -> Tuple["TrainState", fifteen.experiments.TensorboardLogData]:
        """Run LDR score minimization step. Updates encoder parameters."""
        (batch_size,) = minibatch.get_batch_axes()

        (
            score,
            (
                f_state_updated_batch_stats,
                g_state_updated_batch_stats,
                X,
                X_hat,
                Z_dot_Z_hat,
            ),
        ), grads = jax.value_and_grad(
            lambda f_params: self._compute_score(
                minibatch, f_params=f_params, negate_score=True
            ),
            has_aux=True,
        )(
            self.f_state["params"]
        )
        f_params_updates, f_optimizer_state_new = self.f_optimizer.tx.update(
            grads, self.f_optimizer.state, self.f_state["params"]
        )
        with jax_dataclasses.copy_and_mutate(self) as out:
            out.f_optimizer.state = f_optimizer_state_new
            out.f_state = flax.core.FrozenDict(
                params=optax.apply_updates(self.f_state["params"], f_params_updates),
                batch_stats=f_state_updated_batch_stats["batch_stats"],
            )
            out.g_state = flax.core.FrozenDict(
                params=self.g_state["params"],
                batch_stats=g_state_updated_batch_stats["batch_stats"],
            )
            out.steps = self.steps + 1

        # For histograms, we don't need to return every single X/X_hat
        hist_count = min(batch_size, 64)
        X_minus_X_hat = X - X_hat
        return out, fifteen.experiments.TensorboardLogData(
            scalars={
                "max/global_norm": optax.global_norm(grads),
                "max/score": score,
                "max/mse": jnp.linalg.norm(X_minus_X_hat),
            },
            histograms={
                "max/X": X[:hist_count],
                "max/X_hat": X_hat[:hist_count],
                "max/X_minus_X_hat": X_minus_X_hat[:hist_count],
                "max/Z_dot_Z_hat": Z_dot_Z_hat,
            },
        )
