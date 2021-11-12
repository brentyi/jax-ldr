import dataclasses
from typing import Any, Optional, Tuple

import fifteen
import flax
import jax
import jax_dataclasses
import numpy as onp
import optax
from jax import numpy as jnp
from typing_extensions import Literal

from . import ldr, mnist_data, mnist_networks

Pytree = Any


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    scheduler: Literal["linear", "none"] = "linear"
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999


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
                transition_steps=4500,
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

    @jax.jit
    def sequential_minimax_step(
        self,
        minibatch: mnist_data.MnistStruct,
    ) -> Tuple["TrainState", fifteen.experiments.TensorboardLogData]:
        """Run a min, then a max step. Backprops twice."""
        train_state = self
        train_state = _min_step(train_state, minibatch)
        train_state, log_data = _max_step(train_state, minibatch)
        with jax_dataclasses.copy_and_mutate(train_state) as train_state:
            train_state.steps = train_state.steps + 1
        return train_state, log_data

    @jax.jit
    def synchronous_minimax_step(
        self,
        minibatch: mnist_data.MnistStruct,
    ) -> Tuple["TrainState", fifteen.experiments.TensorboardLogData]:
        """Run min and max steps synchronously. Backprops only once.

        Note: if not for compatibility with the sequential minimax interface, this could
        be implemented much more simply. (for example: we could use a single optimizer
        rather than two separate ones)"""
        (batch_size,) = minibatch.get_batch_axes()

        # Compute gradient with respect to LDR score.
        (
            score,
            (
                f_state_updated_batch_stats,
                g_state_updated_batch_stats,
                compute_score_log_data,
            ),
        ), grads = jax.value_and_grad(
            lambda params: _compute_ldr_score(
                self, minibatch, f_params=params[0], g_params=params[1]
            ),
            has_aux=True,
        )(
            (self.f_state["params"], self.g_state["params"])
        )

        f_params_updates, f_optimizer_state_new = self.f_optimizer.tx.update(
            grads[0], self.f_optimizer.state, self.f_state["params"]
        )
        g_params_updates, g_optimizer_state_new = self.g_optimizer.tx.update(
            grads[1], self.g_optimizer.state, self.g_state["params"]
        )

        with jax_dataclasses.copy_and_mutate(self, validate=True) as out:
            out.f_optimizer.state = f_optimizer_state_new
            out.g_optimizer.state = g_optimizer_state_new
            out.f_state = flax.core.FrozenDict(
                # Important: we *subtract* the ADAM step. This turns our objective into
                # a maximization.
                params=jax.tree_map(
                    jnp.subtract, self.f_state["params"], f_params_updates
                ),
                **f_state_updated_batch_stats
            )
            out.g_state = flax.core.FrozenDict(
                # This map is the same as optax.apply_updates.
                params=jax.tree_map(jnp.add, self.g_state["params"], g_params_updates),
                **g_state_updated_batch_stats
            )
            out.steps = self.steps + 1

        return (
            out,
            fifteen.experiments.TensorboardLogData(
                scalars={
                    "gradient_norm": optax.global_norm(grads),
                    "score": score,
                }
            ).merge(compute_score_log_data.prefix("compute_score/")),
        )


def _compute_ldr_score(
    train_state: TrainState,
    minibatch: mnist_data.MnistStruct,
    f_params: Optional[flax.core.FrozenDict] = None,
    g_params: Optional[flax.core.FrozenDict] = None,
) -> Tuple[
    jnp.ndarray,
    Tuple[
        flax.core.FrozenDict,
        flax.core.FrozenDict,
        fifteen.experiments.TensorboardLogData,
    ],
]:
    """Score computation helper for train time.

    Args:
        minibatch (mnist_data.MnistStruct): Minibatch.
        f_params (Optional[flax.core.FrozenDict]): Encoder params override. Optional,
            but useful for gradients.
        g_params (Optional[flax.core.FrozenDict]): Decoder params override. Optional,
            but useful for gradients.

    Returns:
        Score,
    """
    if f_params is None:
        f_params = train_state.f_state["params"]
    if g_params is None:
        g_params = train_state.g_state["params"]

    # We'll be updating our states with batch statistics.
    # TODO: figure out a cleaner way to do this!
    f_state = train_state.f_state
    g_state = train_state.g_state

    X = minibatch.image
    Z, f_state = train_state.f_model.apply(
        {"params": f_params, "batch_stats": f_state["batch_stats"]},
        X,
        train=True,
        mutable=["batch_stats"],
    )
    X_hat, g_state = train_state.g_model.apply(
        {"params": g_params, "batch_stats": g_state["batch_stats"]},
        jax.lax.stop_gradient(Z),
        train=True,
        mutable=["batch_stats"],
    )
    Z_hat, f_state = train_state.f_model.apply(
        {"params": f_params, "batch_stats": f_state["batch_stats"]},
        X_hat,
        train=True,
        mutable=["batch_stats"],
    )
    assert Z.shape == Z_hat.shape == (minibatch.get_batch_axes()[0], 128)

    a, b, c = ldr.ldr_score_terms(
        Z=Z,
        Z_hat=Z_hat,
        one_hot_labels=minibatch.label,
        epsilon_sq=train_state.config.ldr_epsilon_sq,
    )
    score = a + b + c

    histogram_sample_size = 50
    X_minus_X_hat = X[histogram_sample_size:] - X_hat[histogram_sample_size:]
    return score, (
        f_state,
        g_state,
        fifteen.experiments.TensorboardLogData(
            scalars={
                "1_expansive_encode": a,
                "2_compressive_decode": b,
                "3_contrastive_contractive": c,
                "1_minus_2": a - b,
                "mse": jnp.mean(X_minus_X_hat ** 2),
            },
            histograms={
                "X": X[histogram_sample_size:],
                "X_hat": X_hat[histogram_sample_size:],
                "X_minus_X_hat": X_minus_X_hat,
            },
        ),
    )


def _min_step(
    train_state: TrainState,
    minibatch: mnist_data.MnistStruct,
) -> "TrainState":
    """Run one LDR score minimization step. Updates decoder parameters."""
    (batch_size,) = minibatch.get_batch_axes()

    (
        score,
        (
            f_state_updated_batch_stats,
            g_state_updated_batch_stats,
            _compute_score_log_data,
        ),
    ), grads = jax.value_and_grad(
        lambda g_params: _compute_ldr_score(train_state, minibatch, g_params=g_params),
        has_aux=True,
    )(
        train_state.g_state["params"]
    )
    g_params_updates, g_optimizer_state_new = train_state.g_optimizer.tx.update(
        grads, train_state.g_optimizer.state, train_state.g_state["params"]
    )
    with jax_dataclasses.copy_and_mutate(train_state, validate=True) as out:
        out.g_optimizer.state = g_optimizer_state_new
        out.f_state = train_state.f_state.copy(f_state_updated_batch_stats)
        out.g_state = flax.core.FrozenDict(
            params=optax.apply_updates(train_state.g_state["params"], g_params_updates),
            **g_state_updated_batch_stats
        )

    return out


def _max_step(
    train_state: TrainState,
    minibatch: mnist_data.MnistStruct,
) -> Tuple["TrainState", fifteen.experiments.TensorboardLogData]:
    """Run LDR score minimization step. Updates encoder parameters."""
    (batch_size,) = minibatch.get_batch_axes()

    (
        score,
        (
            f_state_updated_batch_stats,
            g_state_updated_batch_stats,
            compute_score_log_data,
        ),
    ), grads = jax.value_and_grad(
        lambda f_params: _compute_ldr_score(train_state, minibatch, f_params=f_params),
        has_aux=True,
    )(
        train_state.f_state["params"]
    )

    # Flip the gradients to maximize instead of minimize.
    grads = jax.tree_map(lambda x: -x, grads)

    f_params_updates, f_optimizer_state_new = train_state.f_optimizer.tx.update(
        grads, train_state.f_optimizer.state, train_state.f_state["params"]
    )
    with jax_dataclasses.copy_and_mutate(train_state, validate=True) as out:
        out.f_optimizer.state = f_optimizer_state_new
        out.f_state = flax.core.FrozenDict(
            params=optax.apply_updates(train_state.f_state["params"], f_params_updates),
            **f_state_updated_batch_stats
        )
        out.g_state = train_state.g_state.copy(g_state_updated_batch_stats)

    return (
        out,
        fifteen.experiments.TensorboardLogData(
            scalars={
                "gradient_norm": optax.global_norm(grads),
                "score": score,
            }
        ).merge(compute_score_log_data.prefix("compute_score/")),
    )
