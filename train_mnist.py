import dataclasses
import time
from typing import Literal, cast

import datasets
import dcargs
import fifteen
import jax
import jax_dataclasses
import numpy as onp
import torch.utils.data
from jax import numpy as jnp
from tqdm.auto import tqdm
from typing_extensions import Annotated

from ldr import mnist_data, mnist_training


@dataclasses.dataclass
class Args:
    batch_size: int = 32  # 2048
    num_epochs: int = 100
    train_config: mnist_training.TrainConfig = mnist_training.TrainConfig()
    seed: int = 94709
    n_dis: int = 1  # Number of discriminator updates per generator update.


def main(args: Args):

    experiment = fifteen.experiments.Experiment(
        identifier="mnist-ldr-" + fifteen.utils.timestamp()
    )
    train_dataloader = fifteen.data.DataLoader(
        mnist_data.MnistDataset("test"),
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=fifteen.data.DataLoader.collate_fn,
    )
    train_state = mnist_training.TrainState.setup(
        config=args.train_config,
        seed=args.seed,
    )

    # Run sequential minimax game.
    for epoch in range(args.num_epochs):
        for minibatch in tqdm(train_dataloader.minibatches(shuffle_seed=epoch)):
            print("Running max step")

            # Max step. (~updating disciminator in a GAN)
            train_state, log_data = train_state.max_step(minibatch)
            experiment.log(
                log_data,
                step=train_state.steps,
                log_scalars_every_n=10,
                log_histograms_every_n=100,
            )

            # Min step. (~updating generator in a GAN)
            if (train_state.steps - args.n_dis + 1) % args.n_dis == 0:
                print("Running min step")
                train_state, log_data = train_state.min_step(minibatch)
                experiment.log(
                    log_data,
                    step=train_state.steps,
                    log_scalars_every_n=10,
                    log_histograms_every_n=100,
                )

            # Checkpoint.
            if train_state.steps % 100 == 0:
                experiment.summary_writer.image(
                    "eval/image",
                )
                experiment.save_checkpoint(train_state, step=train_state.steps)


if __name__ == "__main__":
    fifteen.utils.pdb_safety_net()
    main(dcargs.parse(Args))
