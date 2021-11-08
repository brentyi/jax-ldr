import dataclasses
import time
from typing import Optional

import datasets
import dcargs
import fifteen
import jax
import jax_dataclasses
import numpy as onp
from jax import numpy as jnp
from tqdm.auto import tqdm
from typing_extensions import Annotated

from ldr import mnist_data, mnist_training, mnist_viz


@dataclasses.dataclass
class Args:
    experiment_identifier: Optional[str]
    batch_size: int = 2048
    num_epochs: int = 500000
    train_config: mnist_training.TrainConfig = mnist_training.TrainConfig()
    seed: int = 94709
    n_dis: int = 1  # Number of discriminator updates per generator update.


def main(args: Args):

    if args.experiment_identifier is None:
        experiment = fifteen.experiments.Experiment(
            identifier="mnist-ldr-" + fifteen.utils.timestamp()
        )
    else:
        experiment = fifteen.experiments.Experiment(
            identifier=args.experiment_identifier
        )
    experiment.write_metadata("train_config", args.train_config)

    test_data = mnist_data.load_mnist_dataset("test")

    train_dataset = mnist_data.MnistDataset("train")
    train_dataloader = fifteen.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=fifteen.data.DataLoader.collate_fn,
    )
    train_state = mnist_training.TrainState.setup(
        config=args.train_config,
        seed=args.seed,
    )

    try:
        train_state = experiment.restore_checkpoint(train_state)
    except FileNotFoundError:
        print("No checkpoint found!")

    # Run sequential minimax game.
    for epoch in range(args.num_epochs):

        if epoch % 100 == 0 and epoch > 0:
            experiment.save_checkpoint(
                train_state, step=train_state.steps, prefix=f"epoch_{epoch}_"
            )

        for minibatch in tqdm(train_dataloader.minibatches(shuffle_seed=epoch)):

            # Same some visualizations.
            if train_state.steps % 200 == 0:
                experiment.summary_writer.image(
                    "encode_decode_test_set",
                    mnist_viz.visualize_encode_decode(
                        train_state,
                        test_data,
                    ),
                    step=train_state.steps,
                )
                experiment.summary_writer.image(
                    "encode_decode_train_set",
                    mnist_viz.visualize_encode_decode(
                        train_state,
                        train_dataset.data,
                    ),
                    step=train_state.steps,
                )

            # Min step. (~updating disciminator in a GAN)
            train_state, log_data = train_state.min_step(minibatch)

            # Max step. (~updating generator in a GAN)
            if (train_state.steps - args.n_dis + 1) % args.n_dis == 0:
                train_state, min_log_data = train_state.max_step(minibatch)
                log_data = log_data.extend(min_log_data)

            # Log to Tensorboard.
            experiment.log(
                log_data,
                step=train_state.steps,
                log_scalars_every_n=10,
                log_histograms_every_n=100,
            )

            # Checkpoint.
            if train_state.steps % 200 == 0:
                experiment.save_checkpoint(train_state, step=train_state.steps)


if __name__ == "__main__":
    fifteen.utils.pdb_safety_net()
    main(dcargs.parse(Args))
