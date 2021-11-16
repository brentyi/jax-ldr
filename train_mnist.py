import dataclasses
from typing import Optional

import dcargs
import fifteen
from tqdm.auto import tqdm

from ldr import mnist_data, mnist_training, mnist_viz


@dataclasses.dataclass
class Args:
    experiment_identifier: str = "mnist-ldr-" + fifteen.utils.timestamp()

    simultaneous_minimax: bool = False
    """By default, we do a sequential minimax: we perform minimization and maximization
    steps separately, doing forward and backward passes once for each. This is how the
    preprint by Dai et al ran experiments.

    Since the maximization objective used for the encoder is simply the negated
    minimization objective used for the decoder (in contrast to, say, a GAN with a
    nonsaturating generator loss), however, we can also play a simultaneous minimax game
    by doing forward and backward passes just ~once~ and then flipping the signs for
    half of the gradients. There are potentially some stability risks here (possibly why
    it wasn't done in the original paper), but empirically (for MNIST) these haven't
    been an issue and we get a massive speed up."""

    seed: int = 94709
    restore_existing_checkpoint: bool = True
    batch_size: int = 2048
    train_config: mnist_training.TrainConfig = mnist_training.TrainConfig()


def main(args: Args):

    experiment = fifteen.experiments.Experiment(identifier=args.experiment_identifier)

    test_data = mnist_data.load_mnist_dataset("test")
    train_dataset = mnist_data.MnistDataset("train")
    train_dataloader = fifteen.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4
    )
    train_state = mnist_training.TrainState.setup(
        config=args.train_config,
        seed=args.seed,
    )

    if args.restore_existing_checkpoint:
        try:
            train_state = experiment.restore_checkpoint(train_state)
        except FileNotFoundError:
            print("No checkpoint found!")
    else:
        experiment.clear()

    experiment.write_metadata("train_config", args.train_config)

    # Run sequential minimax game.
    while train_state.steps < args.train_config.training_steps:

        # Shuffle minibatches by epoch count.
        epoch = train_state.steps // train_dataloader.minibatch_count()
        for minibatch in tqdm(train_dataloader.minibatches(shuffle_seed=epoch)):
            # Run minimax step.
            if args.simultaneous_minimax:
                train_state, log_data = train_state.simultaneous_minimax_step(minibatch)
            else:
                train_state, log_data = train_state.sequential_minimax_step(minibatch)

            # Log to Tensorboard.
            experiment.log(
                log_data,
                step=train_state.steps,
                log_scalars_every_n=10,
                log_histograms_every_n=100,
            )

            # Checkpoint.
            if train_state.steps % 50 == 0:
                experiment.save_checkpoint(train_state, step=train_state.steps)
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


if __name__ == "__main__":
    fifteen.utils.pdb_safety_net()
    main(dcargs.parse(Args))
