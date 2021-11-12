import dataclasses
from typing import Optional

import dcargs
import fifteen
from tqdm.auto import tqdm

from ldr import mnist_data, mnist_training, mnist_viz


@dataclasses.dataclass
class Args:
    experiment_identifier: Optional[str]
    restore_existing_checkpoint: bool = True
    batch_size: int = 2048
    training_steps: int = 4500
    train_config: mnist_training.TrainConfig = mnist_training.TrainConfig()
    synchronous_minimax: bool = False
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
    epoch = 0
    while train_state.steps < args.training_steps:

        for minibatch in tqdm(train_dataloader.minibatches(shuffle_seed=epoch)):

            # Run combined minimax step.
            if args.synchronous_minimax:
                train_state, log_data = train_state.synchronous_minimax_step(minibatch)
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

        epoch = epoch + 1


if __name__ == "__main__":
    fifteen.utils.pdb_safety_net()
    main(dcargs.parse(Args))
