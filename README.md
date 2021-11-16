# jax-ldr

*Unofficial* implementation of [Closed-Loop Data Transcription to an LDR via
Minimaxing Rate Reduction](https://arxiv.org/abs/2111.06636), using JAX/flax.

Learns a generative + discriminative representation of visual data with some
nice underlying structures (classes are mapped to mutually incoherent linear
subspaces) via rate reduction.


## Usage

Install:
```
# This will install the CPU version of JAX if there's no JAX package installed.
pip install -r requirements.txt
```

Train:
```
$ python train_mnist.py --help

usage: train_mnist.py [-h] [--experiment-identifier STR] [--simultaneous-minimax] [--seed INT] [--no-restore-existing-checkpoint]
                      [--batch-size INT] [--train-config.training-steps INT] [--train-config.optimizer.learning-rate FLOAT]
                      [--train-config.optimizer.scheduler {none,linear}] [--train-config.optimizer.adam-beta1 FLOAT]
                      [--train-config.optimizer.adam-beta2 FLOAT] [--train-config.ldr-epsilon-sq FLOAT]
                      [--train-config.no-vectorize-over-classes]

optional arguments:
  -h, --help            show this help message and exit
  --experiment-identifier STR
                        (default: mnist-ldr-{timestamp})
  --simultaneous-minimax
                        By default, we do a sequential minimax: we perform minimization and maximization
                        steps separately, doing forward and backward passes once for each. This is how the
                        preprint by Dai et al ran experiments.

                        Since the maximization objective used for the encoder is simply the negated
                        minimization objective used for the decoder (in contrast to, say, a GAN with a
                        nonsaturating generator loss), however, we can also play a simultaneous minimax game
                        by doing forward and backward passes just ~once~ and then flipping the signs for
                        half of the gradients. There are potentially some stability risks here (possibly why
                        it wasn't done in the original paper), but empirically (for MNIST) these haven't
                        been an issue and we get a massive speed up.
  --seed INT            (default: 94709)
  --no-restore-existing-checkpoint
  --batch-size INT      (default: 2048)
  --train-config.training-steps INT
                        (default: 4500)
  --train-config.optimizer.learning-rate FLOAT
                        (default: 0.0001)
  --train-config.optimizer.scheduler {none,linear}
                        (default: linear)
  --train-config.optimizer.adam-beta1 FLOAT
                        (default: 0.5)
  --train-config.optimizer.adam-beta2 FLOAT
                        (default: 0.999)
  --train-config.ldr-epsilon-sq FLOAT
                        $\epsilon^2$ parameter used for MCR losses. (default: 0.5)
  --train-config.no-vectorize-over-classes
                        Set to False to loop over classes instead of vectorizing. This typically leads to
                        a slight runtime hit, but has empirically also sped up sequential (not synchronous!)
                        minimax on some older GPUs. Possibly related to memory layout.
```

## Visualization

Generative quality can be visualized directly in the "Images" tab in Tensorboard:
```
tensorboard --logdir experiments/
```
