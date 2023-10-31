


# LightNorm

This repository contains source code execution of the LightNorm's simulation.


# Features
This repository features...

- Training various neural networks with [block floating point](https://en.wikipedia.org/wiki/Block_floating_point)
- Fully configurable training environment
- Save checkpoints, logs, etc

# Installation

## Setup with docker (Recommended)

1. Install [Docker](https://docs.docker.com/engine/install/) on the targeted machine.
2. Make a docker container as: `docker build . -t $(whoami)/lightnorm:latest`

## Setup without docker
1. Clone this repository
2. Install requirements listed below 
- `torch >= 1.7.1`
- `torchvision >= 0.5.0`
- `numba >= 0.50.1`
- `matplotlib >= 3.4.2`
- `einops >= 0.3.0`

# Execution examples

## Resnet with preset config

For the simple execution of the FP10_154 with the LightNorm data structure, execute

```docker run --rm --gpus '"device=0"' --cpus="4" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app lightnorm:latest python3 -u /app/main.py --mode train --model ResNet18 -bf ResNet18_FB12```
or
```./docker_run $(whoami)/lightnorm:latest 0 main.py --mode train --model ResNet18 --dtype fp10_154 --dataset CIFAR100 --save-name save_test --bfp default_FP10_154_bfloat_dim32```

## More information

More specifically, look at the [docs](/docs/_index.md) for the arguments and setup your custom network, etc...

# Citation

# License

This repository uses [CC BY 4.0](https://creativecommons.org/licenses/)