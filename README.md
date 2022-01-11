# BFPSim
This repository can simulates block-floating point arithmetic in software manner within reasonable time cost

# Features
This repository features...

- Training various neural networks with [block floating point](https://en.wikipedia.org/wiki/Block_floating_point)
- Fully configurable training environment
- Save checkpoints, logs, etc

# Installation

## Setup with docker (Recommended)

1. Install [Docker](https://docs.docker.com/engine/install/) on the targeted machine.
2. Clone this repository
3. Make docker image as `docker build . -t $(whoami)/bfpsim:latest` (It will take a while, so get some coffee break)

## Setup tensorboard

After creating docker container, you should execute tensorboard on background.

To execute tensorboard, move to `tensorboard` directory, and execute `./run.sh [External Port]`. Make sure you are in tensorboard directory. Recommended value for external port is `6006`, but you can change if you know what you are doing. It will take a while if you first executing tensorboard, since docker image is different from main image

If your running this on remote server, make sure you opened the external port using `ufw`, `iptables`, etc, then input `[Remote Server IP]:[External Port]` on your beloved internet browser.

If you are running this on local machine, just type `http://localhost:[External Port]`, and you're good to go. :smiley:


## Setup without docker
1. Clone this repository
2. Install requirements listed below 
- `torch >= 1.9.1`
- `torchvision >= 0.5.0`
- `numba >= 0.53.1`
- `matplotlib >= 3.4.2`
- `einops >= 0.3.0`
- `slack_sdk`
- `tensorboard >= 2.7.0` (tensorboard can be any version, though)

## Additional Setup - Slackbot
It's possible to set train information to your own slack server.

If you want to set by your own, follow [Tutorial](https://github.com/slackapi/python-slack-sdk/blob/main/tutorial/01-creating-the-slack-app.md) and get the slackbot token, and put the token in separate file named `./slackbot.token`, and give `--slackbot` option as true.

# Execution examples

## Resnet with preset config

Execting FP32 Normal ResNet

```docker run --rm --gpus '"device=0"' --cpus="8" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/main.py --model --arch resnet18``

ResNet18 with FB24 

```docker run --rm --gpus '"device=0"' --cpus="8" --user "$(id -u):$(id -g)" --workdir /app -v "$(pwd)":/app $(whoami)/bfpsim:latest python3 -u /app/main.py --mode train --model ResNet18 --dataset CIFAR100 --log True --bfp ResNet18_FB24```
`

## More information

More specifically, look at the [docs](/docs/_index.md) for the arguments and setup your custom network, etc... (TBD)

# Citation

# License

This repository uses [CC BY 4.0](https://creativecommons.org/licenses/)