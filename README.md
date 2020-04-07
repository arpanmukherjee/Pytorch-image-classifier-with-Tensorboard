# Pytorch Image Classification with Tensorboard

## Quick Links
- [About](#about)
- [Setup](#setup)
	- [Installation](#installation)
	- [Training](#training)
- [Results](#results)

## About
This repo contains tutorials covering how to do sentiment analysis using [PyTorch](https://github.com/pytorch/pytorch), [TorchVision](https://github.com/pytorch/vision) and [TensorBoard](https://github.com/tensorflow/tensorboard) using Python 3.7.

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/arpanmukherjee/Pytorch-image-classifier-with-Tensorboard/issues/new). I welcome any feedback, positive or negative!**

## Setup
### Installation
1. Download the GitHub repo by using the following command running from the terminal.
```bash
git clone https://github.com/arpanmukherjee/Pytorch-image-classifier-with-Tensorboard
cd Autoencoders-and-more-using-PyTorch/
```

2. Install `pip` from the terminal, for more details please look [here](https://pypi.org/project/pip/). Go to the following project folder and install all the dependencies by running the following command. By running this command, it will install all the dependencies you will require to run the project.
```bash
pip install -r requirements.txt
```

### Training
The network can be trained using `main.py` script. Currently, it only accepts the following arguments with the allowed values. Please strictly follow the argument name and any of the values.

| argument | accepted values | default value |
|--|--|--|
| epochs | integer | 75 |
| batch-size | integer | 16 |
| learning-rate | float | 0.001 |
| seed | int | 1 |
| data-path | data directory | ../dataset/ |
| dataset | MNIST or STL10 or CIFAR10 | - |
| use_cuda | bool | False |
| weight-decay | float | 1e-5 |
| log-interval | int | 50 |
| save-model | bool | True |

Arguments that have no default value, you must provide value to run the script.
```bash
python main.py --dataset STL10 --use-cuda True
```
If you think the model is taking too much time, you can consider using GPU. Set `use_cuda` argument as `True`.
