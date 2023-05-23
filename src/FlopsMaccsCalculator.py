import torch
import torchvision.models as models
from torchstat import stat
from torchscan import summary
import argparse
from collections import namedtuple
import math
import torch.nn as nn


def parsing():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Training Models')

    # Add arguments
    parser.add_argument('--model_type', type=str, default="vgg16", help='model type between vgg16, resnet, mobilenet')

  # Parse the arguments
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = parsing()

    model = args.model_type

    if model == "vgg16":
        # Load VGG16 model
        model = models.vgg16(pretrained=True)
        # Calculate FLOPS and MACCs per layer
        stat(model, (3, 224, 224))
    elif model == "resnet":
       # Load resnet model
        model = models.resnet18(pretrained=True)
        # Calculate FLOPS and MACCs per layer
        stat(model, (3, 224, 224))
    elif model == "mobilenet":
        # Load mobilenet model
        model = models.mobilenet_v2(pretrained=True)
        # Calculate FLOPS and MACCs per layer
        stat(model, (3, 224, 224))
    else:
        print("Wrong Model!!")













