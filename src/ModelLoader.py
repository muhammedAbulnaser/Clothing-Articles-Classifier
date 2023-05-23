import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import f1_score
import numpy as np

class ModelLoader:
    """
    Class for loading pre-trained models based on the specified model type.

    Args:
        model_type (str): Type of the model to load ("vgg16", "resnet", or "mobilenet").
        num_classes (int): Number of output classes for the model.

    """

    def __init__(self, model_type, num_classes):
        self.model_type = model_type
        self.num_classes = num_classes

    def load_model(self):
        """
        Load the specified pre-trained model based on the model type.

        Returns:
            torch.nn.Module: Loaded pre-trained model.

        Raises:
            ValueError: If an invalid model type is specified.

        """

        if self.model_type == "vgg16":
            model = models.vgg16(pretrained=True)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, self.num_classes)
        elif self.model_type == "resnet":
            model = models.resnet18(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        elif self.model_type == "mobilenet":
            model = models.mobilenet_v2(pretrained=True)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)
        else:
            raise ValueError("Invalid model type")

        return model
