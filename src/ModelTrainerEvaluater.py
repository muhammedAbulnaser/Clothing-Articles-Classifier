import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import f1_score
import numpy as np

class ModelTrainerEvaluater:
    """
    Class for training and evaluating a model.

    Args:
        model (torch.nn.Module): The model to train and evaluate.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        criterion: The loss criterion.
        optimizer: The optimizer for model optimization.
        device: The device to use for training and evaluation.

    """

    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device, modelname):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = modelname

    def train(self, num_epochs):
        """
        Train the model for the specified number of epochs.

        Args:
            num_epochs (int): The number of epochs to train.

        """

        self.model.to(self.device)
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0

            for batch_num, batch in enumerate(self.train_loader, 1):
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if batch_num % 20 == 0:
                    print("\rEpoch: {}, Batch: {} => Loss: {:.4f}".format(epoch, batch_num, loss.item()), flush=True,
                          end='')

            # Evaluate after each epoch
            self.evaluate()

            # Save the model
            torch.save(self.model.state_dict(), 'weights/{}{}.pth'.format(self.model_name, epoch + 1))

            print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(self.train_loader):.4f}')

    def evaluate(self):
        """
        Evaluate the model on the test data and print the micro F1-score and accuracy.

        """

        self.model.to(self.device)
        self.model.eval()
        total_preds = []
        total_labels = []

        for batch_num, batch in enumerate(self.test_loader, 1):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            preds = logits.argmax(dim=-1)
            total_labels.append(labels.cpu().data.numpy())
            total_preds.append(preds.cpu().data.numpy())

        total_preds = np.concatenate(total_preds)
        total_labels = np.concatenate(total_labels)
        accuracy = sum(total_preds == total_labels) / len(total_labels)
        micro_f1 = f1_score(total_labels, total_preds, average='micro')

        print("Micro F1-Score: {:.2%}".format(micro_f1))
        print("Accuracy: {:.2%}".format(accuracy))
