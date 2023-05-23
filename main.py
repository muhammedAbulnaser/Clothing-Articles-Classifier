# imports the required dependencies for the upcoming tasks, such as loading and processing data, training models, and visualizing results.
import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import argparse



from src.preprocessing import clean_unexisting_files, filter_dataframe
from src.ClothDataset import ClothDataset
from src.ModelTrainerEvaluater import ModelTrainerEvaluater
from src.ModelLoader import ModelLoader



def parsing():
  # Create argument parser
  parser = argparse.ArgumentParser(description='Training Models')

  # Add arguments
  parser.add_argument('--data_path', type=str, default="data/", help='path to data folder') # structure should be data folder ---> images folder and csv file
  parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
  parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
  parser.add_argument('--model_type', type=str, default="vgg16", help='model type between vgg16, resnet, mobilenet')

 # Parse the arguments
  args = parser.parse_args()
  return args
  


if __name__ == "__main__":

    args = parsing()

    model_type = args.model_type
    num_epochs = args.epochs
    num_classes = args.num_classes
    DATASET_PATH = args.data_path  # Specify the path to the dataset directory.

    # Read the "styles.csv" file from the dataset directory, ignoring lines with errors.
    df = pd.read_csv(os.path.join(DATASET_PATH, "styles.csv"), error_bad_lines=False)

    # Clean the DataFrame by removing rows with non-existing image files in the dataset
    print("Starting to clean the data...")
    df = clean_unexisting_files(df, DATASET_PATH)

    # Filter the DataFrame to include only categories with counts above the threshold (1000) image
    df = filter_dataframe(df)

    # Split train and test dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=2028)

    # Construct dataset and dataloader
    train_ds = ClothDataset(os.path.join(DATASET_PATH, 'images'), train_df, df) 
    test_ds = ClothDataset(os.path.join(DATASET_PATH, 'images'), test_df, df)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)


    loader = ModelLoader(model_type, num_classes)
    model = loader.load_model()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Set the device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create an instance of the ModelTrainer class
    trainer = ModelTrainerEvaluater(model, train_loader, test_loader, criterion, optimizer, device, model_type)

    # Train the model
    trainer.train(num_epochs)




