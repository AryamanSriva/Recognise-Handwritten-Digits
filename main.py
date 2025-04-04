# Main script to run the MNIST digit classification project

import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim

from data_loader import get_data_loaders
from model import MNISTClassifier
from utils import get_predicted_label, visualize_images
from train import train_model
from test import test_model

def main():
    # Get data loaders
    train_data, test_data = get_data_loaders()
    
    # Visualize some training images
    images, labels = next(iter(train_data))
    visualize_images(images)
    
    # Create model
    input_layer = 784  # 28x28 pixels
    hidden_layer1 = 64
    hidden_layer2 = 32
    output_layer = 10  # 10 digits
    
    model = MNISTClassifier(input_layer, hidden_layer1, hidden_layer2, output_layer)
    
    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    gradient_descent = optim.SGD(model.parameters(), lr=0.1)
    
    # Train the model
    epochs = 20
    train_model(model, train_data, loss_function, gradient_descent, epochs)
    
    # Test a single image
    test_images, test_labels = next(iter(test_data))
    predicted = get_predicted_label(model, test_images[0])
    actual = test_labels.numpy()[0]
    print(f"Single test - Predicted: {predicted}, Actual: {actual}")
    
    # Test the entire dataset
    accuracy = test_model(model, test_data)
    print(f"Accuracy percentage: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
