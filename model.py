# Neural network model for MNIST classification

import torch
from torch import nn

class MNISTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        """
        Initialize the neural network for MNIST classification
        
        Args:
            input_size: Size of the input layer (784 for MNIST)
            hidden_size1: Size of the first hidden layer
            hidden_size2: Size of the second hidden layer
            output_size: Size of the output layer (10 for digits 0-9)
        """
        super(MNISTClassifier, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)
