# Data loader module for MNIST dataset

import torch
from torchvision import datasets, transforms

def get_transformations():
    """Create and return transformations for the MNIST dataset"""
    normalized = transforms.Normalize((0.5,), (0.5,))
    tensor = transforms.ToTensor()
    transformation = transforms.Compose([tensor, normalized])
    return transformation

def get_data_loaders(batch_size=64):
    """Download and load MNIST dataset"""
    transformation = get_transformations()
    
    # Download and load training dataset
    training_dataset = datasets.MNIST('/bytefiles', download=True, train=True, transform=transformation)
    # Download and load testing dataset
    testing_dataset = datasets.MNIST('/bytefiles', download=True, train=False, transform=transformation)
    
    # Create data loaders
    train_data = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)
    
    return train_data, test_data
