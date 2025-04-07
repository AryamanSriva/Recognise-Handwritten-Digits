# Utility functions for the MNIST classification project

import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_images(images, num_images=30):
    """Visualize images from the dataset"""
    rows = (num_images + 9) // 10
    plt.figure(figsize=(15, rows * 2))
    
    for i in range(min(num_images, len(images))):
        plt.subplot(rows, 10, i+1)
        plt.subplots_adjust(wspace=0.3)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def get_predicted_label(model, image):
    """Get the predicted label for a single image"""
    # Reshape image to a flat vector
    image = image.view(1, 784)  # 28x28 = 784
    
    # Disable gradient calculation for prediction
    with torch.no_grad():
        prediction_score = model(image)
    
    # Return the class with highest probability
    return np.argmax(prediction_score)
