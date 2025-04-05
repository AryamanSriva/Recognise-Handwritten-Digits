# Testing functionality for the MNIST classifier

from utils import get_predicted_label

def test_model(model, test_data, verbose=False):
    """
    Test the neural network model on the test dataset
    
    Args:
        model: The trained neural network model
        test_data: DataLoader with test data
        verbose: Whether to print predictions for each image
    
    Returns:
        float: Accuracy percentage
    """
    total_count = 0
    accurate_count = 0
    
    for images, labels in test_data:
        for i in range(len(labels)):
            predicted_label = get_predicted_label(model, images[i])
            actual_label = labels.numpy()[i]
            
            if verbose:
                print(f"Predicted Label: {predicted_label} / Actual Label: {actual_label}")
            
            if predicted_label == actual_label:
                accurate_count += 1
                
        total_count += len(labels)
    
    print(f"Total images tested: {total_count}")
    print(f"Accurate predictions: {accurate_count}")
    
    accuracy = (accurate_count / total_count) * 100
    return accuracy
