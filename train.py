# Training functionality for the MNIST classifier

def train_model(model, train_data, loss_function, optimizer, epochs):
    """
    Train the neural network model
    
    Args:
        model: The neural network model
        train_data: DataLoader with training data
        loss_function: Loss function for training
        optimizer: Optimizer for updating weights
        epochs: Number of training epochs
    """
    print("Starting training...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        last_batch_size = 0
        
        for images, labels in train_data:
            # Reshape images to vectors
            images = images.view(images.shape[0], -1)
            last_batch_size = len(labels)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            running_loss += loss.item() * images.size(0)
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_data.dataset)
        print(f"Epoch: {epoch+1}/{epochs}\tLoss: {epoch_loss:.4f}")
    
    print("Training complete!")
