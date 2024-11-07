import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import scipy.sparse

class CustomNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)

def neuralNetOut(X, y, save_path='nn_results.txt', model_save_path='nn_model.joblib', 
                 epochs=100, batch_size=32, learning_rate=0.001, hidden_size=128, test_size=0.2):
    print("Starting Neural Network...")
    
    # Convert sparse matrix to dense if needed
    if scipy.sparse.issparse(X):
        X = X.toarray()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and training components
    model = CustomNN(input_size=X.shape[1], hidden_size=hidden_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    best_val_accuracy = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            val_pred = torch.argmax(val_outputs, dim=1)
            val_accuracy = accuracy_score(y_test_tensor, val_pred)
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss.item())
        val_accuracies.append(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    # Load best model and make final predictions
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        final_predictions = torch.argmax(model(X_test_tensor), dim=1).numpy()
    
    # Save results
    with open(save_path, 'w') as f:
        f.write("Neural Network Results\n")
        f.write("======================\n\n")
        f.write(f"Best validation accuracy: {best_val_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, final_predictions, target_names=label_encoder.classes_))
    
    print(f"Neural Network results saved to {save_path}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, final_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Neural Network - Confusion Matrix')
    plt.savefig("nn_confusion_matrix.png")
    plt.close()
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("nn_learning_curves.png")
    plt.close()
    
    # Save model
    if model_save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'label_encoder': label_encoder,
            'model_config': {
                'input_size': X.shape[1],
                'hidden_size': hidden_size,
                'num_classes': num_classes
            }
        }, model_save_path)
        print(f'\nModel saved to {model_save_path}')
    
    return model, scaler, label_encoder