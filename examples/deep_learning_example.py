"""
Deep Learning examples for MZ Max

This script demonstrates the deep learning capabilities of MZ Max.
"""

import mz_max as mz
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def pytorch_classification_example():
    """Example of PyTorch classification with MZ Max."""
    print("="*50)
    print("PYTORCH CLASSIFICATION EXAMPLE")
    print("="*50)
    
    # Load and prepare data
    print("Loading iris dataset...")
    data = mz.load_dataset('iris')
    
    # Prepare features and target
    X = data.drop('target', axis=1).values
    y = pd.get_dummies(data['target']).values  # One-hot encode for multi-class
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Create a simple neural network
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return self.softmax(x)
    
    # Create model
    input_size = X_train.shape[1]
    hidden_size = 64
    num_classes = y_train.shape[1]
    
    pytorch_model = SimpleNN(input_size, hidden_size, num_classes)
    
    # Wrap with MZ Max PyTorchModel
    print("\nCreating MZ Max PyTorch model...")
    model = mz.deep_learning.PyTorchModel(pytorch_model)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='crossentropy',
        learning_rate=0.001
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        verbose=True
    )
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Evaluate model
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Print model summary
    print("\nModel Summary:")
    print(model.summary())
    
    return model, history


def cnn_image_example():
    """Example of CNN for image classification."""
    print("\n" + "="*50)
    print("CNN IMAGE CLASSIFICATION EXAMPLE")
    print("="*50)
    
    # Generate synthetic image data for demo
    print("Generating synthetic image data...")
    num_samples = 1000
    img_height, img_width, channels = 32, 32, 3
    num_classes = 10
    
    # Random image data
    X = np.random.randn(num_samples, channels, img_height, img_width).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    
    # Convert to one-hot
    y_onehot = np.eye(num_classes)[y]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )
    
    print(f"Training images shape: {X_train.shape}")
    print(f"Test images shape: {X_test.shape}")
    
    # Create CNN model
    print("\nCreating CNN model...")
    cnn_model = mz.deep_learning.CNN(
        input_channels=channels,
        num_classes=num_classes,
        conv_layers=[32, 64, 128],
        dropout=0.5
    )
    
    # Wrap with MZ Max
    model = mz.deep_learning.PyTorchModel(cnn_model)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='crossentropy',
        learning_rate=0.001
    )
    
    # Train model (fewer epochs for demo)
    print("\nTraining CNN...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,  # Few epochs for demo
        batch_size=32,
        verbose=True
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Evaluate model
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    return model, history


def transformer_example():
    """Example of Transformer model."""
    print("\n" + "="*50)
    print("TRANSFORMER EXAMPLE")
    print("="*50)
    
    # Generate synthetic sequence data
    print("Generating synthetic sequence data...")
    vocab_size = 1000
    seq_length = 50
    num_samples = 500
    num_classes = 5
    
    # Random sequence data
    X = np.random.randint(0, vocab_size, (num_samples, seq_length))
    y = np.random.randint(0, num_classes, num_samples)
    y_onehot = np.eye(num_classes)[y]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )
    
    print(f"Training sequences shape: {X_train.shape}")
    print(f"Test sequences shape: {X_test.shape}")
    
    # Create Transformer model
    print("\nCreating Transformer model...")
    transformer_model = mz.deep_learning.Transformer(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        num_classes=num_classes,
        dropout=0.1
    )
    
    # Wrap with MZ Max
    model = mz.deep_learning.PyTorchModel(transformer_model)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='crossentropy',
        learning_rate=0.0001
    )
    
    # Train model (fewer epochs for demo)
    print("\nTraining Transformer...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=3,  # Few epochs for demo
        batch_size=16,
        verbose=True
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Evaluate model
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    return model, history


def lstm_sequence_example():
    """Example of LSTM for sequence modeling."""
    print("\n" + "="*50)
    print("LSTM SEQUENCE EXAMPLE")
    print("="*50)
    
    # Generate synthetic time series data
    print("Generating synthetic time series data...")
    seq_length = 20
    input_size = 5
    num_samples = 800
    num_classes = 3
    
    # Random sequence data
    X = np.random.randn(num_samples, seq_length, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    y_onehot = np.eye(num_classes)[y]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )
    
    print(f"Training sequences shape: {X_train.shape}")
    print(f"Test sequences shape: {X_test.shape}")
    
    # Create LSTM model
    print("\nCreating LSTM model...")
    lstm_model = mz.deep_learning.LSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.2,
        bidirectional=True
    )
    
    # Wrap with MZ Max
    model = mz.deep_learning.PyTorchModel(lstm_model)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='crossentropy',
        learning_rate=0.001
    )
    
    # Train model
    print("\nTraining LSTM...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=True
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Evaluate model
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    return model, history


def advanced_training_example():
    """Example of advanced training features."""
    print("\n" + "="*50)
    print("ADVANCED TRAINING EXAMPLE")
    print("="*50)
    
    # Load data
    data = mz.load_dataset('iris')
    X = data.drop('target', axis=1).values
    y = pd.get_dummies(data['target']).values
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create model
    class AdvancedNN(nn.Module):
        def __init__(self):
            super(AdvancedNN, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(4, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 3),
                nn.Softmax(dim=1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model_nn = AdvancedNN()
    model = mz.deep_learning.PyTorchModel(model_nn)
    
    # Compile with advanced options
    print("\nCompiling model with advanced options...")
    model.compile(
        optimizer='adamw',
        loss='crossentropy',
        scheduler='cosine',
        learning_rate=0.001
    )
    
    # Create advanced trainer
    print("\nCreating advanced trainer...")
    trainer = mz.deep_learning.PyTorchTrainer(
        model=model,
        early_stopping_patience=10,
        model_checkpoint_path='best_model.pth'
    )
    
    # Prepare data loaders
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Train with advanced features
    print("\nTraining with advanced trainer...")
    history = trainer.train_with_validation(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        verbose=True
    )
    
    print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
    print("Model checkpoint saved to 'best_model.pth'")
    
    return model, history


def main():
    """Run all deep learning examples."""
    print("MZ Max - Deep Learning Examples")
    print("===============================")
    
    # Check if PyTorch is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Run examples
        pytorch_classification_example()
        cnn_image_example()
        transformer_example()
        lstm_sequence_example()
        advanced_training_example()
        
        print("\n" + "="*50)
        print("ALL DEEP LEARNING EXAMPLES COMPLETED!")
        print("="*50)
        print("\nGenerated files:")
        print("  - best_model.pth (model checkpoint)")
        
    except ImportError:
        print("PyTorch not available. Please install PyTorch to run deep learning examples.")
        print("Install with: pip install torch torchvision")


if __name__ == "__main__":
    main()
