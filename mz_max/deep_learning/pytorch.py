"""
PyTorch integration for MZ Max

This module provides PyTorch-specific implementations and utilities
for deep learning in MZ Max.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from ..core.base import BaseModel
from ..core.exceptions import TrainingError, PredictionError


class PyTorchModel(BaseModel):
    """
    Base class for PyTorch models in MZ Max.
    
    This class provides a unified interface for PyTorch models with
    training, evaluation, and prediction capabilities.
    """
    
    def __init__(self, model: nn.Module, device: Optional[str] = None, **kwargs):
        """
        Initialize PyTorch model wrapper.
        
        Args:
            model: PyTorch model
            device: Device to use ('cuda', 'cpu', or 'auto')
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self._model = model
        self.device = self._get_device(device)
        self._model.to(self.device)
        self._optimizer = None
        self._criterion = None
        self._scheduler = None
    
    def _get_device(self, device: Optional[str] = None) -> torch.device:
        """Get the appropriate device."""
        if device is None or device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def compile(self, optimizer: Union[str, optim.Optimizer] = 'adam',
                loss: Union[str, nn.Module] = 'mse',
                scheduler: Optional[Union[str, Any]] = None,
                learning_rate: float = 0.001,
                **kwargs) -> None:
        """
        Compile the model with optimizer, loss function, and scheduler.
        
        Args:
            optimizer: Optimizer name or instance
            loss: Loss function name or instance
            scheduler: Scheduler name or instance
            learning_rate: Learning rate
            **kwargs: Additional arguments
        """
        # Set up optimizer
        if isinstance(optimizer, str):
            optimizer_class = getattr(optim, optimizer.title())
            self._optimizer = optimizer_class(self._model.parameters(), lr=learning_rate, **kwargs)
        else:
            self._optimizer = optimizer
        
        # Set up loss function
        if isinstance(loss, str):
            loss_map = {
                'mse': nn.MSELoss(),
                'mae': nn.L1Loss(),
                'crossentropy': nn.CrossEntropyLoss(),
                'bce': nn.BCELoss(),
                'bce_logits': nn.BCEWithLogitsLoss(),
                'huber': nn.HuberLoss(),
            }
            self._criterion = loss_map.get(loss.lower(), nn.MSELoss())
        else:
            self._criterion = loss
        
        # Set up scheduler
        if scheduler is not None:
            if isinstance(scheduler, str):
                if scheduler.lower() == 'steplr':
                    self._scheduler = optim.lr_scheduler.StepLR(self._optimizer, step_size=30, gamma=0.1)
                elif scheduler.lower() == 'cosine':
                    self._scheduler = optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=100)
                elif scheduler.lower() == 'reduce_plateau':
                    self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, patience=10)
            else:
                self._scheduler = scheduler
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame, DataLoader], 
            y: Optional[Union[np.ndarray, pd.Series]] = None,
            validation_data: Optional[Tuple] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: bool = True,
            callbacks: Optional[List] = None) -> Dict[str, List[float]]:
        """
        Train the PyTorch model.
        
        Args:
            X: Training data or DataLoader
            y: Training targets (if X is not DataLoader)
            validation_data: Validation data tuple (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print training progress
            callbacks: List of callbacks
            
        Returns:
            Training history dictionary
        """
        if self._optimizer is None or self._criterion is None:
            raise TrainingError("Model must be compiled before training")
        
        # Prepare data loader
        if isinstance(X, DataLoader):
            train_loader = X
        else:
            train_dataset = TensorDataset(X, y)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation data loader
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            self._model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self._optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = self._criterion(outputs, batch_y)
                loss.backward()
                self._optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            history['loss'].append(avg_train_loss)
            
            # Validation phase
            val_loss = None
            if val_loader is not None:
                self._model.eval()
                val_loss_total = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self._model(batch_X)
                        loss = self._criterion(outputs, batch_y)
                        val_loss_total += loss.item()
                        val_batches += 1
                
                val_loss = val_loss_total / val_batches
                history['val_loss'].append(val_loss)
            
            # Update scheduler
            if self._scheduler is not None:
                if isinstance(self._scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step(val_loss if val_loss is not None else avg_train_loss)
                else:
                    self._scheduler.step()
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}")
        
        self._fitted = True
        self._training_history = history
        return history
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame, DataLoader]) -> np.ndarray:
        """
        Make predictions with the PyTorch model.
        
        Args:
            X: Input data or DataLoader
            
        Returns:
            Predictions as numpy array
        """
        if not self.is_fitted():
            raise PredictionError("Model must be fitted before making predictions")
        
        self._model.eval()
        predictions = []
        
        # Prepare data loader
        if isinstance(X, DataLoader):
            data_loader = X
        else:
            X_tensor = self._to_tensor(X)
            dataset = torch.utils.data.TensorDataset(X_tensor)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    batch_X = batch[0].to(self.device)
                else:
                    batch_X = batch.to(self.device)
                
                outputs = self._model(batch_X)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def _to_tensor(self, X: Union[np.ndarray, pd.DataFrame]) -> torch.Tensor:
        """Convert input data to PyTorch tensor."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if isinstance(X, np.ndarray):
            return torch.FloatTensor(X)
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")
    
    def _compute_metrics(self, y_true: Union[np.ndarray, pd.Series], y_pred: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from ..core.metrics import Metrics
        
        # Determine task type based on output shape and values
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            # Multi-class classification
            y_pred_labels = np.argmax(y_pred, axis=1)
            return Metrics.classification_metrics(y_true, y_pred_labels, y_pred)
        elif len(np.unique(y_true)) <= 10 and np.all(np.unique(y_true) == np.unique(y_true).astype(int)):
            # Classification (small number of unique integer values)
            return Metrics.classification_metrics(y_true, y_pred.round())
        else:
            # Regression
            return Metrics.regression_metrics(y_true, y_pred)
    
    def save_weights(self, filepath: str) -> None:
        """Save model weights."""
        torch.save(self._model.state_dict(), filepath)
    
    def load_weights(self, filepath: str) -> None:
        """Load model weights."""
        self._model.load_state_dict(torch.load(filepath, map_location=self.device))
    
    def summary(self) -> str:
        """Get model summary."""
        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        
        summary = f"""
Model Summary:
==============
Architecture: {self._model.__class__.__name__}
Device: {self.device}
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Non-trainable parameters: {total_params - trainable_params:,}

Model Structure:
{self._model}
"""
        return summary


class TensorDataset(Dataset):
    """
    Custom dataset class for handling numpy arrays and pandas DataFrames.
    """
    
    def __init__(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None):
        """
        Initialize tensor dataset.
        
        Args:
            X: Input features
            y: Target values (optional)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get item by index."""
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


class PyTorchTrainer:
    """
    Advanced trainer for PyTorch models with additional features.
    """
    
    def __init__(self, model: PyTorchModel, 
                 early_stopping_patience: int = 10,
                 model_checkpoint_path: Optional[str] = None):
        """
        Initialize PyTorch trainer.
        
        Args:
            model: PyTorchModel instance
            early_stopping_patience: Patience for early stopping
            model_checkpoint_path: Path to save model checkpoints
        """
        self.model = model
        self.early_stopping_patience = early_stopping_patience
        self.model_checkpoint_path = model_checkpoint_path
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_with_validation(self, train_loader: DataLoader, 
                            val_loader: DataLoader,
                            epochs: int = 100,
                            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train model with validation and advanced features.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        history = {'loss': [], 'val_loss': [], 'lr': []}
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            history['loss'].append(train_loss)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            history['val_loss'].append(val_loss)
            
            # Learning rate tracking
            current_lr = self.model._optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                if self.model_checkpoint_path:
                    self.model.save_weights(self.model_checkpoint_path)
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Update scheduler
            if self.model._scheduler is not None:
                if isinstance(self.model._scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.model._scheduler.step(val_loss)
                else:
                    self.model._scheduler.step()
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - lr: {current_lr:.6f}")
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model._model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.model.device), batch_y.to(self.model.device)
            
            self.model._optimizer.zero_grad()
            outputs = self.model._model(batch_X)
            loss = self.model._criterion(outputs, batch_y)
            loss.backward()
            self.model._optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model._model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.model.device), batch_y.to(self.model.device)
                outputs = self.model._model(batch_X)
                loss = self.model._criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches


# Alias for backward compatibility
PyTorchDataLoader = DataLoader
