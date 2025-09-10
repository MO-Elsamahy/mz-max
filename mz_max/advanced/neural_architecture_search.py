"""
Neural Architecture Search (NAS) for MZ Max

This module provides automated neural architecture search capabilities
to find optimal network architectures for specific tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from itertools import product
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.base import BaseModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchSpace:
    """Define search space for neural architecture search."""
    
    layers: List[str] = None  # ['conv', 'linear', 'attention', 'lstm']
    activations: List[str] = None  # ['relu', 'gelu', 'swish', 'tanh']
    optimizers: List[str] = None  # ['adam', 'adamw', 'sgd', 'rmsprop']
    learning_rates: List[float] = None  # [0.001, 0.01, 0.1]
    batch_sizes: List[int] = None  # [16, 32, 64, 128]
    hidden_sizes: List[int] = None  # [64, 128, 256, 512]
    num_layers: List[int] = None  # [2, 3, 4, 5]
    dropout_rates: List[float] = None  # [0.0, 0.1, 0.2, 0.3]
    
    def __post_init__(self):
        """Initialize default values."""
        if self.layers is None:
            self.layers = ['conv', 'linear', 'attention']
        if self.activations is None:
            self.activations = ['relu', 'gelu', 'swish']
        if self.optimizers is None:
            self.optimizers = ['adam', 'adamw']
        if self.learning_rates is None:
            self.learning_rates = [0.001, 0.01]
        if self.batch_sizes is None:
            self.batch_sizes = [32, 64]
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 256, 512]
        if self.num_layers is None:
            self.num_layers = [2, 3, 4]
        if self.dropout_rates is None:
            self.dropout_rates = [0.1, 0.2, 0.3]


class ArchitectureBuilder:
    """Build neural architectures from configuration."""
    
    def __init__(self, input_size: int, output_size: int, task_type: str = 'classification'):
        """
        Initialize architecture builder.
        
        Args:
            input_size: Input feature size
            output_size: Output size
            task_type: Type of task ('classification' or 'regression')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.task_type = task_type
    
    def build_architecture(self, config: Dict[str, Any]) -> nn.Module:
        """
        Build neural network from configuration.
        
        Args:
            config: Architecture configuration
            
        Returns:
            PyTorch neural network model
        """
        layers = []
        current_size = self.input_size
        
        # Build hidden layers
        for i in range(config['num_layers']):
            hidden_size = config['hidden_sizes'][i] if isinstance(config['hidden_sizes'], list) else config['hidden_sizes']
            
            if config['layer_type'] == 'linear':
                layers.append(nn.Linear(current_size, hidden_size))
            elif config['layer_type'] == 'conv' and len(self.input_size) > 1:
                # Convolutional layer for image-like data
                if i == 0:
                    layers.append(nn.Conv2d(self.input_size[0], hidden_size, kernel_size=3, padding=1))
                else:
                    layers.append(nn.Conv2d(current_size, hidden_size, kernel_size=3, padding=1))
            else:
                layers.append(nn.Linear(current_size, hidden_size))
            
            # Add activation
            if config['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif config['activation'] == 'gelu':
                layers.append(nn.GELU())
            elif config['activation'] == 'swish':
                layers.append(nn.SiLU())  # SiLU is Swish
            elif config['activation'] == 'tanh':
                layers.append(nn.Tanh())
            
            # Add dropout
            if config.get('dropout_rate', 0) > 0:
                layers.append(nn.Dropout(config['dropout_rate']))
            
            current_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(current_size, self.output_size))
        
        # Add output activation for classification
        if self.task_type == 'classification' and self.output_size > 1:
            layers.append(nn.Softmax(dim=1))
        
        return nn.Sequential(*layers)
    
    def build_attention_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build attention-based model."""
        
        class AttentionModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_heads=8):
                super().__init__()
                self.embedding = nn.Linear(input_size, hidden_size)
                self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)
                self.feed_forward = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                self.output = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(config.get('dropout_rate', 0.1))
            
            def forward(self, x):
                # Add sequence dimension if needed
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                
                # Embedding
                x = self.embedding(x)
                
                # Self-attention
                attn_output, _ = self.attention(x, x, x)
                x = self.norm1(x + self.dropout(attn_output))
                
                # Feed forward
                ff_output = self.feed_forward(x)
                x = self.norm2(x + self.dropout(ff_output))
                
                # Output
                x = x.mean(dim=1)  # Global average pooling
                return self.output(x)
        
        hidden_size = config.get('hidden_sizes', 256)
        if isinstance(hidden_size, list):
            hidden_size = hidden_size[0]
        
        return AttentionModel(self.input_size, hidden_size, self.output_size)


class NASOptimizer:
    """
    Neural Architecture Search optimizer using evolutionary algorithm.
    """
    
    def __init__(self, search_space: SearchSpace, 
                 population_size: int = 20,
                 generations: int = 10,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        """
        Initialize NAS optimizer.
        
        Args:
            search_space: Architecture search space
            population_size: Size of population for evolutionary search
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.search_space = search_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.fitness_scores = []
        self.best_architecture = None
        self.best_score = -float('inf')
    
    def generate_random_architecture(self) -> Dict[str, Any]:
        """Generate a random architecture configuration."""
        config = {
            'layer_type': random.choice(self.search_space.layers),
            'activation': random.choice(self.search_space.activations),
            'optimizer': random.choice(self.search_space.optimizers),
            'learning_rate': random.choice(self.search_space.learning_rates),
            'batch_size': random.choice(self.search_space.batch_sizes),
            'hidden_sizes': random.choice(self.search_space.hidden_sizes),
            'num_layers': random.choice(self.search_space.num_layers),
            'dropout_rate': random.choice(self.search_space.dropout_rates)
        }
        return config
    
    def initialize_population(self):
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            self.population.append(self.generate_random_architecture())
    
    def evaluate_architecture(self, config: Dict[str, Any], 
                            X_train: torch.Tensor, y_train: torch.Tensor,
                            X_val: torch.Tensor, y_val: torch.Tensor,
                            input_size: int, output_size: int, 
                            task_type: str = 'classification') -> float:
        """
        Evaluate architecture performance.
        
        Args:
            config: Architecture configuration
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            input_size: Input feature size
            output_size: Output size
            task_type: Task type
            
        Returns:
            Validation accuracy/score
        """
        try:
            # Build model
            builder = ArchitectureBuilder(input_size, output_size, task_type)
            
            if config['layer_type'] == 'attention':
                model = builder.build_attention_model(config)
            else:
                model = builder.build_architecture(config)
            
            # Setup training
            if config['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            elif config['optimizer'] == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
            elif config['optimizer'] == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
            else:
                optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            
            if task_type == 'classification':
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()
            
            # Quick training (limited epochs for NAS)
            model.train()
            for epoch in range(5):  # Limited training for speed
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                
                if task_type == 'classification':
                    _, predicted = torch.max(val_outputs.data, 1)
                    accuracy = (predicted == y_val).float().mean().item()
                    return accuracy
                else:
                    val_loss = criterion(val_outputs, y_val)
                    return -val_loss.item()  # Negative loss as score
        
        except Exception as e:
            logger.warning(f"Architecture evaluation failed: {e}")
            return -1.0  # Poor score for failed architectures
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Create offspring through crossover."""
        offspring = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                offspring[key] = parent1[key]
            else:
                offspring[key] = parent2[key]
        return offspring
    
    def mutate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate architecture configuration."""
        mutated = config.copy()
        
        for key, value in mutated.items():
            if random.random() < self.mutation_rate:
                if key == 'layer_type':
                    mutated[key] = random.choice(self.search_space.layers)
                elif key == 'activation':
                    mutated[key] = random.choice(self.search_space.activations)
                elif key == 'optimizer':
                    mutated[key] = random.choice(self.search_space.optimizers)
                elif key == 'learning_rate':
                    mutated[key] = random.choice(self.search_space.learning_rates)
                elif key == 'batch_size':
                    mutated[key] = random.choice(self.search_space.batch_sizes)
                elif key == 'hidden_sizes':
                    mutated[key] = random.choice(self.search_space.hidden_sizes)
                elif key == 'num_layers':
                    mutated[key] = random.choice(self.search_space.num_layers)
                elif key == 'dropout_rate':
                    mutated[key] = random.choice(self.search_space.dropout_rates)
        
        return mutated
    
    def search(self, X_train: torch.Tensor, y_train: torch.Tensor,
               X_val: torch.Tensor, y_val: torch.Tensor,
               input_size: int, output_size: int,
               task_type: str = 'classification') -> Dict[str, Any]:
        """
        Perform neural architecture search.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            input_size: Input feature size
            output_size: Output size
            task_type: Task type
            
        Returns:
            Best architecture configuration
        """
        logger.info("Starting Neural Architecture Search...")
        
        # Initialize population
        self.initialize_population()
        
        for generation in range(self.generations):
            logger.info(f"Generation {generation + 1}/{self.generations}")
            
            # Evaluate population
            self.fitness_scores = []
            for config in self.population:
                score = self.evaluate_architecture(
                    config, X_train, y_train, X_val, y_val,
                    input_size, output_size, task_type
                )
                self.fitness_scores.append(score)
                
                # Track best architecture
                if score > self.best_score:
                    self.best_score = score
                    self.best_architecture = config.copy()
            
            logger.info(f"Best score so far: {self.best_score:.4f}")
            
            # Selection and reproduction
            new_population = []
            
            # Keep best individuals (elitism)
            sorted_indices = sorted(range(len(self.fitness_scores)), 
                                  key=lambda i: self.fitness_scores[i], reverse=True)
            elite_size = max(1, self.population_size // 4)
            
            for i in range(elite_size):
                new_population.append(self.population[sorted_indices[i]].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring = self.crossover(parent1, parent2)
                else:
                    offspring = parent1.copy()
                
                # Mutation
                offspring = self.mutate(offspring)
                new_population.append(offspring)
            
            self.population = new_population
        
        logger.info(f"NAS completed. Best architecture score: {self.best_score:.4f}")
        return self.best_architecture
    
    def _tournament_selection(self, tournament_size: int = 3) -> Dict[str, Any]:
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(self.population)), 
                                          min(tournament_size, len(self.population)))
        tournament_scores = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_scores.index(max(tournament_scores))]
        return self.population[winner_idx]


class AutoArchitecture(BaseModel):
    """
    Automated architecture search and training.
    """
    
    def __init__(self, search_space: Optional[SearchSpace] = None,
                 nas_generations: int = 10,
                 nas_population: int = 20,
                 final_epochs: int = 100,
                 task_type: str = 'classification'):
        """
        Initialize AutoArchitecture.
        
        Args:
            search_space: Architecture search space
            nas_generations: NAS generations
            nas_population: NAS population size
            final_epochs: Final training epochs
            task_type: Task type
        """
        super().__init__()
        self.search_space = search_space or SearchSpace()
        self.nas_generations = nas_generations
        self.nas_population = nas_population
        self.final_epochs = final_epochs
        self.task_type = task_type
        self.nas_optimizer = None
        self.best_config = None
        self._fitted = False
    
    def fit(self, X, y, validation_split: float = 0.2):
        """
        Fit the model using neural architecture search.
        
        Args:
            X: Training data
            y: Training labels
            validation_split: Validation split ratio
        """
        # Convert to tensors
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            if self.task_type == 'classification':
                y = torch.LongTensor(y)
            else:
                y = torch.FloatTensor(y)
        
        # Split data
        n_val = int(len(X) * validation_split)
        indices = torch.randperm(len(X))
        
        X_train, X_val = X[indices[n_val:]], X[indices[:n_val]]
        y_train, y_val = y[indices[n_val:]], y[indices[:n_val]]
        
        # Determine sizes
        input_size = X.shape[1]
        if self.task_type == 'classification':
            output_size = len(torch.unique(y))
        else:
            output_size = 1 if y.dim() == 1 else y.shape[1]
        
        # Perform NAS
        self.nas_optimizer = NASOptimizer(
            self.search_space,
            population_size=self.nas_population,
            generations=self.nas_generations
        )
        
        self.best_config = self.nas_optimizer.search(
            X_train, y_train, X_val, y_val,
            input_size, output_size, self.task_type
        )
        
        # Build and train final model
        builder = ArchitectureBuilder(input_size, output_size, self.task_type)
        
        if self.best_config['layer_type'] == 'attention':
            self._model = builder.build_attention_model(self.best_config)
        else:
            self._model = builder.build_architecture(self.best_config)
        
        # Final training
        if self.best_config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(self._model.parameters(), lr=self.best_config['learning_rate'])
        else:
            optimizer = optim.Adam(self._model.parameters(), lr=self.best_config['learning_rate'])
        
        if self.task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # Training loop
        self._model.train()
        for epoch in range(self.final_epochs):
            optimizer.zero_grad()
            outputs = self._model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.final_epochs}, Loss: {loss.item():.4f}")
        
        self._fitted = True
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X)
            
            if self.task_type == 'classification':
                _, predicted = torch.max(outputs.data, 1)
                return predicted.numpy()
            else:
                return outputs.numpy()
    
    def _compute_metrics(self, y_true, y_pred):
        """Compute evaluation metrics."""
        from ..core.metrics import Metrics
        
        if self.task_type == 'classification':
            return Metrics.classification_metrics(y_true, y_pred)
        else:
            return Metrics.regression_metrics(y_true, y_pred)
    
    def get_best_architecture(self) -> Optional[Dict[str, Any]]:
        """Get the best found architecture configuration."""
        return self.best_config
    
    def get_search_history(self) -> Dict[str, Any]:
        """Get NAS search history."""
        if self.nas_optimizer is None:
            return {}
        
        return {
            'best_score': self.nas_optimizer.best_score,
            'generations': self.nas_generations,
            'population_size': self.nas_population,
            'best_config': self.best_config
        }
