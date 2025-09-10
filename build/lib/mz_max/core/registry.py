"""
Model registry for MZ Max

This module provides a centralized registry for managing models, datasets,
and other components in the MZ Max ecosystem.
"""

from typing import Dict, Any, List, Optional, Callable, Type
from abc import ABC, abstractmethod
import inspect
from .exceptions import ModelNotFoundError


class Registry(ABC):
    """
    Abstract base class for registries.
    """
    
    def __init__(self):
        self._registry = {}
    
    @abstractmethod
    def register(self, name: str, item: Any, **kwargs) -> None:
        """Register an item in the registry."""
        pass
    
    @abstractmethod
    def get(self, name: str) -> Any:
        """Get an item from the registry."""
        pass
    
    def list_items(self) -> List[str]:
        """List all registered items."""
        return list(self._registry.keys())
    
    def exists(self, name: str) -> bool:
        """Check if an item exists in the registry."""
        return name in self._registry
    
    def remove(self, name: str) -> None:
        """Remove an item from the registry."""
        if name in self._registry:
            del self._registry[name]
        else:
            raise KeyError(f"Item '{name}' not found in registry")


class ModelRegistry(Registry):
    """
    Registry for machine learning models.
    
    This registry allows registration and retrieval of model classes,
    factory functions, and pre-trained models.
    """
    
    def register(self, name: str, model_class: Type, 
                 description: str = "", 
                 task_type: str = "general",
                 framework: str = "sklearn",
                 **kwargs) -> None:
        """
        Register a model class.
        
        Args:
            name: Model name
            model_class: Model class or factory function
            description: Model description
            task_type: Type of ML task (classification, regression, etc.)
            framework: ML framework (sklearn, pytorch, tensorflow, etc.)
            **kwargs: Additional metadata
        """
        if not (inspect.isclass(model_class) or callable(model_class)):
            raise ValueError("model_class must be a class or callable")
        
        self._registry[name] = {
            'class': model_class,
            'description': description,
            'task_type': task_type,
            'framework': framework,
            'metadata': kwargs
        }
    
    def get(self, name: str) -> Type:
        """
        Get a model class from the registry.
        
        Args:
            name: Model name
            
        Returns:
            Model class
        """
        if name not in self._registry:
            raise ModelNotFoundError(f"Model '{name}' not found in registry")
        
        return self._registry[name]['class']
    
    def create(self, name: str, **kwargs) -> Any:
        """
        Create an instance of a registered model.
        
        Args:
            name: Model name
            **kwargs: Arguments to pass to model constructor
            
        Returns:
            Model instance
        """
        model_class = self.get(name)
        return model_class(**kwargs)
    
    def get_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a registered model.
        
        Args:
            name: Model name
            
        Returns:
            Model information dictionary
        """
        if name not in self._registry:
            raise ModelNotFoundError(f"Model '{name}' not found in registry")
        
        return self._registry[name].copy()
    
    def list_by_task(self, task_type: str) -> List[str]:
        """
        List models by task type.
        
        Args:
            task_type: Task type to filter by
            
        Returns:
            List of model names
        """
        return [name for name, info in self._registry.items() 
                if info['task_type'] == task_type]
    
    def list_by_framework(self, framework: str) -> List[str]:
        """
        List models by framework.
        
        Args:
            framework: Framework to filter by
            
        Returns:
            List of model names
        """
        return [name for name, info in self._registry.items() 
                if info['framework'] == framework]


class DatasetRegistry(Registry):
    """
    Registry for datasets.
    """
    
    def register(self, name: str, loader_func: Callable,
                 description: str = "",
                 task_type: str = "general",
                 size: str = "medium",
                 **kwargs) -> None:
        """
        Register a dataset loader function.
        
        Args:
            name: Dataset name
            loader_func: Function to load the dataset
            description: Dataset description
            task_type: Type of ML task
            size: Dataset size (small, medium, large)
            **kwargs: Additional metadata
        """
        if not callable(loader_func):
            raise ValueError("loader_func must be callable")
        
        self._registry[name] = {
            'loader': loader_func,
            'description': description,
            'task_type': task_type,
            'size': size,
            'metadata': kwargs
        }
    
    def get(self, name: str) -> Callable:
        """
        Get a dataset loader function.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset loader function
        """
        if name not in self._registry:
            raise KeyError(f"Dataset '{name}' not found in registry")
        
        return self._registry[name]['loader']
    
    def load(self, name: str, **kwargs) -> Any:
        """
        Load a dataset.
        
        Args:
            name: Dataset name
            **kwargs: Arguments to pass to loader function
            
        Returns:
            Loaded dataset
        """
        loader = self.get(name)
        return loader(**kwargs)


class TransformerRegistry(Registry):
    """
    Registry for data transformers.
    """
    
    def register(self, name: str, transformer_class: Type,
                 description: str = "",
                 input_type: str = "general",
                 output_type: str = "general",
                 **kwargs) -> None:
        """
        Register a transformer class.
        
        Args:
            name: Transformer name
            transformer_class: Transformer class
            description: Transformer description
            input_type: Input data type
            output_type: Output data type
            **kwargs: Additional metadata
        """
        if not inspect.isclass(transformer_class):
            raise ValueError("transformer_class must be a class")
        
        self._registry[name] = {
            'class': transformer_class,
            'description': description,
            'input_type': input_type,
            'output_type': output_type,
            'metadata': kwargs
        }
    
    def get(self, name: str) -> Type:
        """Get a transformer class from the registry."""
        if name not in self._registry:
            raise KeyError(f"Transformer '{name}' not found in registry")
        
        return self._registry[name]['class']
    
    def create(self, name: str, **kwargs) -> Any:
        """Create an instance of a registered transformer."""
        transformer_class = self.get(name)
        return transformer_class(**kwargs)


# Global registry instances
model_registry = ModelRegistry()
dataset_registry = DatasetRegistry()
transformer_registry = TransformerRegistry()


# Decorator for registering models
def register_model(name: str, task_type: str = "general", framework: str = "sklearn", **kwargs):
    """
    Decorator to register a model class.
    
    Args:
        name: Model name
        task_type: Type of ML task
        framework: ML framework
        **kwargs: Additional metadata
    """
    def decorator(model_class):
        model_registry.register(name, model_class, task_type=task_type, framework=framework, **kwargs)
        return model_class
    return decorator


# Decorator for registering datasets
def register_dataset(name: str, task_type: str = "general", size: str = "medium", **kwargs):
    """
    Decorator to register a dataset loader function.
    
    Args:
        name: Dataset name
        task_type: Type of ML task
        size: Dataset size
        **kwargs: Additional metadata
    """
    def decorator(loader_func):
        dataset_registry.register(name, loader_func, task_type=task_type, size=size, **kwargs)
        return loader_func
    return decorator


# Decorator for registering transformers
def register_transformer(name: str, input_type: str = "general", output_type: str = "general", **kwargs):
    """
    Decorator to register a transformer class.
    
    Args:
        name: Transformer name
        input_type: Input data type
        output_type: Output data type
        **kwargs: Additional metadata
    """
    def decorator(transformer_class):
        transformer_registry.register(name, transformer_class, 
                                    input_type=input_type, output_type=output_type, **kwargs)
        return transformer_class
    return decorator
