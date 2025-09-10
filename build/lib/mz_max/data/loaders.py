"""
Data loading utilities for MZ Max

This module provides various data loading functions for different
data formats and sources.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any, List
from pathlib import Path
import requests
import zipfile
import tarfile
from sklearn import datasets
from ..core.exceptions import DataError
from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_dataset(name: str, **kwargs) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a dataset by name.
    
    Args:
        name: Dataset name
        **kwargs: Additional arguments for dataset loading
        
    Returns:
        Loaded dataset
    """
    # Built-in sklearn datasets
    sklearn_datasets = {
        'iris': datasets.load_iris,
        'diabetes': datasets.load_diabetes,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer,
        'digits': datasets.load_digits,
        'california_housing': datasets.fetch_california_housing,
        '20newsgroups': datasets.fetch_20newsgroups,
        'lfw_people': datasets.fetch_lfw_people,
        'olivetti_faces': datasets.fetch_olivetti_faces,
        'species_distributions': datasets.fetch_species_distributions,
        'covtype': datasets.fetch_covtype,
        'kddcup99': datasets.fetch_kddcup99,
        'rcv1': datasets.fetch_rcv1,
    }
    
    if name in sklearn_datasets:
        try:
            dataset = sklearn_datasets[name](**kwargs)
            if hasattr(dataset, 'data') and hasattr(dataset, 'target'):
                # Create DataFrame with features and target
                if hasattr(dataset, 'feature_names'):
                    feature_names = dataset.feature_names
                else:
                    feature_names = [f'feature_{i}' for i in range(dataset.data.shape[1])]
                
                df = pd.DataFrame(dataset.data, columns=feature_names)
                
                if hasattr(dataset, 'target_names') and len(dataset.target_names) > 0:
                    target_name = 'target'
                    if len(np.unique(dataset.target)) == len(dataset.target_names):
                        # Map numeric targets to names
                        df[target_name] = [dataset.target_names[i] for i in dataset.target]
                    else:
                        df[target_name] = dataset.target
                else:
                    df['target'] = dataset.target
                
                return df
            else:
                return dataset
        except Exception as e:
            raise DataError(f"Failed to load dataset '{name}': {str(e)}")
    
    # Try to load as file
    try:
        return load_data_file(name, **kwargs)
    except:
        pass
    
    # Try to download from common repositories
    try:
        return download_dataset(name, **kwargs)
    except:
        pass
    
    raise DataError(f"Dataset '{name}' not found")


def load_data_file(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load data from a file.
    
    Args:
        filepath: Path to data file
        **kwargs: Additional arguments for pandas read functions
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise DataError(f"File not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    try:
        if suffix == '.csv':
            return pd.read_csv(filepath, **kwargs)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(filepath, **kwargs)
        elif suffix == '.json':
            return pd.read_json(filepath, **kwargs)
        elif suffix == '.parquet':
            return pd.read_parquet(filepath, **kwargs)
        elif suffix == '.feather':
            return pd.read_feather(filepath, **kwargs)
        elif suffix in ['.h5', '.hdf5']:
            return pd.read_hdf(filepath, **kwargs)
        elif suffix == '.pkl':
            return pd.read_pickle(filepath, **kwargs)
        else:
            # Try to read as CSV by default
            return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        raise DataError(f"Failed to load file '{filepath}': {str(e)}")


def download_dataset(name: str, cache_dir: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Download a dataset from online repositories.
    
    Args:
        name: Dataset name
        cache_dir: Directory to cache downloaded files
        **kwargs: Additional arguments
        
    Returns:
        Downloaded dataset
    """
    if cache_dir is None:
        cache_dir = Path.home() / '.mz_max_cache' / 'datasets'
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Common dataset URLs
    dataset_urls = {
        'titanic': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
        'housing': 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv',
        'adult': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        'heart': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
        'mushroom': 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',
    }
    
    if name not in dataset_urls:
        raise DataError(f"Dataset '{name}' not available for download")
    
    url = dataset_urls[name]
    cache_file = cache_dir / f"{name}.csv"
    
    # Download if not cached
    if not cache_file.exists():
        logger.info(f"Downloading dataset '{name}' from {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Dataset cached to {cache_file}")
        except Exception as e:
            raise DataError(f"Failed to download dataset '{name}': {str(e)}")
    
    # Load cached file
    try:
        if name == 'adult':
            # Adult dataset has specific column names
            columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                      'marital-status', 'occupation', 'relationship', 'race', 'sex',
                      'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
            return pd.read_csv(cache_file, names=columns, skipinitialspace=True)
        elif name == 'heart':
            # Heart disease dataset column names
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            return pd.read_csv(cache_file, names=columns)
        elif name == 'mushroom':
            # Mushroom dataset column names
            columns = ['class', 'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
                      'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size',
                      'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                      'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring',
                      'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
                      'population', 'habitat']
            return pd.read_csv(cache_file, names=columns)
        else:
            return pd.read_csv(cache_file, **kwargs)
    except Exception as e:
        raise DataError(f"Failed to load cached dataset '{name}': {str(e)}")


class CSVLoader:
    """Enhanced CSV loader with automatic type inference and preprocessing."""
    
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
    
    def load(self, **kwargs) -> pd.DataFrame:
        """Load CSV with enhanced options."""
        # Default parameters for better parsing
        default_kwargs = {
            'encoding': 'utf-8',
            'low_memory': False,
            'na_values': ['', 'NA', 'N/A', 'NULL', 'null', 'None', 'none', '?', '-'],
        }
        default_kwargs.update(kwargs)
        
        try:
            df = pd.read_csv(self.filepath, **default_kwargs)
            
            # Automatic type inference
            df = self._infer_types(df)
            
            return df
        except Exception as e:
            raise DataError(f"Failed to load CSV '{self.filepath}': {str(e)}")
    
    def _infer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer and convert column types."""
        for col in df.columns:
            # Try to convert to numeric
            if df[col].dtype == 'object':
                # Check if it's a date
                if self._is_date_column(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col])
                        continue
                    except:
                        pass
                
                # Try numeric conversion
                try:
                    # Check if all non-null values can be converted to numeric
                    pd.to_numeric(df[col].dropna())
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        return df
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a series likely contains dates."""
        if series.dtype != 'object':
            return False
        
        # Sample a few non-null values
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        # Common date patterns
        date_indicators = [
            '/', '-', ':', 'T', 'Z',  # Common date separators
            '2019', '2020', '2021', '2022', '2023', '2024', '2025',  # Recent years
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',  # Month names
        ]
        
        sample_str = ' '.join(sample.astype(str))
        return any(indicator in sample_str for indicator in date_indicators)


class ImageLoader:
    """Image data loader."""
    
    def __init__(self, directory: Union[str, Path]):
        self.directory = Path(directory)
    
    def load(self, image_size: Tuple[int, int] = (224, 224), **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Load images from directory structure."""
        try:
            from PIL import Image
        except ImportError:
            raise DataError("PIL/Pillow is required for image loading")
        
        if not self.directory.exists():
            raise DataError(f"Directory not found: {self.directory}")
        
        images = []
        labels = []
        class_names = []
        
        # Assume directory structure: root/class1/, root/class2/, etc.
        for class_dir in self.directory.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_names.append(class_name)
                class_idx = len(class_names) - 1
                
                for img_file in class_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        try:
                            img = Image.open(img_file)
                            img = img.convert('RGB')
                            img = img.resize(image_size)
                            img_array = np.array(img) / 255.0  # Normalize
                            
                            images.append(img_array)
                            labels.append(class_idx)
                        except Exception as e:
                            logger.warning(f"Failed to load image {img_file}: {e}")
        
        if not images:
            raise DataError("No valid images found")
        
        return np.array(images), np.array(labels)


class TextLoader:
    """Text data loader."""
    
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
    
    def load(self, encoding: str = 'utf-8', **kwargs) -> List[str]:
        """Load text data."""
        if not self.filepath.exists():
            raise DataError(f"File not found: {self.filepath}")
        
        try:
            with open(self.filepath, 'r', encoding=encoding) as f:
                if self.filepath.suffix.lower() == '.txt':
                    return f.readlines()
                else:
                    return [f.read()]
        except Exception as e:
            raise DataError(f"Failed to load text file '{self.filepath}': {str(e)}")


class AudioLoader:
    """Audio data loader."""
    
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
    
    def load(self, sr: int = 22050, **kwargs) -> Tuple[np.ndarray, int]:
        """Load audio data."""
        try:
            import librosa
        except ImportError:
            raise DataError("librosa is required for audio loading")
        
        if not self.filepath.exists():
            raise DataError(f"File not found: {self.filepath}")
        
        try:
            audio, sample_rate = librosa.load(self.filepath, sr=sr, **kwargs)
            return audio, sample_rate
        except Exception as e:
            raise DataError(f"Failed to load audio file '{self.filepath}': {str(e)}")


# Convenience function
def DataLoader(source: Union[str, Path], loader_type: str = 'auto', **kwargs):
    """
    Create appropriate data loader based on source and type.
    
    Args:
        source: Data source (file path, directory, URL, etc.)
        loader_type: Type of loader ('auto', 'csv', 'image', 'text', 'audio')
        **kwargs: Additional arguments for the loader
        
    Returns:
        Appropriate loader instance
    """
    source = Path(source)
    
    if loader_type == 'auto':
        if source.is_file():
            suffix = source.suffix.lower()
            if suffix == '.csv':
                loader_type = 'csv'
            elif suffix in ['.txt', '.md']:
                loader_type = 'text'
            elif suffix in ['.wav', '.mp3', '.flac', '.m4a']:
                loader_type = 'audio'
            else:
                loader_type = 'csv'  # Default
        elif source.is_dir():
            loader_type = 'image'  # Assume image directory
        else:
            loader_type = 'csv'  # Default
    
    if loader_type == 'csv':
        return CSVLoader(source)
    elif loader_type == 'image':
        return ImageLoader(source)
    elif loader_type == 'text':
        return TextLoader(source)
    elif loader_type == 'audio':
        return AudioLoader(source)
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")
