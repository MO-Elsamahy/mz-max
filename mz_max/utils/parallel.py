"""
Parallel processing utilities for MZ Max

This module provides utilities for parallel and distributed computing.
"""

import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Union, Iterator, Dict
from functools import partial
import time

from .logging import get_logger

logger = get_logger(__name__)


def get_optimal_n_jobs(n_jobs: Optional[int] = None) -> int:
    """
    Get optimal number of jobs for parallel processing.
    
    Args:
        n_jobs: Number of jobs (-1 for all cores, None for auto)
        
    Returns:
        Optimal number of jobs
    """
    if n_jobs is None:
        # Auto-detect based on system
        cpu_count = mp.cpu_count()
        if cpu_count <= 2:
            return 1
        elif cpu_count <= 4:
            return cpu_count - 1
        else:
            return max(2, cpu_count - 2)  # Leave some cores free
    elif n_jobs == -1:
        return mp.cpu_count()
    else:
        return max(1, min(n_jobs, mp.cpu_count()))


def parallel_map(func: Callable, iterable: List[Any], 
                n_jobs: int = -1, backend: str = 'thread',
                chunk_size: Optional[int] = None,
                timeout: Optional[float] = None) -> List[Any]:
    """
    Apply function to iterable in parallel.
    
    Args:
        func: Function to apply
        iterable: Items to process
        n_jobs: Number of parallel jobs
        backend: 'thread' or 'process'
        chunk_size: Chunk size for processing
        timeout: Timeout in seconds
        
    Returns:
        List of results
    """
    n_jobs = get_optimal_n_jobs(n_jobs)
    
    if n_jobs == 1 or len(iterable) == 1:
        # Sequential processing
        return [func(item) for item in iterable]
    
    if chunk_size is None:
        chunk_size = max(1, len(iterable) // (n_jobs * 4))
    
    logger.info(f"Parallel processing {len(iterable)} items with {n_jobs} {backend} workers")
    
    start_time = time.time()
    
    try:
        if backend == 'thread':
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(func, iterable, chunksize=chunk_size, timeout=timeout))
        elif backend == 'process':
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(func, iterable, chunksize=chunk_size, timeout=timeout))
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Parallel processing completed in {elapsed_time:.2f} seconds")
        
        return results
    
    except Exception as e:
        logger.error(f"Parallel processing failed: {str(e)}")
        raise


def parallel_apply(func: Callable, items: List[Any], 
                  n_jobs: int = -1, backend: str = 'thread',
                  progress_callback: Optional[Callable] = None) -> List[Any]:
    """
    Apply function to items with progress tracking.
    
    Args:
        func: Function to apply
        items: Items to process
        n_jobs: Number of parallel jobs
        backend: 'thread' or 'process'
        progress_callback: Callback for progress updates
        
    Returns:
        List of results
    """
    n_jobs = get_optimal_n_jobs(n_jobs)
    
    if n_jobs == 1:
        results = []
        for i, item in enumerate(items):
            result = func(item)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(items))
        return results
    
    logger.info(f"Processing {len(items)} items with {n_jobs} {backend} workers")
    
    results = [None] * len(items)
    completed = 0
    
    executor_class = ThreadPoolExecutor if backend == 'thread' else ProcessPoolExecutor
    
    with executor_class(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(func, item): i 
            for i, item in enumerate(items)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(items))
                    
            except Exception as e:
                logger.error(f"Task {index} failed: {str(e)}")
                results[index] = None
    
    return results


def batch_process(items: List[Any], batch_size: int, 
                 process_func: Callable, n_jobs: int = -1,
                 backend: str = 'thread') -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: Items to process
        batch_size: Size of each batch
        process_func: Function to process each batch
        n_jobs: Number of parallel jobs
        backend: 'thread' or 'process'
        
    Returns:
        List of batch results
    """
    # Create batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    logger.info(f"Processing {len(items)} items in {len(batches)} batches of size {batch_size}")
    
    # Process batches in parallel
    return parallel_map(process_func, batches, n_jobs=n_jobs, backend=backend)


class ParallelProcessor:
    """
    Configurable parallel processor.
    """
    
    def __init__(self, n_jobs: int = -1, backend: str = 'thread',
                 timeout: Optional[float] = None):
        """
        Initialize parallel processor.
        
        Args:
            n_jobs: Number of parallel jobs
            backend: 'thread' or 'process'
            timeout: Timeout for operations
        """
        self.n_jobs = get_optimal_n_jobs(n_jobs)
        self.backend = backend
        self.timeout = timeout
        self.executor = None
    
    def __enter__(self):
        """Start executor."""
        executor_class = ThreadPoolExecutor if self.backend == 'thread' else ProcessPoolExecutor
        self.executor = executor_class(max_workers=self.n_jobs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop executor."""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def map(self, func: Callable, iterable: List[Any]) -> List[Any]:
        """
        Map function over iterable.
        
        Args:
            func: Function to apply
            iterable: Items to process
            
        Returns:
            List of results
        """
        if not self.executor:
            raise RuntimeError("Processor not initialized. Use as context manager.")
        
        return list(self.executor.map(func, iterable, timeout=self.timeout))
    
    def submit(self, func: Callable, *args, **kwargs):
        """
        Submit single task.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Future object
        """
        if not self.executor:
            raise RuntimeError("Processor not initialized. Use as context manager.")
        
        return self.executor.submit(func, *args, **kwargs)


def parallel_reduce(func: Callable, iterable: List[Any], 
                   reduce_func: Callable, n_jobs: int = -1,
                   backend: str = 'thread') -> Any:
    """
    Map-reduce pattern with parallel map phase.
    
    Args:
        func: Function to apply to each item
        iterable: Items to process
        reduce_func: Function to reduce results
        n_jobs: Number of parallel jobs
        backend: 'thread' or 'process'
        
    Returns:
        Reduced result
    """
    # Parallel map phase
    mapped_results = parallel_map(func, iterable, n_jobs=n_jobs, backend=backend)
    
    # Sequential reduce phase
    result = mapped_results[0]
    for item in mapped_results[1:]:
        result = reduce_func(result, item)
    
    return result


def chunked_parallel_map(func: Callable, iterable: List[Any],
                        chunk_size: int, n_jobs: int = -1,
                        backend: str = 'thread') -> List[Any]:
    """
    Process large iterables in chunks with parallel processing.
    
    Args:
        func: Function to apply
        iterable: Items to process
        chunk_size: Size of each chunk
        n_jobs: Number of parallel jobs
        backend: 'thread' or 'process'
        
    Returns:
        List of results
    """
    # Create chunks
    chunks = [iterable[i:i + chunk_size] for i in range(0, len(iterable), chunk_size)]
    
    # Define chunk processor
    def process_chunk(chunk):
        return [func(item) for item in chunk]
    
    # Process chunks in parallel
    chunk_results = parallel_map(process_chunk, chunks, n_jobs=n_jobs, backend=backend)
    
    # Flatten results
    results = []
    for chunk_result in chunk_results:
        results.extend(chunk_result)
    
    return results


class AsyncBatchProcessor:
    """
    Asynchronous batch processor for handling large datasets.
    """
    
    def __init__(self, batch_size: int = 100, n_jobs: int = -1,
                 backend: str = 'thread', max_queue_size: int = 1000):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Size of each batch
            n_jobs: Number of parallel workers
            backend: 'thread' or 'process'
            max_queue_size: Maximum queue size
        """
        self.batch_size = batch_size
        self.n_jobs = get_optimal_n_jobs(n_jobs)
        self.backend = backend
        self.max_queue_size = max_queue_size
        self.results = []
    
    def process_stream(self, data_stream: Iterator[Any], 
                      process_func: Callable) -> List[Any]:
        """
        Process streaming data in batches.
        
        Args:
            data_stream: Iterator of data items
            process_func: Function to process each batch
            
        Returns:
            List of all results
        """
        import queue
        import threading
        
        # Create queues
        input_queue = queue.Queue(maxsize=self.max_queue_size)
        output_queue = queue.Queue()
        
        # Producer thread to read stream
        def producer():
            batch = []
            for item in data_stream:
                batch.append(item)
                if len(batch) >= self.batch_size:
                    input_queue.put(batch)
                    batch = []
            
            # Put remaining items
            if batch:
                input_queue.put(batch)
            
            # Signal end
            input_queue.put(None)
        
        # Consumer threads to process batches
        def consumer():
            while True:
                batch = input_queue.get()
                if batch is None:
                    input_queue.task_done()
                    break
                
                try:
                    result = process_func(batch)
                    output_queue.put(result)
                except Exception as e:
                    logger.error(f"Batch processing failed: {str(e)}")
                    output_queue.put(None)
                
                input_queue.task_done()
        
        # Start threads
        producer_thread = threading.Thread(target=producer)
        consumer_threads = [threading.Thread(target=consumer) for _ in range(self.n_jobs)]
        
        producer_thread.start()
        for thread in consumer_threads:
            thread.start()
        
        # Collect results
        results = []
        batches_processed = 0
        
        producer_thread.join()
        
        # Wait for all batches to be processed
        input_queue.join()
        
        # Signal consumers to stop
        for _ in consumer_threads:
            input_queue.put(None)
        
        # Wait for consumers to finish
        for thread in consumer_threads:
            thread.join()
        
        # Collect all results
        while not output_queue.empty():
            result = output_queue.get()
            if result is not None:
                results.extend(result if isinstance(result, list) else [result])
                batches_processed += 1
        
        logger.info(f"Processed {batches_processed} batches")
        return results


def parallel_grid_search(param_grid: Dict[str, List[Any]], 
                        evaluate_func: Callable,
                        n_jobs: int = -1, backend: str = 'process') -> Dict[str, Any]:
    """
    Parallel grid search over parameter combinations.
    
    Args:
        param_grid: Dictionary of parameter names and values
        evaluate_func: Function to evaluate each parameter combination
        n_jobs: Number of parallel jobs
        backend: 'thread' or 'process'
        
    Returns:
        Dictionary with best parameters and score
    """
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    logger.info(f"Evaluating {len(param_combinations)} parameter combinations")
    
    # Create parameter dictionaries
    param_dicts = [
        dict(zip(param_names, combo)) 
        for combo in param_combinations
    ]
    
    # Evaluate in parallel
    scores = parallel_map(evaluate_func, param_dicts, n_jobs=n_jobs, backend=backend)
    
    # Find best combination
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_params = param_dicts[best_idx]
    best_score = scores[best_idx]
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_params': param_dicts,
        'all_scores': scores
    }


def set_multiprocessing_start_method(method: str = 'spawn'):
    """
    Set multiprocessing start method.
    
    Args:
        method: Start method ('spawn', 'fork', 'forkserver')
    """
    try:
        mp.set_start_method(method, force=True)
        logger.info(f"Multiprocessing start method set to: {method}")
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing start method: {str(e)}")


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for parallel processing.
    
    Returns:
        System information dictionary
    """
    return {
        'cpu_count': mp.cpu_count(),
        'start_method': mp.get_start_method(),
        'platform': os.name,
        'available_methods': mp.get_all_start_methods()
    }
