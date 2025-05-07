import os
import sys
import logging
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue, Lock
import atexit

# Global queue for logging
log_queue = Queue()
queue_listener = None
log_lock = Lock()

def setup_logger(log_dir: str = "logs", experiment_name: str = None) -> logging.Logger:
    """Set up logging configuration with shared file and console handlers"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('LlamaProcessor')
    logger.setLevel(logging.DEBUG)
    
    # If handlers already exist, return the logger
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create single log file for the experiment
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'processing_{timestamp}.log'
    if experiment_name:
        filename = f'{experiment_name}_{filename}'
    
    file_handler = logging.FileHandler(
        os.path.join(log_dir, filename),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Set up queue listener
    global queue_listener
    queue_listener = QueueListener(log_queue, file_handler, console_handler)
    queue_listener.start()
    
    # Add queue handler to logger
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    
    # Register cleanup function
    atexit.register(cleanup_logging)
    
    return logger

def cleanup_logging():
    """Clean up logging resources"""
    global queue_listener
    if queue_listener:
        queue_listener.stop()
        queue_listener = None

def get_process_logger() -> logging.Logger:
    """Get logger for child processes"""
    logger = logging.getLogger('LlamaProcessor')
    
    # If the logger doesn't have handlers (new process), add queue handler
    if not logger.handlers:
        queue_handler = QueueHandler(log_queue)
        logger.addHandler(queue_handler)
    
    return logger