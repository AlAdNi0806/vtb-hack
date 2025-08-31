"""
Logging utility for the Conversational AI Backend
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config import settings


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger with consistent formatting"""
    
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file or settings.LOG_FILE:
        file_path = log_file or settings.LOG_FILE
        
        # Create logs directory if it doesn't exist
        log_dir = Path(file_path).parent
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
