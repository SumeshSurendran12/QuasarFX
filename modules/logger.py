"""
Logger module for the Forex Trading Bot
"""

import logging
import os
try:
    from .config import LOG_FILE_PATH, LOGS_DIRECTORY
except ImportError:  # pragma: no cover - script mode fallback
    from config import LOG_FILE_PATH, LOGS_DIRECTORY

def setup_logger():
    
       # Ensure logs directory exists
    os.makedirs(LOGS_DIRECTORY, exist_ok=True)
    """
    Configure and return a logger for the application
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(LOGS_DIRECTORY):
        os.makedirs(LOGS_DIRECTORY)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("forex_trading_bot")
    return logger

# Create a global logger instance
logger = setup_logger()

# Configure logging
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler) 
