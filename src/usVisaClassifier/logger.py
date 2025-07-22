import logging
import os
from datetime import datetime

def from_root():
    """
    Returns the absolute path to the project root directory.
    Assumes this file is located at src/usVisaClassifier/logger.py
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Generate timestamped log filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create log directory inside project root
log_dir = os.path.join(from_root(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Path to the log file
logs_path = os.path.join(log_dir, LOG_FILE)

# Create custom logger
logger = logging.getLogger("us_visa_logger")
logger.setLevel(logging.INFO)  # Only INFO, WARNING, ERROR, CRITICAL

# File handler (writes to file)
file_handler = logging.FileHandler(logs_path)
file_handler.setLevel(logging.INFO)

# Optional: console output (during development or testing)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Define common formatter
formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Attach handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
