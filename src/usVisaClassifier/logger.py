import logging
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root  # Custom utility to get project root
from datetime import datetime

# -------------------- Configuration Constants --------------------

LOG_DIR = "logs"  # Folder where logs will be saved
LOG_FILENAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # Timestamped log file name
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB max per file
BACKUP_COUNT = 3  # Keep up to 3 old log files

# -------------------- Construct Full Log File Path --------------------

log_directory = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_directory, exist_ok=True)  # Create log directory if not exists

log_file_path = os.path.join(log_directory, LOG_FILENAME)

# -------------------- Logging Configuration Function --------------------

def configure_logger() -> None:
    """
    Sets up a logger that:
    - Logs debug and above messages to a rotating file
    - Logs info and above messages to the console
    - Uses a consistent, readable format
    """
    logger = logging.getLogger()  # Root logger
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Prevent duplicate handlers if configure_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Format for logs
    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    # File handler with rotation
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Attach both handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# -------------------- Initialize Logging Immediately --------------------

configure_logger()
