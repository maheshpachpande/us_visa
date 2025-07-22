import logging
import os
from datetime import datetime

# Step 1: Create log directory and filename
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(log_dir, log_file)

# Step 2: Set up basicConfig for file logging
logging.basicConfig(
    filename=log_path,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Step 3: Create a StreamHandler for console (terminal) logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Step 4: Add console handler to root logger
logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console_handler)

