from src.usVisaClassifier.logger import logger
from src.usVisaClassifier.exception import USvisaException
import sys

import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv("MONGODB_URL_KEY"))
