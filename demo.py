from src.usVisaClassifier.logger import logger
from src.usVisaClassifier.exception import USvisaException
import sys

try:
    x = 1 / 0
except Exception as e:
    logger.info("...............................................")
    raise USvisaException(e, sys)
