import sys
from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.exception import USvisaException

from src.usVisaClassifier.pipeline.stage_01_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.usVisaClassifier.pipeline.stage_02_data_validation_pipeline import DataValidationTrainingPipeline




STAGE_NAME = "Data Ingestion stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
except Exception as e:
    raise USvisaException(e, sys)


STAGE_NAME = "Data Validation stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    
except Exception as e:
    raise USvisaException(e, sys)