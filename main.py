
from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.pipeline.stage_01_data_ingestion_pipeline import DataIngestionTrainingPipeline



STAGE_NAME = "Data Ingestion stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e