import sys
from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.exception import USvisaException

from src.usVisaClassifier.pipeline.stage_01_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.usVisaClassifier.pipeline.stage_02_data_validation_pipeline import DataValidationTrainingPipeline
from src.usVisaClassifier.pipeline.stage_03_data_transformation_pipeline import DataTransformationTrainingPipeline
from src.usVisaClassifier.pipeline.stage_04_model_trainer_pipeline import ModelTrainingPipeline
from src.usVisaClassifier.pipeline.stage_05_model_evaluation_pipeline import ModelEvaluationPipeline
from src.usVisaClassifier.pipeline.stage_06_model_pusher_pipeline import ModelPusherPipeline

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


STAGE_NAME = "Data Transformation stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "MOdel trainer stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Model Evaluation Stage"

try:
    logging.info(f">>>>>> STAGE: {STAGE_NAME} started <<<<<<")
    pipeline = ModelEvaluationPipeline()
    pipeline.main()
    logging.info(f">>>>>> STAGE: {STAGE_NAME} completed <<<<<<\n\nx==========x\n")
except Exception as e:
    logging.exception(f"❌ Exception occurred in {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Model Pusher Stage"

try:
    logging.info(f">>>>>> STAGE: {STAGE_NAME} started <<<<<<")
    pipeline = ModelPusherPipeline()
    pipeline.main()
    logging.info(f">>>>>> STAGE: {STAGE_NAME} completed <<<<<<\n\nx==========x\n")
except Exception as e:
    logging.exception(f"❌ Exception occurred in {STAGE_NAME}: {e}")
    raise e
