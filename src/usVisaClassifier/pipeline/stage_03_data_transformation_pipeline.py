# =============================================
# 📦 IMPORTS & SETUP
# =============================================

import sys
from src.usVisaClassifier.components.data_transformation import DataTransformation
from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging

from src.usVisaClassifier.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig
)

from src.usVisaClassifier.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)

# =============================================
# 🚀 STAGE NAME
# =============================================

STAGE_NAME = "Data Transformation stage"


# =============================================
# 🧠 CLASS: DataTransformationTrainingPipeline
# =============================================

class DataTransformationTrainingPipeline:
    def __init__(self):
        """
        Initializes the transformation training pipeline.
        Currently no internal setup needed.
        """
        pass

    def main(self):
        """
        Executes the data transformation stage:
        - Loads config and validation artifacts
        - Initializes DataTransformation class
        - Triggers data transformation pipeline
        """
        try:
            # =============================================
            # 🔧 Load configuration and artifacts
            # =============================================
            data_ingestion_config = DataIngestionConfig()
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=data_ingestion_config.training_file_path,
                test_file_path=data_ingestion_config.testing_file_path
            )

            data_validation_config = DataValidationConfig()

            # =============================================
            # 📄 Load validation artifact from previous stage
            # =============================================
            validation_artifact = DataValidationArtifact.load(
                file_path="artifact/data_validation/data_validation_artifact.yaml"
            )

            # =============================================
            # 🔁 Instantiate and run DataTransformation
            # =============================================
            transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=DataTransformationConfig(),
                data_validation_artifact=DataValidationArtifact(
                    validation_status=validation_artifact.validation_status,
                    drift_report_file_path=data_validation_config.drift_report_file_path,
                    data_validation_report=data_validation_config.data_validation_report,
                    message=validation_artifact.message
                )
            )

            transformation.initiate_data_transformation()

        except Exception as e:
            raise USvisaException(e, sys)


# =============================================
# 🚦 SCRIPT EXECUTION ENTRYPOINT
# =============================================

if __name__ == '__main__':
    try:
        logging.info(f"\n\n🚀>>>>>> Stage [{STAGE_NAME}] started <<<<<<🚀")
        pipeline = DataTransformationTrainingPipeline()
        pipeline.main()
        logging.info(f"✅>>>>>> Stage [{STAGE_NAME}] completed <<<<<<✅\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
