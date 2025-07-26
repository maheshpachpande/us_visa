# ================================
# 📦 IMPORTS
# ================================

import sys
from src.usVisaClassifier.components.data_validation import DataValidation
from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging

from src.usVisaClassifier.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig
)

from src.usVisaClassifier.entity.artifact_entity import (
    DataValidationArtifact
)


# ================================
# 🔖 STAGE IDENTIFIER
# ================================

#  Defines a human-readable name for this pipeline stage
STAGE_NAME = "Data Validation stage"


# ================================
# 🧱 CLASS: DataValidationTrainingPipeline
# ================================

class DataValidationTrainingPipeline:
    """
    Class responsible for orchestrating the data validation stage.
    It initializes configurations, triggers validation, and returns the result artifact.
    """

    def __init__(self):
        """
         Placeholder for any future initialization logic.
        Currently, no specific state is set in __init__.
        """
        pass

    # ================================
    # 🚀 MAIN PIPELINE ENTRY POINT
    # ================================

    def main(self) -> DataValidationArtifact:
        """
        Triggers the data validation process:
        - Loads configs
        - Instantiates the DataValidation component
        - Executes validation and logs results
        - Returns the artifact with validation outcome
        """
        try:
            #  Load configuration entities for validation and ingestion
            data_validation_config = DataValidationConfig()
            data_ingestion_cnf = DataIngestionConfig()

            #  Create validation component and run it
            validator = DataValidation(data_validation_config, data_ingestion_cnf)
            validation_artifact = validator.initiate_data_validation()

            #  Log successful completion of the stage
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")

            return validation_artifact

        except Exception as e:
            #  Raise a custom exception with full system traceback
            raise USvisaException(e, sys)


# ================================
# 🏁 STANDALONE SCRIPT EXECUTION (OPTIONAL)
# ================================

#  This block allows the file to be executed directly for standalone runs or testing
# Uncomment the following to run without orchestrators like Airflow

if __name__ == '__main__':
    try:
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
