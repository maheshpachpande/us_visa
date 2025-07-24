
import sys
from src.usVisaClassifier.components.data_validation import DataValidation

from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.entity.config_entity import (DataIngestionConfig,
                                                       DataValidationConfig)


from src.usVisaClassifier.entity.artifact_entity import (DataValidationArtifact)


STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            data_validation_config = DataValidationConfig()
            data_ingestion_cnf = DataIngestionConfig()
            validator = DataValidation(data_validation_config, data_ingestion_cnf)
            validation_artifact = validator.initiate_data_validation()
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return validation_artifact
        except Exception as e:
            raise USvisaException(e, sys)




# if __name__ == '__main__':
#     try:
#         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         obj = DataIngestionTrainingPipeline()
#         obj.main()
#         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#     except Exception as e:
#         logging.exception(e)
#         raise e