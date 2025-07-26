
import sys
from src.usVisaClassifier.components.data_ingestion import DataIngestion

from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.entity.config_entity import (DataIngestionConfig)


from src.usVisaClassifier.entity.artifact_entity import (DataIngestionArtifact)


STAGE_NAME = "Data Ingestion stage"

# To trigger and manage the data ingestion process...............................
class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            # Configuration class that defines file paths, collection names, split ratios, etc.
            data_ingestion_config = DataIngestionConfig()
            
            # Class that handles actual data loading, feature store export, and train/test splitting
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            
            # Structured output containing paths to the saved train and test files
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e




# if __name__ == '__main__':
#     try:
#         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         obj = DataIngestionTrainingPipeline()
#         obj.main()
#         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#     except Exception as e:
#         logging.exception(e)
#         raise e