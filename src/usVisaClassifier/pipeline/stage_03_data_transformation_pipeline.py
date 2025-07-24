
import sys
from src.usVisaClassifier.components.data_transformation import DataTransformation

from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging

from src.usVisaClassifier.entity.config_entity import (DataIngestionConfig,
                                                       DataValidationConfig,
                                                       DataTransformationConfig)


from src.usVisaClassifier.entity.artifact_entity import (DataIngestionArtifact,
                                                         DataValidationArtifact)


STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        This method of TrainPipeline class is responsible for starting data transformation component
        """
        try:
            data_ingetion_cnf = DataIngestionConfig()
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=data_ingetion_cnf.training_file_path,
                test_file_path=data_ingetion_cnf.testing_file_path
            )
            
            data_val_cnf = DataValidationConfig()
            
            status = DataValidationArtifact.load(file_path="artifact/data_validation/data_validation_artifact.yaml")
            val_stat = status.validation_status
            message = status.message

            trf = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                    data_transformation_config=DataTransformationConfig(),
                                    data_validation_artifact = DataValidationArtifact(
                                        validation_status=val_stat,
                                        drift_report_file_path=data_val_cnf.drift_report_file_path,
                                        data_validation_report=data_val_cnf.data_validation_report,
                                        message="Validation successful"))
                                    
            trf.initiate_data_transformation()
                
            
        except Exception as e:
            raise USvisaException(e, sys)



# if __name__ == '__main__':
#     try:
#         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         obj = DataTransformationTrainingPipeline()
#         obj.main()
#         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#     except Exception as e:
#         logging.exception(e)
#         raise e