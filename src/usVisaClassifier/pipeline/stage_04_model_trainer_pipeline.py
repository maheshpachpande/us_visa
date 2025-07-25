
import sys
from src.usVisaClassifier.components.model_trainer import ModelTrainer

from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging

from src.usVisaClassifier.entity.config_entity import (DataTransformationConfig,
                                                       ModelTrainerConfig)


from src.usVisaClassifier.entity.artifact_entity import (DataTransformationArtifact)


STAGE_NAME = "MOdel trainer stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        This method of TrainPipeline class is responsible for starting data transformation component
        """
        try:
            trf_cnf = DataTransformationConfig()
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=trf_cnf.transformed_train_file_path,
                transformed_test_file_path=trf_cnf.transformed_test_file_path,
                transformed_object_file_path=trf_cnf.transformed_object_file_path
            )
            model_cnf = ModelTrainerConfig()        
            # Create and run ModelTrainer
            trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=model_cnf
            )

            model_trainer_artifact = trainer.initiate_model_trainer()
            # Log results
            print("✅ Model training completed.")
            print("Trained model saved at:", model_trainer_artifact.trained_model_file_path)
            print("F1 Score:", model_trainer_artifact.metric_artifact.f1_score)
            print("Precision:", model_trainer_artifact.metric_artifact.precision_score)
            print("Recall:", model_trainer_artifact.metric_artifact.recall_score)
        except Exception as e:
            raise USvisaException(e,sys)


# if __name__ == '__main__':
#     try:
#         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         obj = ModelTrainingPipeline()
#         obj.main()
#         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#     except Exception as e:
#         logging.exception(e)
#         raise e





    