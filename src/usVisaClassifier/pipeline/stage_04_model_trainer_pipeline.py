# ==========================================
# 📦 MODEL TRAINING PIPELINE SCRIPT
# ==========================================

import sys
from src.usVisaClassifier.components.model_trainer import ModelTrainer
from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.entity.config_entity import (
    DataTransformationConfig,
    ModelTrainerConfig
)
from src.usVisaClassifier.entity.artifact_entity import (
    DataTransformationArtifact
)


STAGE_NAME = "Model Trainer Stage"


class ModelTrainingPipeline:
    """
    Orchestrates the model training step of the US Visa classification pipeline.
    This includes loading transformed data and training multiple models with hyperparameter tuning.
    """

    def __init__(self):
        pass

    def main(self):
        try:
            # Load configuration for transformed data paths
            trf_config = DataTransformationConfig()
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=trf_config.transformed_train_file_path,
                transformed_test_file_path=trf_config.transformed_test_file_path,
                transformed_object_file_path=trf_config.transformed_object_file_path
            )

            # Load model trainer config
            model_config = ModelTrainerConfig()

            # Initialize and run training
            trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=model_config
            )

            model_trainer_artifact = trainer.initiate_model_trainer()

            # Log results
            print("✅ Model training completed.")
            print("📍 Trained model saved at:", model_trainer_artifact.trained_model_file_path)
            print("📊 F1 Score:", model_trainer_artifact.metric_artifact.f1_score)
            print("📊 Precision:", model_trainer_artifact.metric_artifact.precision_score)
            print("📊 Recall:", model_trainer_artifact.metric_artifact.recall_score)

        except Exception as e:
            raise USvisaException(e, sys)


# ==========================================
# 🚀 ENTRY POINT
# ==========================================
if __name__ == '__main__':
    try:
        logging.info(f">>>>>> STAGE: {STAGE_NAME} started <<<<<<")
        pipeline = ModelTrainingPipeline()
        pipeline.main()
        logging.info(f">>>>>> STAGE: {STAGE_NAME} completed <<<<<<\n\nx==========x\n")
    except Exception as e:
        logging.exception(f"❌ Exception occurred in {STAGE_NAME}: {e}")
        raise e
