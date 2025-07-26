# ==========================================
# 📦 MODEL TRAINING PIPELINE SCRIPT
# ==========================================

import sys
from src.usVisaClassifier.components.model_evaluation import ModelEvaluation
from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.utils import read_yaml_file
from src.usVisaClassifier.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from src.usVisaClassifier.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)


STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:
    """
    Orchestrates the model training step of the US Visa classification pipeline.
    This includes loading transformed data and training multiple models with hyperparameter tuning.
    """

    def __init__(self):
        pass

    def main(self):
        try:
            logging.info("🔍 Starting model evaluation...")

            # Load Configurations
            data_ingestion_config = DataIngestionConfig()
            data_transformation_config = DataTransformationConfig()
            model_trainer_config = ModelTrainerConfig()
            model_evaluation_config = ModelEvaluationConfig()

            # Load Artifacts
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=data_ingestion_config.training_file_path,
                test_file_path=data_ingestion_config.testing_file_path
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=data_transformation_config.transformed_object_file_path
            )

            metric_data = read_yaml_file("artifact/model_trainer/metric_artifact.yaml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=model_trainer_config.trained_model_file_path,
                metric_artifact=ClassificationMetricArtifact(
                    f1_score=metric_data["f1_score"],
                    precision_score=metric_data["precision_score"],
                    recall_score=metric_data["recall_score"]
                )
            )

            # Run Evaluation
            evaluator = ModelEvaluation(
                model_eval_config=model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )

            result = evaluator.initiate_model_evaluation()

            logging.info(f"✅ Evaluation Completed. Model Accepted: {result.is_model_accepted}")
            print("\n📊 Evaluation Result:")
            print(result)

        except Exception as e:
            logging.exception("❌ Error during model evaluation")
            print(f"Error: {e}")



# ==========================================
# 🚀 ENTRY POINT
# ==========================================
if __name__ == '__main__':
    try:
        logging.info(f">>>>>> STAGE: {STAGE_NAME} started <<<<<<")
        pipeline = ModelEvaluationPipeline()
        pipeline.main()
        logging.info(f">>>>>> STAGE: {STAGE_NAME} completed <<<<<<\n\nx==========x\n")
    except Exception as e:
        logging.exception(f"❌ Exception occurred in {STAGE_NAME}: {e}")
        raise e
