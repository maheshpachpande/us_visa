# ===============================================================
# 📦 IMPORTS & CONFIGURATION
# ===============================================================

import sys
import pandas as pd
import numpy as np
from src.usVisaClassifier.utils import write_yaml_file
import os

from dataclasses import dataclass
from typing import Optional, Union

from sklearn.metrics import f1_score

from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.exception import USvisaException
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
    ModelEvaluationArtifact,
    ClassificationMetricArtifact
)

from src.usVisaClassifier.entity.s3_estimator import USvisaEstimator
from src.usVisaClassifier.entity.estimator import TargetValueMapping


# ===============================================================
# 📊 Dataclass: Evaluation Result Structure
# ===============================================================
@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Union[float, np.ndarray]
    is_model_accepted: bool
    difference: float


# ===============================================================
# 🚀 CLASS: ModelEvaluation
# Purpose: Compare the newly trained model with the previous best model (from S3)
# ===============================================================
class ModelEvaluation:

    def __init__(self,
                 model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        self.model_eval_config = model_eval_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_artifact = model_trainer_artifact

    def get_best_model(self) -> Optional[USvisaEstimator]:
        """
        Fetch the current production model from S3 if it exists.
        """
        try:
            usvisa_estimator = USvisaEstimator(
                bucket_name=self.model_eval_config.bucket_name,
                model_path=self.model_eval_config.s3_model_key_path
            )

            if usvisa_estimator.is_model_present(model_path=self.model_eval_config.s3_model_key_path):
                return usvisa_estimator
            return None
        except Exception as e:
            raise USvisaException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Compare trained model with the best model from S3 using F1 score.
        """
        try:
            cnf = read_yaml_file(file_path="config.yaml")
            target_column = cnf['target']['column']
            current_year = cnf['target']['current_year']

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df["company_age"] = current_year - test_df["yr_of_estab"]

            x = test_df.drop(target_column, axis=1)
            y = test_df[target_column].replace(TargetValueMapping()._asdict()).infer_objects(copy=False)

            trained_model_f1_score = float(self.model_trainer_artifact.metric_artifact.f1_score)

            best_model_f1_score = 0.0
            best_model = self.get_best_model()

            if best_model is not None:
                y_hat = best_model.predict(x)
                if y_hat is not None:
                    best_model_f1_score = float(f1_score(y, y_hat))
                else:
                    logging.warning("⚠️ Best model prediction returned None.")

            is_accepted = bool(np.any(trained_model_f1_score > best_model_f1_score))

            return EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_accepted,
                difference=float(trained_model_f1_score - best_model_f1_score
            ))

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiates model evaluation and returns a structured artifact.
        """
        try:
            evaluation_result = self.evaluate_model()
            
            result = ModelEvaluationArtifact(
                is_model_accepted=evaluation_result.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluation_result.difference
            )
            # 🔽 Save evaluation artifact to YAML file for next pipeline stage
            evaluation_artifact_path = os.path.join("artifact", "model_evaluation", "model_evaluation_artifact.yaml")
            os.makedirs(os.path.dirname(evaluation_artifact_path), exist_ok=True)

            # Convert to dict and save
            write_yaml_file(file_path=evaluation_artifact_path, content=result.__dict__)


            return result

        except Exception as e:
            raise USvisaException(e, sys)


# ===============================================================
# 🔰 ENTRY POINT
# ===============================================================
if __name__ == "__main__":
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
