# =============================================
# 📦 IMPORTS & SETUP
# =============================================

import sys
import numpy as np
import yaml
from typing import Tuple, Any, Dict
from importlib import import_module
from dataclasses import dataclass

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.utils import (load_numpy_array_data,
                                        read_yaml_file,
                                        load_object,
                                        save_object)

from src.usVisaClassifier.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from src.usVisaClassifier.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact


# =============================================
# 📊 DATACLASS: Holds best model details after training and tuning
# =============================================

@dataclass
class BestModelDetail:
    best_model: Any                     # Trained model object
    best_score: float                  # F1 score of best model
    best_params: Dict[str, Any]        # Best hyperparameters found by GridSearchCV


# ======================================================================================================
# 🚀 CLASS: ModelTrainer - Responsible for selecting, training, and saving the best-performing model
# ======================================================================================================

class ModelTrainer:
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        Initializes ModelTrainer with:
        - data_transformation_artifact: contains paths to transformed train/test arrays
        - model_trainer_config: contains paths to YAML, expected accuracy, model save path
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    
    # =============================================
    # 🧠 Load all models and their hyperparameters from YAML
    # =============================================
    
    def _load_models_from_yaml(self, yaml_path: str) -> Dict[str, Tuple[BaseEstimator, dict]]:
        """
        Reads model definitions and their param grids from YAML file.
        Returns dictionary like:
        {
            "RandomForest": (RandomForestClassifier(), {"n_estimators": [100, 200]}),
            ...
        }
        """
        try:
            logging.info(f"🔍 Loading models from YAML file: {yaml_path}")
            config = read_yaml_file(yaml_path)

            model_selection_config = config.get("model_selection", {})
            models_config = model_selection_config.get("models", {})

            if not models_config:
                raise ValueError("No models defined under 'model_selection.models' in YAML.")

            models = {}

            for model_name, model_info in models_config.items():
                try:
                    class_path = model_info.get("class", "")
                    if not class_path:
                        raise ValueError(f"No class path provided for model: {model_name}")

                    # Dynamically load model class
                    module_path, class_name = class_path.rsplit(".", 1)
                    model_class = getattr(import_module(module_path), class_name)

                    # Clean params (handle None/null)
                    raw_params = model_info.get("params", {})
                    clean_params = {
                        k: [None if val is None else val for val in v]
                        for k, v in raw_params.items()
                    }

                    # Store model and params
                    models[model_name] = (model_class(), clean_params)

                    logging.info(f"✅ Loaded {model_name} with params: {clean_params}")

                except (ImportError, AttributeError) as e:
                    logging.error(f"❌ Failed to load class for {model_name}: {e}")
                except Exception as e:
                    logging.error(f"⚠️ Error with model '{model_name}': {e}")

            if not models:
                raise RuntimeError("No valid models loaded from YAML.")

            return models

        except Exception as e:
            logging.exception("Failed to load models from YAML.")
            raise


    # =============================================
    # 📊 Train and evaluate models, return best one
    # =============================================

    def get_model_object_and_report(self, train: np.ndarray, test: np.ndarray) -> Tuple[BestModelDetail, ClassificationMetricArtifact]:
        """
        Trains all models from YAML using GridSearchCV.
        Selects the best model based on F1 Score.
        Also returns precision and recall.
        """
        try:
            logging.info("🧪 Training and evaluating all models...")

            # Split X and y from numpy arrays
            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            # Load all candidate models
            models = self._load_models_from_yaml(self.model_trainer_config.model_config_file_path)

            # Init trackers
            best_model, best_score, best_params = None, 0.0, {}
            precision, recall = 0.0, 0.0

            # Grid search for each model
            for model_name, (model_obj, param_grid) in models.items():
                try:
                    logging.info(f"🔧 GridSearchCV for model: {model_name}")
                    grid = GridSearchCV(model_obj, param_grid=param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
                    grid.fit(x_train, y_train)

                    preds = grid.predict(x_test)
                    f1 = f1_score(y_test, preds, average='macro')

                    logging.info(f"📈 {model_name} F1 Score: {f1:.4f}")

                    if f1 > best_score:
                        best_model = grid.best_estimator_
                        best_params = grid.best_params_
                        best_score = f1
                        precision = precision_score(y_test, preds, average='macro', zero_division=0)
                        recall = recall_score(y_test, preds, average='macro', zero_division=0)

                except Exception as e:
                    logging.error(f"❌ Error training model {model_name}: {e}")
                    continue

            # Fallback: if all models failed
            if best_model is None:
                logging.warning("⚠️ No model met F1 score threshold. Returning default fallback model.")
                first_model_name = next(iter(models))
                return BestModelDetail(models[first_model_name][0], 0.0, {}), ClassificationMetricArtifact(0.0, 0.0, 0.0)

            # Save metrics
            metric_artifact = ClassificationMetricArtifact(
                f1_score=float(best_score),
                precision_score=float(precision),
                recall_score=float(recall)
            )
            metric_file_path = "artifact/model_trainer/metric_artifact.yaml"
            with open(metric_file_path, 'w') as f:
                yaml.safe_dump(metric_artifact.__dict__, f)

            logging.info(f"📁 Metric artifact saved to {metric_file_path}")

            return BestModelDetail(best_model, float(best_score), best_params), metric_artifact

        except Exception as e:
            raise USvisaException(e, sys)


    # =============================================
    # 🚀 Main Entry: Train, Evaluate, Save Model
    # =============================================

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Orchestrates full model training:
        - Loads transformed data
        - Trains models
        - Selects best
        - Saves trained model and metrics
        """
        try:
            logging.info("🚀 Initiating model training pipeline")

            # Step 1: Load transformed train/test arrays
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            # Step 2: Train and evaluate models
            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            # Display best model info
            print("✅ Best Model:", best_model_detail.best_model)
            print("🔢 Best Score:", best_model_detail.best_score)
            print("🧪 Best Params:", best_model_detail.best_params)

            # Step 3: Check accuracy threshold
            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                raise USvisaException(Exception("Model did not meet expected accuracy threshold."), sys)

            # Step 4: Load preprocessing pipeline
            preprocessing_obj = load_object(self.data_transformation_artifact.transformed_object_file_path)

            # Step 5: Save model bundle
            model_bundle = {
                "preprocessing_object": preprocessing_obj,
                "trained_model_object": best_model_detail.best_model
            }
            save_object(self.model_trainer_config.trained_model_file_path, model_bundle)

            # Step 6: Return artifact
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

        except Exception as e:
            raise USvisaException(e, sys)
        
        
if __name__ == "__main__":

    try:
        logging.info("🚦 Starting model training script...")

        # Load transformation config (contains paths to transformed train/test)
        transformation_config = DataTransformationConfig()

        # Create artifact manually or load it from file
        data_transformation_artifact = DataTransformationArtifact(
            transformed_train_file_path=transformation_config.transformed_train_file_path,
            transformed_test_file_path=transformation_config.transformed_test_file_path,
            transformed_object_file_path=transformation_config.transformed_object_file_path
        )

        # Load model trainer config (e.g., YAML path, accuracy threshold, save path)
        trainer_config = ModelTrainerConfig()

        # Initialize trainer
        trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=trainer_config
        )

        # Start training
        trainer_artifact = trainer.initiate_model_trainer()

        # Log and print results
        logging.info("✅ Model training completed successfully.")
        print("✅ Trained model saved at:", trainer_artifact.trained_model_file_path)
        print("📊 F1 Score:", trainer_artifact.metric_artifact.f1_score)
        print("📊 Precision:", trainer_artifact.metric_artifact.precision_score)
        print("📊 Recall:", trainer_artifact.metric_artifact.recall_score)

    except Exception as e:
        logging.exception("❌ Error occurred during model training.")
        raise e

