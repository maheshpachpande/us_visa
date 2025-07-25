import sys
from typing import Tuple, Any, Dict
import numpy as np
from importlib import import_module
from sklearn.metrics import (f1_score, 
                             precision_score, 
                             recall_score)

from sklearn.model_selection import GridSearchCV

from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.exception import USvisaException

from src.usVisaClassifier.utils import (load_numpy_array_data,
                                        read_yaml_file,
                                        load_object,
                                        save_object)

from src.usVisaClassifier.entity.config_entity import (DataTransformationConfig,
                                                        ModelTrainerConfig)

from src.usVisaClassifier.entity.artifact_entity import (DataTransformationArtifact,
                                                         ModelTrainerArtifact,
                                                         ClassificationMetricArtifact)

  # adjust as per your structure
from sklearn.base import BaseEstimator

from dataclasses import dataclass



@dataclass
class BestModelDetail:
    best_model: Any
    best_score: float
    best_params: Dict[str, Any]



class ModelTrainer:
    
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    

    def _load_models_from_yaml(self, yaml_path: str) -> Dict[str, Tuple[BaseEstimator, dict]]:
        """
        Loads models and their hyperparameters from a YAML config file.

        Args:
            yaml_path (str): Path to YAML file.

        Returns:
            dict: A dictionary of model name to (model class instance, hyperparameters dict)
        """
        try:
            logging.info(f"Loading models from YAML file: {yaml_path}")
            config = read_yaml_file(yaml_path)

            # Safe access to nested dictionary
            model_selection_config = config.get("model_selection", {})
            models_config = model_selection_config.get("models", {})

            if not models_config:
                raise ValueError("No models found under 'model_selection.models' in YAML.")

            models = {}

            for model_name, model_info in models_config.items():
                try:
                    # Parse class path
                    class_path = model_info.get("class", "")
                    if not class_path:
                        raise ValueError(f"No 'class' field found for model: {model_name}")

                    module_path, class_name = class_path.rsplit(".", 1)
                    model_class = getattr(import_module(module_path), class_name)

                    # Load parameters
                    raw_params = model_info.get("params", {})
                    # Replace 'null' (YAML None) with Python None in lists
                    clean_params = {
                        k: [None if val is None else val for val in v]
                        for k, v in raw_params.items()
                    }

                    # Save
                    models[model_name] = (model_class(), clean_params)

                    logging.info(f"Loaded model: {model_name} from {class_path} with params: {clean_params}")

                except (ImportError, AttributeError) as e:
                    logging.error(f"Failed to import model class for {model_name}: {e}")
                except Exception as e:
                    logging.error(f"Error processing model '{model_name}': {e}")

            if not models:
                raise RuntimeError("No valid models were loaded from YAML.")

            return models

        except Exception as e:
            logging.exception(f"Failed to load models from YAML file: {e}")
            raise


    def get_model_object_and_report(self, train: np.ndarray, test: np.ndarray) -> Tuple[BestModelDetail, ClassificationMetricArtifact]:
        """
        Trains multiple models using GridSearchCV and returns the best model and its metrics.
        """
        try:
            logging.info("Training multiple models using GridSearchCV")

            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            models = self._load_models_from_yaml(self.model_trainer_config.model_config_file_path)

            best_model = None
            best_score = 0.0
            precision = 0.0
            recall = 0.0
            best_params = {}

            for model_name, (model_obj, param_grid) in models.items():
                try:
                    logging.info(f"Training model: {model_name}")
                    grid = GridSearchCV(model_obj, param_grid=param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
                    grid.fit(x_train, y_train)

                    preds = grid.predict(x_test)

                    f1 = f1_score(y_test, preds, average='macro')
                    logging.info(f"{model_name} - F1 Score: {f1:.4f}")

                    if f1 > best_score:
                        best_model = grid.best_estimator_
                        best_params = grid.best_params_

                        best_score = f1
                        precision = precision_score(y_test, preds, average='macro', zero_division=0)
                        recall = recall_score(y_test, preds, average='macro', zero_division=0)

                except Exception as e:
                    logging.error(f"Error training model {model_name}: {e}")
                    continue  # Try other models

            if best_model is None:
                logging.warning("All models failed or had F1 score <= 0.0")
                first_model_name = next(iter(models))
                return BestModelDetail(
                    best_model=models[first_model_name][0],
                    best_score=0.0,
                    best_params={}
                ), ClassificationMetricArtifact(
                    f1_score=0.0,
                    precision_score=0.0,
                    recall_score=0.0
                )

            metric_artifact = ClassificationMetricArtifact(
                f1_score=float(best_score),
                precision_score=float(precision),
                recall_score=float(recall)
            )

            return BestModelDetail(
                best_model=best_model,
                best_score=float(best_score),
                best_params=best_params  # ✅ Added here
            ), metric_artifact

            # return BestModelDetail(best_model=best_model, best_score=float(best_score),), metric_artifact

        except Exception as e:
            raise USvisaException(e, sys)
        
        
   
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Trains the model and saves the preprocessing object and best trained model,
        without wrapping them into a custom USvisaModel class.
        """
        try:
            logging.info("🚀 Initiating model trainer pipeline")

            # Load transformed training and testing data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            # Train models and get the best one with metrics
            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            print("Model Name ",best_model_detail.best_model)
            print("Best Score ", best_model_detail.best_score)
            print("Best params ",best_model_detail.best_params)
            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                raise USvisaException(Exception("Model did not meet expected accuracy threshold."), sys)

            # Load preprocessing object used during transformation (like StandardScaler, LabelEncoder, etc.)
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # Option 1: Save both preprocessing and model as a dictionary
            model_bundle = {
                "preprocessing_object": preprocessing_obj,
                "trained_model_object": best_model_detail.best_model
            }

            # Save the model bundle to disk
            save_object(self.model_trainer_config.trained_model_file_path, model_bundle)

            # Return model trainer artifact
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

        except Exception as e:
            raise USvisaException(e, sys)


    
# if __name__ == "__main__":
#     try:
#         trf_cnf = DataTransformationConfig()
#         data_transformation_artifact = DataTransformationArtifact(
#             transformed_train_file_path=trf_cnf.transformed_train_file_path,
#             transformed_test_file_path=trf_cnf.transformed_test_file_path,
#             transformed_object_file_path=trf_cnf.transformed_object_file_path
#         )
#         model_cnf = ModelTrainerConfig()        
#         # Create and run ModelTrainer
#         trainer = ModelTrainer(
#             data_transformation_artifact=data_transformation_artifact,
#             model_trainer_config=model_cnf
#         )

#         model_trainer_artifact = trainer.initiate_model_trainer()
#         # Log results
#         print("✅ Model training completed.")
#         print("Trained model saved at:", model_trainer_artifact.trained_model_file_path)
#         print("F1 Score:", model_trainer_artifact.metric_artifact.f1_score)
#         print("Precision:", model_trainer_artifact.metric_artifact.precision_score)
#         print("Recall:", model_trainer_artifact.metric_artifact.recall_score)
#     except Exception as e:
#         raise USvisaException(e,sys)