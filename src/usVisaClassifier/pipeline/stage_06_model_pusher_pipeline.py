# ==========================================
# 📦 MODEL TRAINING PIPELINE SCRIPT
# ==========================================

import sys
from src.usVisaClassifier.components.model_pusher import ModelPusher
from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.utils import read_yaml_file
from src.usVisaClassifier.entity.config_entity import (
    ModelPusherConfig,
    ModelEvaluationConfig
)
from src.usVisaClassifier.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelPusherArtifact)


STAGE_NAME = "Model Pusher Stage"


class ModelPusherPipeline:
    """
    Orchestrates the model training step of the US Visa classification pipeline.
    This includes loading transformed data and training multiple models with hyperparameter tuning.
    """

    def __init__(self):
        pass

    def main(self):
        try:
            model_evaluation_cnf = ModelEvaluationConfig()
            model_evaluation_metrics = read_yaml_file(file_path="artifact/model_evaluation/model_evaluation_artifact.yaml")
            
            model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=model_evaluation_metrics['is_model_accepted'],
                                                                s3_model_path=model_evaluation_metrics['s3_model_path'],
                                                                trained_model_path=model_evaluation_metrics['trained_model_path'],
                                                                changed_accuracy=model_evaluation_metrics['changed_accuracy'])
            
            model_pusher_cnf = ModelPusherConfig()
            model_pusher_artifact = ModelPusherArtifact(bucket_name=model_pusher_cnf.bucket_name,
                                                    s3_model_path=model_pusher_cnf.s3_model_key_path)
            
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
                                    model_pusher_config=model_pusher_cnf)
            model_pusher_artifact = model_pusher.initiate_model_pusher()   
        
        except Exception as e:
            raise USvisaException(e, sys)



# ==========================================
# 🚀 ENTRY POINT
# ==========================================
if __name__ == '__main__':
    try:
        logging.info(f">>>>>> STAGE: {STAGE_NAME} started <<<<<<")
        pipeline = ModelPusherPipeline()
        pipeline.main()
        logging.info(f">>>>>> STAGE: {STAGE_NAME} completed <<<<<<\n\nx==========x\n")
    except Exception as e:
        logging.exception(f"❌ Exception occurred in {STAGE_NAME}: {e}")
        raise e
