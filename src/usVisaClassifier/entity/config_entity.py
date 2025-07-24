import os
from src.usVisaClassifier.utils import read_yaml_file
from dataclasses import dataclass

config = read_yaml_file("config.yaml")

# TrainingPipelineConfig
PIPELINE_NAME = config["pipeline"]['name']
ARTIFACT_DIR = config["pipeline"]['artifact_dir']

# Data Ingestion related constants
DATA_INGESTION_DIR_NAME = config["data_ingestion"]['dir_name']
DATA_INGESTION_FEATURE_STORE_DIR = config["data_ingestion"]['feature_store_dir']
DATA_INGESTION_INGESTED_DIR = config["data_ingestion"]['ingested_dir']
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = config["data_ingestion"]['train_test_split_ratio']
DATA_INGESTION_COLLECTION_NAME = config["database"]['collection_name']
FILE_NAME = config["files"]['raw_file']
TRAIN_FILE_NAME = config["files"]['train_file']
TEST_FILE_NAME = config["files"]['test_file']


# Data Validation related constants
DATA_VALIDATION_DIR_NAME = config["data_validation"]['dir_name']
DATA_VALIDATION_DRIFT_REPORT_DIR = config["data_validation"]['drift_report_dir']
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = config["data_validation"]["drift_report_file_name"]
DATA_VALIDATION_FILE = config["files"]['validation_file']


# Data Transformation related constants
DATA_TRANSFORMATION_DIR_NAME = config["data_transformation"]["data_transformation_dir_name"]
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = config["data_transformation"]["data_transformation_transformed_data_dir"]
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = config["data_transformation"]["data_transformation_transformed_object_dir"]
PREPROCSSING_OBJECT_FILE_NAME = config["files"]['preprocessing_object_file']



@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = ARTIFACT_DIR
   
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME
    

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    data_validation_report: str = os.path.join(data_validation_dir, DATA_VALIDATION_FILE)
    drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR,
                                               DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    
    transformed_train_file_path: str = os.path.join(data_transformation_dir, 
                                                    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                    TRAIN_FILE_NAME.replace("csv", "npy"))
    
    transformed_test_file_path: str = os.path.join(data_transformation_dir, 
                                                   DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   TEST_FILE_NAME.replace("csv", "npy"))
    
    transformed_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCSSING_OBJECT_FILE_NAME)
    
