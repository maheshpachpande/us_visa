import os
from src.usVisaClassifier.utils import read_yaml_file
from dataclasses import dataclass

config = read_yaml_file("config.yaml")

# Configuration constants
PIPELINE_NAME = config["pipeline"]['name']
ARTIFACT_DIR = config["pipeline"]['artifact_dir']
DATA_INGESTION_DIR_NAME = config["data_ingestion"]['dir_name']
DATA_INGESTION_FEATURE_STORE_DIR = config["data_ingestion"]['feature_store_dir']
DATA_INGESTION_INGESTED_DIR = config["data_ingestion"]['ingested_dir']
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = config["data_ingestion"]['train_test_split_ratio']
DATA_INGESTION_COLLECTION_NAME = config["data_ingestion"]['collection_name']
FILE_NAME = config["files"]['raw_file']
TRAIN_FILE_NAME = config["files"]['train_file']
TEST_FILE_NAME = config["files"]['test_file']



@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = ARTIFACT_DIR
   
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


# Configuration constants
PIPELINE_NAME = config["pipeline"]['name']
ARTIFACT_DIR = config["pipeline"]['artifact_dir']
DATA_INGESTION_DIR_NAME = config["data_ingestion"]['dir_name']
DATA_INGESTION_FEATURE_STORE_DIR = config["data_ingestion"]['feature_store_dir']
DATA_INGESTION_INGESTED_DIR = config["data_ingestion"]['ingested_dir']
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = config["data_ingestion"]['train_test_split_ratio']
DATA_INGESTION_COLLECTION_NAME = config["data_ingestion"]['collection_name']
FILE_NAME = config["files"]['raw_file']
TRAIN_FILE_NAME = config["files"]['train_file']
TEST_FILE_NAME = config["files"]['test_file']

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME
    

DATA_VALIDATION_DIR_NAME = config["DATA_VALIDATION"]['DIR_NAME']
DATA_VALIDATION_DRIFT_REPORT_DIR = config["DATA_VALIDATION"]['DRIFT_REPORT_DIR']
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = config["DATA_VALIDATION"]['DRIFT_REPORT_FILE_NAME']


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR,
                                               DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
    

