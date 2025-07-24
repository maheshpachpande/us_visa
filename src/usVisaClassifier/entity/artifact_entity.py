from dataclasses import dataclass
from dataclasses import dataclass, asdict
import yaml
import os


@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str 


@dataclass
class DataValidationArtifact:
    validation_status: bool
    drift_report_file_path: str
    data_validation_report: str
    message: str

    @classmethod
    def load(cls, file_path: str) -> "DataValidationArtifact":
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str

