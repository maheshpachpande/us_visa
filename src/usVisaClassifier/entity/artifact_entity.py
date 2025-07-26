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


@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetricArtifact
    
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    s3_model_path:str 
    trained_model_path:str
    
    def to_dict(self) -> dict:
        return {
            "is_model_accepted": self.is_model_accepted,
            "s3_model_path": self.s3_model_path,
            "trained_model_path": self.trained_model_path,
            "changed_accuracy": self.changed_accuracy
        }


@dataclass
class ModelPusherArtifact:
    bucket_name:str
    s3_model_path:str