import os
import sys
import json
from pandas import DataFrame
import pandas as pd

from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.utils import (read_yaml_file, 
                                        write_yaml_file)
from src.usVisaClassifier.entity.artifact_entity import (DataIngestionArtifact, 
                                                         DataValidationArtifact)

from src.usVisaClassifier.entity.config_entity import (DataIngestionConfig, 
                                                       DataValidationConfig)

from evidently.report import Report
from evidently.metrics.data_drift.data_drift_table import DataDriftTable



# ============================================================================================
# Data Validation Class
# ============================================================================================

class Datavalidation:
    
    def __init__(self, data_validation_config: DataValidationConfig, 
                #  data_ingestion_artifact: DataIngestionArtifact,
                 data_ingestion_cnf: DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            # self.data_ingestion_artifact = data_ingestion_artifact
            self.data_ingestion_cnf = data_ingestion_cnf
            self._schema_config = read_yaml_file(file_path="schema.yaml")
            self._config = read_yaml_file(file_path=("config.yaml"))
            
        except Exception as e:
            raise USvisaException(e, sys)
        
        
    def validate_number_of_columns(self, dataframe: DataFrame) -> bool: # type: ignore[return]
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config['columns'])
            logging.info(f"Number of columns validation status: {status}")
            return status
        
        except Exception as e:
            raise USvisaException(e, sys)
        
        
    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
                    
                if len(missing_numerical_columns)>0:
                    logging.info(f"Missing numerical column: {missing_numerical_columns}")


            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise USvisaException(e, sys) from e


    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)



    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Method Name :   detect_dataset_drift
        Description :   This method validates if drift is detected

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Running data drift detection using Evidently")

            # Create a new report with DataDriftPreset metric
            drift_report = Report(metrics=[DataDriftTable()])

            # Run the drift detection
            drift_report.run(reference_data=reference_df, current_data=current_df)

            # Save report as JSON
            drift_json = drift_report.as_dict()
            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=drift_json
            )

            # Extract drift result summary
            n_features = drift_json['metrics'][0]['result']['number_of_columns']
            n_drifted_features = drift_json['metrics'][0]['result']['number_of_drifted_columns']
            drift_status = drift_json['metrics'][0]['result']['dataset_drift']

            logging.info(f"{n_drifted_features}/{n_features} features drifted.")
            return drift_status

        except Exception as e:
            logging.exception("Exception occurred during dataset drift detection")
            raise USvisaException(e, sys)



    def initiate_data_validation(self) -> DataValidationArtifact: # type: ignore[return]
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            dv = self
            train_df, test_df = (dv.read_data(file_path=self.data_ingestion_cnf.training_file_path),
                                 dv.read_data(file_path=self.data_ingestion_cnf.testing_file_path))
            
            
            # Validate number of columns and existence of required columns
            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.validate_number_of_columns(dataframe=test_df)
            
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."
                
                
            # Validate existence of numerical and categorical columns
            status = self.is_column_exist(df=train_df)            
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"columns are missing in test dataframe."
                
            
            # If Validation Status is True, check for drift
            validation_status = len(validation_error_msg) == 0
            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info(f"Drift detected.")
                    validation_error_msg = "Drift detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation_error: {validation_error_msg}")

            # Create DataValidationArtifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys)
        


# if __name__ == "__main__":
    
#     try:        
#         data_validation_config = DataValidationConfig()
#         data_ingestion_cnf = DataIngestionConfig()
        
#         validator = Datavalidation(data_validation_config, data_ingestion_cnf)
#         validation_artifact = validator.initiate_data_validation()
#         print(validation_artifact)
#     except USvisaException as e:
#         logging.error(f"Error during data validation: {e}")