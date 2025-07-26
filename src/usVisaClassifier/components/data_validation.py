#=============================================
# 📦 IMPORTS & SETUP
#=============================================

import os
import sys
import json
import pandas as pd
from pandas import DataFrame

from evidently.report import Report
from evidently.metrics.data_drift.data_drift_table import DataDriftTable

from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.utils import read_yaml_file, write_yaml_file

from src.usVisaClassifier.entity.artifact_entity import (
    DataIngestionArtifact, 
    DataValidationArtifact
)

from src.usVisaClassifier.entity.config_entity import (
    DataIngestionConfig, 
    DataValidationConfig
)

#=============================================
# 🧠 CLASS: DataValidation
#=============================================

class DataValidation:
    """
    Validates data against schema and checks for data drift.
    """

#=============================================
# 🚀 INIT METHOD
#=============================================

    def __init__(self, 
                 data_validation_config: DataValidationConfig,
                 data_ingestion_cnf: DataIngestionConfig):
        try:
            # Initializes config and loads YAML files
            logging.info(f"{'>>'*20} Data Validation Started {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_cnf = data_ingestion_cnf

            # Load schema and config
            self._schema_config = read_yaml_file("schema.yaml")
            self._config = read_yaml_file("config.yaml")
        except Exception as e:
            raise USvisaException(e, sys)

#=============================================
# ✅ COLUMN COUNT VALIDATION
#=============================================

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Ensures the number of columns matches the schema.
        """
        try:
            expected_cols = len(self._schema_config['columns'])
            actual_cols = len(dataframe.columns)
            # Confirms the column count matches the schema
            status = actual_cols == expected_cols
            
            logging.info(f"Column count validation: Expected={expected_cols}, Found={actual_cols} => Status={status}")
            return status
        except Exception as e:
            raise USvisaException(e, sys)

#=============================================
# ✅ REQUIRED COLUMNS VALIDATION
#=============================================

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Checks for required numerical and categorical columns.
        """
        try:
            df_columns = df.columns
            missing_numerical = [col for col in self._schema_config["numerical_columns"] if col not in df_columns]
            missing_categorical = [col for col in self._schema_config["categorical_columns"] if col not in df_columns]

            
            # Verifies required categorical/numerical columns exist
            if missing_numerical:
                logging.warning(f"❌ Missing numerical columns: {missing_numerical}")
            if missing_categorical:
                logging.warning(f"❌ Missing categorical columns: {missing_categorical}")

            return not (missing_numerical or missing_categorical)
        except Exception as e:
            raise USvisaException(e, sys)

#=============================================
# 📄 READ DATA FROM CSV
#=============================================

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        """
        Loads data from a CSV file.
        """
        try:
            # Reads CSV into DataFrame
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

#=============================================
# 📊 DRIFT DETECTION USING EVIDENTLY
#=============================================

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Runs drift detection and saves a YAML report.
        """
        try:
            logging.info("🔍 Running data drift detection with Evidently")
            
            # Uses evidently to detect statistical distribution shifts

            drift_report = Report(metrics=[DataDriftTable()])
            drift_report.run(reference_data=reference_df, current_data=current_df)

            drift_dict = drift_report.as_dict()

            write_yaml_file(self.data_validation_config.drift_report_file_path, drift_dict)

            drift_result = drift_dict['metrics'][0]['result']
            drift_status = drift_result['dataset_drift']

            logging.info(f"📈 Drift detected in {drift_result['number_of_drifted_columns']} out of {drift_result['number_of_columns']} columns.")
            return drift_status
        except Exception as e:
            logging.error("❌ Drift detection failed")
            raise USvisaException(e, sys)

#=============================================
# 🚦 INITIATE DATA VALIDATION PROCESS
#=============================================

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Orchestrates validation steps and returns a validation artifact.
        """
        try:
            logging.info("🚧 Initiating data validation process")
            
            # Combines all checks, generates a validation report and artifact
            train_df = self.read_data(self.data_ingestion_cnf.training_file_path)
            test_df = self.read_data(self.data_ingestion_cnf.testing_file_path)

            error_msg = ""

            # 1. Column count validation
            if not self.validate_number_of_columns(train_df):
                error_msg += "Train file has incorrect number of columns. "
            if not self.validate_number_of_columns(test_df):
                error_msg += "Test file has incorrect number of columns. "

            # 2. Column existence validation
            if not self.is_column_exist(train_df):
                error_msg += "Missing required columns in train data. "
            if not self.is_column_exist(test_df):
                error_msg += "Missing required columns in test data. "

            # 3. Drift check if no schema errors
            validation_status = False if error_msg else True
            if validation_status:
                drift_detected = self.detect_dataset_drift(train_df, test_df)
                error_msg = "Drift detected" if drift_detected else "No drift detected"
                validation_status = not drift_detected

            # 4. Build and save validation artifact
            artifact = DataValidationArtifact(
                validation_status=validation_status,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                data_validation_report=self.data_validation_config.data_validation_report,
                message=error_msg
            )

            write_yaml_file(self.data_validation_config.data_validation_report, artifact.__dict__)
            logging.info(f"✅ Data validation completed. Artifact: {artifact}")
            return artifact

        except Exception as e:
            raise USvisaException(e, sys)
