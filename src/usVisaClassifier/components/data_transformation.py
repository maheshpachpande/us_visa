# =============================================
# 📦 IMPORTS & CONFIGURATION
# =============================================
import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
)

from src.usVisaClassifier.entity.config_entity import (
    DataIngestionConfig, DataValidationConfig, DataTransformationConfig
)
from src.usVisaClassifier.entity.artifact_entity import (
    DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
)
from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.utils import (
    save_object, save_numpy_array_data, read_yaml_file, drop_columns
)
from src.usVisaClassifier.entity.estimator import TargetValueMapping

import warnings
warnings.filterwarnings("ignore")

# =============================================
# 🧠 CLASS: DataTransformation
# =============================================
class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        Initialize transformation class with required configs and artifacts.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path="schema.yaml")
            self._cnf = read_yaml_file(file_path="config.yaml")
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Loads CSV file into DataFrame."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """Builds ColumnTransformer with all defined preprocessing steps."""
        try:
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()
            power_transformer = Pipeline([
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            preprocessor = ColumnTransformer([
                ("OneHotEncoder", oh_transformer, self._schema_config['oh_columns']),
                ("Ordinal_Encoder", ordinal_encoder, self._schema_config['or_columns']),
                ("Transformer", power_transformer, self._schema_config['transform_columns']),
                ("StandardScaler", numeric_transformer, self._schema_config['num_features'])
            ])

            return preprocessor
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Execute full data transformation pipeline:
        - Apply transformations
        - Perform SMOTEENN resampling
        - Save numpy arrays and preprocessor
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            preprocessor = self.get_data_transformer_object()
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # =================== Preprocess Train ===================
            input_train = train_df.drop(columns=[self._cnf['target']['column']])
            input_train['company_age'] = 2025 - input_train['yr_of_estab']
            input_train = drop_columns(df=input_train, cols=self._schema_config['drop_columns'])
            target_train = train_df[self._cnf['target']['column']].replace(TargetValueMapping()._asdict())

            # =================== Preprocess Test ===================
            input_test = test_df.drop(columns=[self._cnf['target']['column']])
            input_test['company_age'] = 2025 - input_test['yr_of_estab']
            input_test = drop_columns(df=input_test, cols=self._schema_config['drop_columns'])
            target_test = test_df[self._cnf['target']['column']].replace(TargetValueMapping()._asdict())

            # =================== Transform Features ===================
            input_train_arr = preprocessor.fit_transform(input_train)
            input_test_arr = preprocessor.transform(input_test)

            # =================== Handle Imbalance with SMOTEENN ===================
            smt = SMOTEENN(sampling_strategy="minority")
            
            train_final = smt.fit_resample(input_train_arr, target_train)
            input_train_final = train_final[0]
            target_train_final = train_final[1]
            
            test_final = smt.fit_resample(input_test_arr, target_test)
            input_test_final = test_final[0]
            target_test_final = test_final[1]

            # =================== Save Transformed Data ===================
            train_arr = np.c_[input_train_final, np.array(target_train_final)]
            test_arr = np.c_[input_test_final, np.array(target_test_final)]

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise USvisaException(e, sys)


# =============================================
# 🚦 RUN TRANSFORMATION PIPELINE (Main Block)
# =============================================
if __name__ == "__main__":
    try:
        data_ingetion_cnf = DataIngestionConfig()
        data_ingestion_artifact = DataIngestionArtifact(
            trained_file_path=data_ingetion_cnf.training_file_path,
            test_file_path=data_ingetion_cnf.testing_file_path
        )

        data_val_cnf = DataValidationConfig()
        validation_artifact = DataValidationArtifact.load(
            file_path="artifact/data_validation/data_validation_artifact.yaml"
        )

        trf = DataTransformation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_transformation_config=DataTransformationConfig(),
            data_validation_artifact=validation_artifact
        )

        trf.initiate_data_transformation()

    except Exception as e:
        raise USvisaException(e, sys)
