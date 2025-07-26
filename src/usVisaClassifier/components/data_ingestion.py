# =======================================================================================================
"""
Essence of the Code:
    📌 Goal:
        - Automate data ingestion for a machine learning pipeline.

    ✅ Responsibilities:
        - Extract data from MongoDB
        - Save it as a CSV in a feature store
        - Log key data stats (nulls, duplicates)
        - Split data into train and test sets
        - Return paths as structured artifact
"""
# =======================================================================================================

import os
import sys
import time
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.usVisaClassifier.entity.config_entity import DataIngestionConfig
from usVisaClassifier.entity.artifact_entity import DataIngestionArtifact
from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.data_access import USvisaData


class DataIngestion:
    """
    ⚙️ Class responsible for the full data ingestion pipeline.
    It exports data from MongoDB, logs data quality stats,
    saves to disk, splits into train/test, and returns an artifact.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initializes the DataIngestion component with configuration.

        :param data_ingestion_config: Configuration entity with Mongo collection name,
                                      feature store file path, split ratio, etc.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise USvisaException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        🔄 Extracts raw data from MongoDB and saves it as a CSV file.
        ✅ Logs key data statistics for transparency and debugging.
        :return: Extracted DataFrame
        """
        try:
            logging.info("🚀 Exporting data from MongoDB...")
            start_time = time.time()

            # 1. Extract data from MongoDB
            usvisa_data = USvisaData()
            dataframe = usvisa_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            logging.info(f"✅ Data shape: {dataframe.shape}")

            # 2. Log missing value summary
            null_counts = dataframe.isnull().sum()
            logging.info("🕳️ Null values per column:\n" + str(null_counts[null_counts > 0]))

            # 3. Log duplicates
            duplicate_count = dataframe.duplicated().sum()
            logging.info(f"🔁 Duplicate rows: {duplicate_count}")

            # 4. Save as CSV to feature store
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)
            dataframe.to_csv(feature_store_path, index=False, header=True)

            # 5. Log elapsed time
            elapsed_time = round(time.time() - start_time, 2)
            logging.info(f"💾 Feature store saved in {elapsed_time} seconds at: {feature_store_path}")

            return dataframe

        except Exception as e:
            raise USvisaException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        ✂️ Splits the dataset into training and testing datasets and saves them to disk.
        """
        logging.info("📤 Splitting data into train/test sets...")

        try:
            # 1. Perform train/test split
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info(f"✅ Split completed: Train={train_set.shape}, Test={test_set.shape}")

            # 2. Create output directory if needed
            output_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(output_dir, exist_ok=True)

            # 3. Save the splits
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info(f"📁 Train saved to: {self.data_ingestion_config.training_file_path}")
            logging.info(f"📁 Test saved to: {self.data_ingestion_config.testing_file_path}")

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        🚦 Orchestrates the full ingestion flow:
            1. Exports data from MongoDB
            2. Logs basic data stats
            3. Splits and saves train/test
            4. Returns an artifact with paths
        """
        logging.info("🔁 Starting full data ingestion process...")
        start_time = time.time()

        try:
            # Step 1: Extract and save raw data
            dataframe = self.export_data_into_feature_store()

            # Step 2: Split and save train/test
            self.split_data_as_train_test(dataframe)

            # Step 3: Log duration
            duration = round(time.time() - start_time, 2)
            logging.info(f"✅ Data ingestion completed in {duration} seconds")

            # Step 4: Create artifact to return
            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"📦 Data Ingestion Artifact: {artifact}")
            return artifact

        except Exception as e:
            raise USvisaException(e, sys)


# =======================================================================================================
#  Local Debugging or CLI Trigger
# =======================================================================================================
if __name__ == "__main__":
    try:
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        print("🧾 Artifact Output:")
        print(data_ingestion_artifact)

    except Exception as e:
        raise USvisaException(e, sys)
