import os
import sys
import pandas as pd
import numpy as np
import dill
import yaml
from pandas import DataFrame

from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging
from pandas import DataFrame
from scipy.stats import ks_2samp, chi2_contingency


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise USvisaException(e, sys) from e
    


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise USvisaException(e, sys) from e
    



def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        raise USvisaException(e, sys) from e
    


def save_numpy_array_data(file_path: str, array: np.ndarray):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise USvisaException(e, sys) from e
    



def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise USvisaException(e, sys) from e




def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise USvisaException(e, sys) from e



def drop_columns(df: DataFrame, cols: list)-> DataFrame:

    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns methon of utils")

    try:
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        raise USvisaException(e, sys) from e


def get_s3_sync_operations(s3_uri: str) -> list:
    """
    Generates a list of AWS S3 sync operations based on the provided S3 URI.
    This function is intended to be used for syncing data to/from S3.

    Args:
        s3_uri (str): The S3 URI (e.g., "s3://your-bucket/your-prefix").

    Returns:
        list: A list of strings, where each string represents a part of the S3 sync command.
              For example, ["aws", "s3", "sync", "--exact-timestamps", "s3://your-bucket/your-prefix", "local/path"].
    """
    return ["aws", "s3", "sync", "--exact-timestamps", s3_uri]




class DriftDetector:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def detect_drift(self, reference_df: DataFrame, current_df: DataFrame,
                     numerical_cols: list, categorical_cols: list) -> dict:
        report = {
            "numerical_drift": {},
            "categorical_drift": {},
            "drift_detected": False
        }

        # Check drift in numerical columns using KS Test
        for col in numerical_cols:
            if col in reference_df and col in current_df:
                try:
                    stat, p_val = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
                    drifted = float(p_val) < self.threshold
                except Exception as e:
                    logging.warning(f"KS test failed for '{col}': {e}")
                    p_val, drifted = None, False

                report["numerical_drift"][col] = {"p_value": p_val, "drifted": drifted}

        # Check drift in categorical columns using Chi-Square Test
        for col in categorical_cols:
            if col in reference_df and col in current_df:
                try:
                    ref_freq = reference_df[col].value_counts(normalize=True)
                    cur_freq = current_df[col].value_counts(normalize=True)
                    combined = pd.concat([ref_freq, cur_freq], axis=1).fillna(0)
                    stat, p_val, _, _ = chi2_contingency([combined.iloc[:, 0], combined.iloc[:, 1]])
                    drifted = float(p_val) < self.threshold
                except Exception as e:
                    logging.warning(f"Chi-square test failed for '{col}': {e}")
                    p_val, drifted = None, False

                report["categorical_drift"][col] = {"p_value": p_val, "drifted": drifted}

        # Flag if any drift detected
        report["drift_detected"] = any(
            drift["drifted"] for drift in report["numerical_drift"].values()
        ) or any(
            drift["drifted"] for drift in report["categorical_drift"].values()
        )

        return report