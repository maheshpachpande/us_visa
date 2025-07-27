import os
import sys
import pickle
from io import StringIO
from typing import Union, List, Any, Optional

import pandas as pd
from pandas import DataFrame, read_csv
from botocore.exceptions import ClientError

from src.usVisaClassifier.configuration.aws_conn import S3Client
from src.usVisaClassifier.logger import logging
from src.usVisaClassifier.exception import USvisaException

from mypy_boto3_s3.service_resource import S3ServiceResource, Bucket, ObjectSummary, Object as S3Object
from mypy_boto3_s3.client import S3Client as Boto3S3Client

from mypy_boto3_s3 import S3ServiceResource
from mypy_boto3_s3.client import S3Client as Boto3S3Client

from mypy_boto3_s3.service_resource import S3ServiceResource
from mypy_boto3_s3.client import S3Client as Boto3S3Client

class SimpleStorageService:
    s3_resource: S3ServiceResource
    s3_client: Boto3S3Client

    def __init__(self):
        s3_client = S3Client()

        if s3_client.s3_resource is None or s3_client.s3_client is None:
            raise ValueError("S3 resource or client is not initialized")

        self.s3_resource = s3_client.s3_resource  # type: ignore
        self.s3_client = s3_client.s3_client  # type: ignore


    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [fo for fo in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects) > 0
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_object(
        object_name: S3Object,
        decode: bool = True,
        make_readable: bool = False,
    ) -> Union[StringIO, str, bytes]:
        logging.info("Entered the read_object method of SimpleStorageService")
        try:
            data = object_name.get()["Body"].read()
            if decode:
                data = data.decode()
            if make_readable:
                data = StringIO(data)
            logging.info("Exited the read_object method of SimpleStorageService")
            return data
        except Exception as e:
            raise USvisaException(e, sys) 

    def get_bucket(self, bucket_name: str) -> Bucket:
        logging.info("Entered the get_bucket method of SimpleStorageService")
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logging.info("Exited the get_bucket method of SimpleStorageService")
            return bucket
        except Exception as e:
            raise USvisaException(e, sys) 

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[S3Object], S3Object]:
        logging.info("Entered the get_file_object method of SimpleStorageService")
        try:
            bucket = self.get_bucket(bucket_name)
            object_summaries: List[ObjectSummary] = list(bucket.objects.filter(Prefix=filename))
            object_objects: List[S3Object] = [
                self.s3_resource.Object(bucket.name, obj.key) for obj in object_summaries
            ]

            result: Union[List[S3Object], S3Object] = (
                object_objects[0] if len(object_objects) == 1 else object_objects
            )

            logging.info("Exited the get_file_object method of SimpleStorageService")
            return result
        except Exception as e:
            raise USvisaException(e, sys) 

    def load_model(self, model_name: str, bucket_name: str, model_dir: Optional[str] = None) -> Any:
        logging.info("Entered the load_model method of SimpleStorageService")
        try:
            model_file = f"{model_dir}/{model_name}" if model_dir else model_name
            file_object = self.get_file_object(model_file, bucket_name)

            if isinstance(file_object, list):
                if len(file_object) == 0:
                    raise USvisaException(Exception(f"No model files found for key prefix '{model_file}'"), sys)
                if len(file_object) > 1:
                    raise USvisaException(Exception(f"Multiple model files found for key prefix '{model_file}', please specify exact key"), sys)
                file_object = file_object[0]

            model_obj = self.read_object(file_object, decode=False)
            if not isinstance(model_obj, (bytes, bytearray)):
                raise USvisaException(Exception("Expected bytes when reading model object for pickle"), sys)

            model = pickle.loads(model_obj)
            logging.info("Exited the load_model method of SimpleStorageService")
            return model
        except Exception as e:
            raise USvisaException(e, sys) 

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        logging.info("Entered the create_folder method of SimpleStorageService")
        folder_obj = folder_name.rstrip("/") + "/"  # Moved outside try block
        try:
            self.s3_resource.Object(bucket_name, folder_obj).load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)
                logging.info(f"Created folder '{folder_obj}' in bucket '{bucket_name}'")
            else:
                raise
        logging.info("Exited the create_folder method of SimpleStorageService")


    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True) -> None:
        logging.info("Entered the upload_file method of SimpleStorageService")
        try:
            logging.info(f"Uploading '{from_filename}' to '{to_filename}' in bucket '{bucket_name}'")
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
            logging.info(f"Successfully uploaded '{from_filename}' to '{to_filename}' in bucket '{bucket_name}'")
            if remove:
                os.remove(from_filename)
                logging.info(f"Removed local file '{from_filename}' after upload")
            else:
                logging.info(f"Kept local file '{from_filename}' after upload")
        except Exception as e:
            raise USvisaException(e, sys) 
        logging.info("Exited the upload_file method of SimpleStorageService")

    def upload_df_as_csv(self, data_frame: DataFrame, local_filename: str, bucket_filename: str, bucket_name: str) -> None:
        logging.info("Entered the upload_df_as_csv method of SimpleStorageService")
        try:
            data_frame.to_csv(local_filename, index=False, header=True)
            self.upload_file(local_filename, bucket_filename, bucket_name)
            logging.info("Exited the upload_df_as_csv method of SimpleStorageService")
        except Exception as e:
            raise USvisaException(e, sys) 

    from io import BytesIO

    def get_df_from_object(self, object_: S3Object) -> DataFrame:
        logging.info("Entered the get_df_from_object method of SimpleStorageService")
        try:
            content = self.read_object(object_, decode=True, make_readable=True)

            # Ensure the type is acceptable for read_csv
            if isinstance(content, (StringIO, str)):
                df = read_csv(content, na_values="na")
            else:
                raise USvisaException(Exception(f"Unsupported type for CSV content: {type(content)}"), sys)

            logging.info("Exited the get_df_from_object method of SimpleStorageService")
            return df
        except Exception as e:
            raise USvisaException(e, sys)


    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        logging.info("Entered the read_csv method of SimpleStorageService")
        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            if isinstance(csv_obj, list):
                if len(csv_obj) == 0:
                    raise USvisaException(Exception(f"No CSV files found for prefix '{filename}'"), sys)
                if len(csv_obj) > 1:
                    raise USvisaException(Exception(f"Multiple CSV files found for prefix '{filename}', specify exact key"), sys)
                csv_obj = csv_obj[0]
            df = self.get_df_from_object(csv_obj)
            logging.info("Exited the read_csv method of SimpleStorageService")
            return df
        except Exception as e:
            raise USvisaException(e, sys) 



if __name__ == "__main__":
    pass