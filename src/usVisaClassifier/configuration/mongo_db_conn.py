import sys
import os

from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging

from src.usVisaClassifier.constants import DATABASE_NAME, MONGODB_URL_KEY
import pymongo
import certifi
from dotenv import load_dotenv

load_dotenv()
ca = certifi.where()

class MongoDBClient:
    """
    Class Name :   export_data_into_feature_store
    Description :   This method exports the dataframe from mongodb feature store as dataframe 
    
    Output      :   connection to mongodb database
    On Failure  :   raises an exception
    """
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv("MONGODB_URL_KEY")
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection succesfull")
        except Exception as e:
            raise USvisaException(e,sys)
        
# if __name__ == "__main__":
#     try:
#         mongo_client = MongoDBClient()
#         print(f"Connected to MongoDB database: {mongo_client.database_name}")
#     except USvisaException as e:
#         logging.error(f"Error connecting to MongoDB: {e}")
#         sys.exit(1)
