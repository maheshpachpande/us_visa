from src.usVisaClassifier.configuration.mongo_db_conn import MongoDBClient

from src.usVisaClassifier.utils import read_yaml_file
from src.usVisaClassifier.exception import USvisaException
import pandas as pd
import sys
from typing import Optional
import numpy as np

config = read_yaml_file("config.yaml")
DATABASE_NAME = config['database']['name']

class USvisaData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
        """
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise USvisaException(e,sys)
        

    def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:
        try:
            """
            export entire collectin as dataframe:
            return pd.DataFrame of collection
            """
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.database[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise USvisaException(e,sys)
        
# if __name__ == "__main__":
#     usvisa_data = USvisaData()
#     df = usvisa_data.export_collection_as_dataframe(collection_name="visa_data")
#     print(df.shape)