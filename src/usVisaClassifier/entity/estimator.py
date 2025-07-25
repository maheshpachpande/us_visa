import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging
from sklearn.base import ClassifierMixin



class TargetValueMapping:
    def __init__(self):
        self.Certified:int = 0
        self.Denied:int = 1
        
    def _asdict(self):
        """
        Returns a dictionary representation of the TargetValueMapping instance.
        This method is used to convert the instance attributes into a dictionary format.
        """
        return self.__dict__    
    
    def reverse_mapping(self):
        """
        Returns a dictionary that reverses the mapping of target values to their corresponding labels.
        This method is useful for converting numerical target values back to their original string labels.
        """
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))
    



# class USvisaModel:
#     def __init__(self, preprocessing_object, trained_model_object):
#         self.preprocessing_object = preprocessing_object
#         self.trained_model_object = trained_model_object

#     def predict(self, dataframe: DataFrame) -> DataFrame:
#         logging.info("Entered predict method of USvisaModel class")

#         try:
#             logging.info("Using the trained model to get predictions")

#             transformed_feature = self.preprocessing_object.transform(dataframe)
#             logging.info("Used the trained model to get predictions")

#             return self.trained_model_object.predict(transformed_feature)

#         except Exception as e:
#             raise USvisaException(e, sys) from e

#     def __repr__(self):
#         return f"{type(self.trained_model_object).__name__}()"

#     def __str__(self):
#         return f"{type(self.trained_model_object).__name__}()"
