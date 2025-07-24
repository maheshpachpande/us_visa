import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.usVisaClassifier.exception import USvisaException
from src.usVisaClassifier.logger import logging



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
    
