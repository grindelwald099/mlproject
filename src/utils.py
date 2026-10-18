import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.compose import ColumnTransformer

def save_object(file_path: str,obj: object):
    '''
    This function is used to save the object to the file path
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        logging.exception("Error occurred while saving object")
        raise CustomException(e, sys) from e

def load_object(file_path: str,obj: object):
    '''
    This function is used to load the object from the file path
    '''
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.exception("Error occurred while loading object")
        raise CustomException(e, sys) from e


def save_numpy_array(file_path: str,arr: np.ndarray):
    '''
    This function is used to save the numpy array to the file path
    '''
    try:
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,arr)
    except Exception as e:
        logging.exception("Error occurred while saving numpy array")
        raise CustomException(e, sys) from e

def load_numpy_array(file_path: str,arr: np.ndarray):
    '''
    This function is used to load the numpy array from the file path
    '''
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        logging.exception("Error occurred while loading numpy array")
        raise CustomException(e, sys) from e
        
