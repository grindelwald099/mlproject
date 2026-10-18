import os
import sys
import dill
import numpy as np

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: object):
    """
    Save a Python object to the specified file path.
    """
    try:
        dir_path = os.path.dirname(file_path)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.exception("Error occurred while saving object")
        raise CustomException(e, sys) from e


def load_object(file_path: str):
    """
    Load a Python object from the specified file path.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        logging.exception("Error occurred while loading object")
        raise CustomException(e, sys) from e


def save_numpy_array(file_path: str, arr: np.ndarray):
    """
    Save a NumPy array to the specified file path.
    """
    try:
        dir_path = os.path.dirname(file_path)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, arr)

        logging.info(f"NumPy array saved successfully at {file_path}")

    except Exception as e:
        logging.exception("Error occurred while saving NumPy array")
        raise CustomException(e, sys) from e


def load_numpy_array(file_path: str) -> np.ndarray:
    """
    Load a NumPy array from the specified file path.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=False)

    except Exception as e:
        logging.exception("Error occurred while loading NumPy array")
        raise CustomException(e, sys) from e
