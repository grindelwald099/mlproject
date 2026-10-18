
import os
import sys
import dill
import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: object):
    """Save a Python object to the specified file path."""
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
    """Load a Python object from the specified file path."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        logging.exception("Error occurred while loading object")
        raise CustomException(e, sys) from e


def save_numpy_array(file_path: str, arr: np.ndarray):
    """Save a NumPy array to the specified file path."""
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
    """Load a NumPy array from the specified file path."""
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=False)

    except Exception as e:
        logging.exception("Error occurred while loading NumPy array")
        raise CustomException(e, sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Train and evaluate multiple regression models.

    Returns:
        report: Dictionary containing metrics for every model.
        trained_models: Dictionary containing fitted model objects.
    """
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate evaluation metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(
                mean_squared_error(y_test, y_test_pred)
            )

            report[model_name] = {
                "Train R2": train_r2,
                "Test R2": test_r2,
                "Test MAE": test_mae,
                "Test RMSE": test_rmse,
            }

            trained_models[model_name] = model

            logging.info(
                f"{model_name} | "
                f"Train R2: {train_r2:.4f} | "
                f"Test R2: {test_r2:.4f} | "
                f"Test MAE: {test_mae:.4f} | "
                f"Test RMSE: {test_rmse:.4f}"
            )

        return report, trained_models

    except Exception as e:
        logging.exception("Error occurred during model evaluation")
        raise CustomException(e, sys) from e

def load_object(file_path: str):
        """Load a Python object from the specified file path."""
        try:
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj)
        except Exception as e:
            logging.exception("Error occurred while loading object")
            raise CustomException(e, sys) from e
            
            
