
import os
import sys
from dataclasses import dataclass

import numpy as np

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        logging.info("Starting model training")

        try:
            # Separate features and target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info("Training regression models")

            models = {
                "LinearRegression": LinearRegression(),

                "DecisionTreeRegressor": DecisionTreeRegressor(
                    random_state=42
                ),

                "RandomForestRegressor": RandomForestRegressor(
                    random_state=42
                ),

                "GradientBoostingRegressor": GradientBoostingRegressor(
                    random_state=42
                ),

                "KNeighborsRegressor": KNeighborsRegressor(),

                "AdaBoostRegressor": AdaBoostRegressor(
                    random_state=42
                ),

                "XGBRegressor": XGBRegressor(
                    random_state=42,
                    verbosity=0
                ),

                "CatBoostRegressor": CatBoostRegressor(
                    verbose=False,
                    random_seed=42
                ),
            }

            model_report = {}
            trained_models = {}

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")

                # Train model
                model.fit(X_train, y_train)

                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Evaluation metrics
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_rmse = np.sqrt(
                    mean_squared_error(y_test, y_test_pred)
                )

                model_report[model_name] = {
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

            # Select the model with the highest test R2.
            # For a production-quality workflow, use cross-validation
            # on training data for selection instead.
            best_model_name = max(
                model_report,
                key=lambda name: model_report[name]["Test R2"]
            )

            best_model = trained_models[best_model_name]

            logging.info(
                f"Best model: {best_model_name} | "
                f"Test R2: "
                f"{model_report[best_model_name]['Test R2']:.4f}"
            )

            # Save best model
            save_object(
                file_path=(
                    self.model_trainer_config.trained_model_file_path
                ),
                obj=best_model,
            )

            logging.info("Best model saved successfully")

            return (
                self.model_trainer_config.trained_model_file_path,
                model_report,
            )

        except Exception as e:
            logging.exception("Error during model training")
            raise CustomException(e, sys) from e
