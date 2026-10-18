
import os
import sys
from dataclasses import dataclass

import numpy as np

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

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
        logging.info("Starting model training and hyperparameter tuning")

        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            # Include Linear Regression as a baseline.
            models = {
                "LinearRegression": (
                    LinearRegression(),
                    {}
                ),
                "Ridge": (
                    Ridge(),
                    {"alpha": [0.1, 1.0, 10.0, 100.0]}
                ),
                "DecisionTreeRegressor": (
                    DecisionTreeRegressor(random_state=42),
                    {
                        "max_depth": [None, 3, 5, 8, 12],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    }
                ),
                "RandomForestRegressor": (
                    RandomForestRegressor(random_state=42, n_jobs=-1),
                    {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [None, 5, 10, 15],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": [1.0, "sqrt"]
                    }
                ),
                "GradientBoostingRegressor": (
                    GradientBoostingRegressor(random_state=42),
                    {
                        "n_estimators": [100, 200, 300],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "max_depth": [2, 3, 4],
                        "min_samples_leaf": [1, 2, 4]
                    }
                ),
                "KNeighborsRegressor": (
                    KNeighborsRegressor(),
                    {
                        "n_neighbors": [3, 5, 7, 9, 11, 15],
                        "weights": ["uniform", "distance"],
                        "p": [1, 2]
                    }
                ),
                "AdaBoostRegressor": (
                    AdaBoostRegressor(random_state=42),
                    {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]
                    }
                ),
                "XGBRegressor": (
                    XGBRegressor(
                        random_state=42,
                        verbosity=0,
                        n_jobs=1
                    ),
                    {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [2, 3, 5, 7],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "subsample": [0.8, 1.0],
                        "colsample_bytree": [0.8, 1.0]
                    }
                ),
                "CatBoostRegressor": (
                    CatBoostRegressor(
                        verbose=False,
                        random_seed=42,
                        allow_writing_files=False,
                        thread_count=1
                    ),
                    {
                        "iterations": [200, 400, 600],
                        "depth": [4, 6, 8],
                        "learning_rate": [0.01, 0.03, 0.1]
                    }
                )
            }

            results = {}
            best_estimators = {}

            for model_name, (model, param_grid) in models.items():
                logging.info(f"Evaluating {model_name}")

                if param_grid:
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grid,
                        n_iter=min(10, int(np.prod([
                            len(values)
                            for values in param_grid.values()
                        ]))),
                        scoring="r2",
                        cv=cv,
                        random_state=42,
                        n_jobs=-1,
                        refit=True,
                        error_score="raise"
                    )

                    search.fit(X_train, y_train)

                    best_estimator = search.best_estimator_
                    cv_r2 = search.best_score_
                    best_params = search.best_params_

                else:
                    best_estimator = clone(model)
                    scores = cross_val_score(
                        best_estimator,
                        X_train,
                        y_train,
                        scoring="r2",
                        cv=cv,
                        n_jobs=-1
                    )
                    cv_r2 = scores.mean()
                    best_estimator.fit(X_train, y_train)
                    best_params = {}

                # Evaluate only after tuning, on held-out test data.
                y_test_pred = best_estimator.predict(X_test)

                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_rmse = np.sqrt(
                    mean_squared_error(y_test, y_test_pred)
                )

                results[model_name] = {
                    "CV R2": float(cv_r2),
                    "Test R2": float(test_r2),
                    "Test MAE": float(test_mae),
                    "Test RMSE": float(test_rmse),
                    "Best Parameters": best_params
                }

                best_estimators[model_name] = best_estimator

                logging.info(
                    f"{model_name} | CV R2={cv_r2:.4f} | "
                    f"Test R2={test_r2:.4f} | "
                    f"Test MAE={test_mae:.4f} | "
                    f"Test RMSE={test_rmse:.4f} | "
                    f"Best parameters={best_params}"
                )

            # Select by cross-validation, not test performance.
            best_model_name = max(
                results,
                key=lambda name: results[name]["CV R2"]
            )

            best_model = best_estimators[best_model_name]
            best_result = results[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                f"Best model: {best_model_name} | "
                f"CV R2={best_result['CV R2']:.4f} | "
                f"Test R2={best_result['Test R2']:.4f}"
            )

            print("\n========== TUNED MODEL RESULTS ==========")
            print(f"Best model: {best_model_name}")
            print(f"Cross-validation R2: {best_result['CV R2']:.4f}")
            print(f"Test R2: {best_result['Test R2']:.4f}")
            print(f"Test MAE: {best_result['Test MAE']:.4f}")
            print(f"Test RMSE: {best_result['Test RMSE']:.4f}")
            print(f"Best parameters: {best_result['Best Parameters']}")
            print(
                f"Saved model: "
                f"{self.model_trainer_config.trained_model_file_path}"
            )
            print("========================================\n")

            return (
                self.model_trainer_config.trained_model_file_path,
                results
            )

        except Exception as e:
            logging.exception("Error during model training and tuning")
            raise CustomException(e, sys) from e
