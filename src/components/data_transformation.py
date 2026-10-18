
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        Creates preprocessing pipelines for numerical
        and categorical features.
        """
        try:
            numerical_columns = [
                "writing_score",
                "reading_score"
            ]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numerical_pipeline",
                        numerical_pipeline,
                        numerical_columns
                    ),
                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_columns
                    )
                ]
            )

            return preprocessor

        except Exception as e:
            logging.exception("Error creating preprocessing object")
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read training and testing datasets")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "math_score"

            # Separate input features and target
            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing")

            # Fit only on training data
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            # Transform test data using fitted preprocessor
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # Combine features with target
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            # Save fitted preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.exception("Error during data transformation")
            raise CustomException(e, sys) from e
