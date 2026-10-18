
import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            # Read the source dataset
            source_path = os.path.join(
                "notebook", "data", "stud.csv"
            )
            df = pd.read_csv(source_path)

            logging.info("Dataset read successfully")

            # Create artifacts directory
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.raw_data_path
                ),
                exist_ok=True
            )

            # Save raw dataset
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            logging.info("Raw dataset saved successfully")

            # Split into train and test datasets
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save train and test datasets
            train_df.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            test_df.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.exception("Error occurred during data ingestion")
            raise CustomException(e, sys) from e



if __name__ == "__main__":
    data_ingestion = DataIngestion()

    # Step 1: Data ingestion
    raw_data, train_data, test_data = (
        data_ingestion.initiate_data_ingestion()
    )

    logging.info("Data ingestion completed")

    # Step 2: Data transformation
    data_transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_data,
            test_data
        )
    )

    logging.info("Data transformation completed")
    logging.info(f"Preprocessor saved at: {preprocessor_path}")

    
    # Step 3: Model training
    model_trainer = ModelTrainer()

    model_path, model_report = (
        model_trainer.initiate_model_training(
            train_arr,
            test_arr
        )
    )

    # Find the model with the highest test R2
    best_model_name = max(
        model_report,
        key=lambda name: (
            model_report[name]["Test R2"]
            if isinstance(model_report[name], dict)
            else model_report[name]
        )
    )

    best_test_r2 = (
        model_report[best_model_name]["Test R2"]
        if isinstance(model_report[best_model_name], dict)
        else model_report[best_model_name]
    )

    logging.info("Model training completed")

    print("\n========== BEST MODEL RESULTS ==========")
    print(f"Best Model Name : {best_model_name}")
    print(f"Highest Test R2 : {best_test_r2:.4f}")
    print(f"Saved Model Path: {model_path}")
    print("========================================")


