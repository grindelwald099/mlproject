
import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation


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

    raw_data, train_data, test_data = (
        data_ingestion.initiate_data_ingestion()
    )

    logging.info("Starting data transformation")

    data_transformation = DataTransformation()

    transformed_data = (
        data_transformation.initiate_data_transformation(
            train_data, test_data
        )
    )

    print("Data ingestion and transformation completed successfully!")
