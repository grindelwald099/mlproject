
import sys
import os
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join(
                "artifacts", "preprocessor.pkl"
            )

            logging.info("Loading trained model and preprocessor")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("Transforming input features")

            data_scaled = preprocessor.transform(features)

            logging.info("Generating prediction")

            preds = model.predict(data_scaled)

            print("\n========== PREDICTION ==========")
            print("Input features:")
            print(features)
            print(f"Predicted Maths Score: {float(preds[0]):.2f}")
            print("================================\n")

            return preds

        except Exception as e:
            logging.exception("Error occurred while predicting")
            raise CustomException(e, sys) from e


class CustomData:
    def __init__(
        self,
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = (
            parental_level_of_education
        )
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_frame(self):
        try:
            data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [
                    self.parental_level_of_education
                ],
                "lunch": [self.lunch],
                "test_preparation_course": [
                    self.test_preparation_course
                ],
                "reading_score": [float(self.reading_score)],
                "writing_score": [float(self.writing_score)]
            }

            return pd.DataFrame(data)

        except Exception as e:
            logging.exception(
                "Error occurred while creating input DataFrame"
            )
            raise CustomException(e, sys) from e
