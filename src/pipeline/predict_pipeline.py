import sys, os
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, input_Features):
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "components",
                "artifacts",
                "model.pkl",
            )
            preprocessor_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "components",
                "artifacts",
                "preprocessor.pkl",
            )

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            preprocessed_data = preprocessor.transform(input_Features)
            prediction = model.predict(preprocessed_data)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)


@dataclass
class Inputed_data:
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
