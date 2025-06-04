import os, sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def intializing_data_ingestion(self, raw_data_path: str):
        try:
            df = pd.read_csv(raw_data_path)
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Split Train and Test data was Succesfully")

            train_data.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_data.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info(
                "to_csv Train and Test data was Succesfully, Ingestion of the data is completed"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(
                f"There is a problem with intializing_data_ingestion: {e}"
            )


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.intializing_data_ingestion(r'G:\1\Projects\1. First project\1-First-ML\NoteBooks\data\stud.csv')

    data_transformation = DataTransformation()
    preprocessed_train_data, preprocessed_test_data, _ = data_transformation.intializing_data_transformation(
        train_data, test_data
    )

    modeltrainer = ModelTrainer()
    print(modeltrainer.initializing_model_trainer(preprocessed_train_data, preprocessed_test_data))
