import os, sys
import pandas as pd
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path = os.join.path("artifacts", "train.csv")
    test_data_path = os.join.path("artifacts", "test.csv")
    raw_data_path = os.join.path("artifacts", "raw_data.csv")


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
                "to_csv Train and Test data was Succesfully \n Ingestion of the data iss completed"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(
                f"There is a problem with initializing_model_trainer: {e}"
            )


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
