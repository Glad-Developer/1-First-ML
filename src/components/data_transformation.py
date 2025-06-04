import sys
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_Config_file = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    # its not recommend for cat columns : ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_columns", numerical_pipeline, numerical_columns),
                    ("categorical_columns", categorical_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(
                f"There is a problem with transforming data \n --- \n {e}", sys
            )

    def intializing_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data ,were Sucessfully")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math_score"
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            train_df_preprocessed = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            test_df_preprocessed = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Preprocessing train and test data ,were Sucessfully")

            preprocessed_train_data = np.c_[
                train_df_preprocessed, np.array(target_feature_train_df)
            ]
            preprocessed_test_data = np.c_[
                test_df_preprocessed, np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_Config_file,
                obj=preprocessing_obj,
            )

            return (
                preprocessed_train_data,
                preprocessed_test_data,
                self.data_transformation_config.preprocessor_Config_file,
            )

        except Exception as e:
            raise CustomException(
                f"There is a problem with initializing data \n --- \n {e}", sys
            )
