import os, sys
from dataclasses import dataclass
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params, cv=3):
        try:
            report = {}
            best_fitted_models = {}
            model_names = list(models.keys())
            model_objs = list(models.values())

            for i, model in enumerate(model_objs):
                param = params.get(model_names[i], {})
                grid_search = GridSearchCV(model, param_grid=param, cv=cv)
                grid_search.fit(X_train, y_train)

                #best_model = grid_search.best_estimator_
                model.set_params(**grid_search.best_params_)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)

                report[model_names[i]] = {
                    "train_score": train_score,
                    "test_score": test_score,
                    "best_params": grid_search.best_params_,
                }
                best_fitted_models[model_names[i]]=model
            return report ,best_fitted_models
        except Exception as e:
            raise CustomException(f"Error evaluating models: {e}", sys)

    def initializing_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                },
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Gradient Boosting": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    # 'loss':['linear','square','exponential'],
                },
            }

            model_report, best_fitted_models = self.evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            best_model_score = max(
                [model_scores["test_score"] for model_scores in model_report.values()]
            )
            best_model_name = next(
                name
                for name, scores in model_report.items()
                if scores["test_score"] == best_model_score
            )

            if best_model_score < 0.6:
                logging.info("There is no model with score greater than 0.6 r2_Score")
                raise CustomException(
                    "There is no model with score greater than 0.6 r2_Score"
                )
            else:
                best_model = best_fitted_models[best_model_name]

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model,
                )
                logging.info(
                    f"The best fitted model is {best_model_name} with Test_R2_Score : {best_model_score}"
                )

        except Exception as e:
            raise CustomException(
                f"There is a problem with initializing_model_trainer: {e}"
            )
