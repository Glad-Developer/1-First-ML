import sys, os, pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(f"Error saving object to {file_path}: {e}", sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(f"Error saving object to {file_path}: {e}", sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params, cv=3):
    try:
        report = {}
        model_names = list(models.keys())
        model_objs = list(models.values())

        for i, model in enumerate(model_objs):
            # param = params[model_names[i]]

            param = params.get(model_names[i], {})
            grid_search = GridSearchCV(model, param_grid=param, cv=cv)
            grid_search.fit(X_train, y_train)

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
        return report
    except Exception as e:
        raise CustomException(f"Error evaluating models: {e}", sys)
