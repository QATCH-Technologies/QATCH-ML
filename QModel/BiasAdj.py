import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random
from QMultiModel import QPredictor
from QImageClusterer import QClusterer
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from QConstants import *
from hyperopt.early_stop import no_progress_loss

TRAIN_PATH = "content/test_data/train"
TEST_PATH = "content/test_data/test"
TEST_BATCH_SIZE = 1.0
predictor_0 = QPredictor("QModel/SavedModels/QMultiType_0.json")
predictor_1 = QPredictor("QModel/SavedModels/QMultiType_1.json")
predictor_2 = QPredictor("QModel/SavedModels/QMultiType_2.json")
cluster_model = QClusterer(model_path="QModel/SavedModels/cluster.joblib")


class XGBRegressorTuner:
    def __init__(self, X, y, n_iter=250):
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.best_params = None
        self.best_score = float("inf")
        self.trials = Trials()
        self.model = None

    def objective(self, params):
        model = xgb.XGBRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            min_child_weight=params["min_child_weight"],
            colsample_bytree=params["colsample_bytree"],
            subsample=params["subsample"],
            tree_method="gpu_hist",  # Use GPU acceleration
            predictor="gpu_predictor",  # Use GPU for prediction
        )

        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = median_absolute_error(y_val, y_pred)
        return {"loss": score, "status": STATUS_OK}

    def tune(self):
        search_space = {
            "n_estimators": hp.quniform("n_estimators", 50, 500, 10),
            "max_depth": hp.quniform("max_depth", 3, 15, 1),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
            "gamma": hp.uniform("gamma", 0, 0.5),
            "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "subsample": hp.uniform("subsample", 0.5, 1),
        }

        best = fmin(
            fn=self.objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.n_iter,
            trials=self.trials,
            early_stop_fn=no_progress_loss(10),
        )

        self.best_params = {
            "n_estimators": int(best["n_estimators"]),
            "max_depth": int(best["max_depth"]),
            "learning_rate": best["learning_rate"],
            "gamma": best["gamma"],
            "min_child_weight": int(best["min_child_weight"]),
            "colsample_bytree": best["colsample_bytree"],
            "subsample": best["subsample"],
        }
        self.best_score = min([trial["result"]["loss"] for trial in self.trials.trials])

        return self.best_params, self.best_score

    def train_model(self):
        if self.best_params is None:
            raise ValueError("You must call `tune()` before `train_best_model()`")

        self.model = xgb.XGBRegressor(
            n_estimators=self.best_params["n_estimators"],
            max_depth=self.best_params["max_depth"],
            learning_rate=self.best_params["learning_rate"],
            gamma=self.best_params["gamma"],
            min_child_weight=self.best_params["min_child_weight"],
            colsample_bytree=self.best_params["colsample_bytree"],
            subsample=self.best_params["subsample"],
            tree_method="gpu_hist",  # Use GPU acceleration
            predictor="gpu_predictor",  # Use GPU for prediction
        )

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        test_score = median_absolute_error(y_test, y_pred)

        return test_score

    def predict(self, X_new):
        if self.model is None:
            raise ValueError("You must call `train_best_model()` before `predict()`")

        return self.model.predict(X_new)

    def get_best_params(self):
        return self.best_params

    def get_best_score(self):
        return self.best_score

    def get_model(self):
        return self.model


def load_test_dataset(path, test_size):
    content = []
    for root, dirs, files in os.walk(path):
        for file in files:
            content.append(os.path.join(root, file))

    num_files_to_select = int(len(content) * test_size)

    if num_files_to_select == 0 and len(content) > 0:
        num_files_to_select = 1

    if num_files_to_select > len(content):
        return content

    return random.sample(content, num_files_to_select)


def run_model(filename):
    label = cluster_model.predict_label(filename)
    if label == 0:
        predictions, _ = predictor_0.predict(filename, type=label)
    elif label == 1:
        predictions, _ = predictor_1.predict(filename, type=label)
    elif label == 2:
        predictions, _ = predictor_2.predict(filename, type=label)
    else:
        raise ValueError(f"Invalid predicted label was: {label}")

    return np.array(predictions), label


def residual(predictions, actual):
    return actual - predictions


def run():
    TARGET_TYPE = 2
    content = load_test_dataset(TRAIN_PATH, TEST_BATCH_SIZE)
    X = []
    y = []
    for filename in tqdm(content, desc="<<Running Training>>"):
        if (
            filename.endswith(".csv")
            and not filename.endswith("_poi.csv")
            and not filename.endswith("_lower.csv")
        ):
            poi_file = filename.replace(".csv", "_poi.csv")
            if os.path.exists(poi_file):
                predictions, label = run_model(filename)
                if label == TARGET_TYPE:
                    actual = pd.read_csv(poi_file, header=None).values
                    actual = [int(x[0]) for x in actual]
                    bias = residual(predictions, actual)
                    X.append(predictions)
                    y.append(bias)

    # bais_adjuster = MultiOutputRegressor(base_regressor)
    X = np.array(X)
    y = np.array(y)
    bais_adjuster = XGBRegressorTuner(X, y)
    bais_adjuster.tune()
    bais_adjuster.train_model()
    best_params = bais_adjuster.get_best_params()
    print(best_params)
    content = load_test_dataset(TEST_PATH, TEST_BATCH_SIZE)
    avg_mse_before = []
    avg_mse_after = []
    for filename in tqdm(content, desc="<<Running Tests>>"):
        if (
            filename.endswith(".csv")
            and not filename.endswith("_poi.csv")
            and not filename.endswith("_lower.csv")
        ):
            poi_file = filename.replace(".csv", "_poi.csv")
            if os.path.exists(poi_file):
                df = pd.read_csv(filename)
                diss = df["Dissipation"]
                predictions, label = run_model(filename)
                if label == TARGET_TYPE:
                    actual = pd.read_csv(poi_file, header=None).values
                    actual = [int(x[0]) for x in actual]
                    bias_correction = bais_adjuster.predict(np.array([predictions]))

                    corrected_predictions = predictions + bias_correction
                    print(f">> Original: {predictions}")
                    print(f">> Corrected: {corrected_predictions[0]}")
                    print(f">> Actual {actual}")
                    mse_before = mean_absolute_error(
                        actual, predictions, multioutput="uniform_average"
                    )

                    mse_after = mean_absolute_error(
                        actual, corrected_predictions[0], multioutput="uniform_average"
                    )
                    avg_mse_before.append(mse_before)
                    avg_mse_after.append(mse_after)
                    # plt.figure()
                    # plt.plot(diss, color="grey", label="Dissipation")
                    # for i, p in enumerate(actual):
                    #     if i == 0:
                    #         plt.axvline(
                    #             p, linestyle="--", color="black", label="Actual POIs"
                    #         )
                    #     plt.axvline(p, linestyle="--", color="black")
                    # for i, p in enumerate(predictions):
                    #     if i == 0:
                    #         plt.axvline(p, color="blue", label="Predicted POIs")
                    #     plt.axvline(p, color="blue")
                    # for i, p in enumerate(corrected_predictions[0]):
                    #     if i == 0:
                    #         plt.axvline(p, color="green", label="Corrected POIs")
                    #     plt.axvline(p, color="green")
                    # plt.legend()
                    # plt.title(f"Type {label}: {filename}")
                    # plt.show()

    print(f"Avg MSE Before: {np.average(np.array(avg_mse_before))}")
    print(f"Avg MSE After: {np.average(np.array(avg_mse_after))}")


if __name__ == "__main__":
    run()
