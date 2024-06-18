import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
from scipy.signal import find_peaks, peak_prominences, savgol_filter, butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from pykalman import KalmanFilter
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


# Constants for model configuration and dataset splitting
VALID_SIZE = 0.20
TEST_SIZE = 0.20
NUMBER_KFOLDS = 5
RANDOM_STATE = 2018
MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 1000
VERBOSE_EVAL = 50
TARGET = "Class"
PREDICTORS = [
    "Relative_time",
    "Dissipation",
    "Resonance_Frequency",
    "Peak Magnitude (RAW)",
    "Difference",
]


class QModel:
    """
    A class to encapsulate the XGBoost model for training and tuning.

    Attributes:
        __params__ (dict): XGBoost parameters.
        __train_df__ (DataFrame): Training dataset.
        __valid_df__ (DataFrame): Validation dataset.
        __test_df__ (DataFrame): Test dataset.
        __dtrain__ (DMatrix): DMatrix for training data.
        __dvalid__ (DMatrix): DMatrix for validation data.
        __dtest__ (DMatrix): DMatrix for test data.
        __watchlist__ (list): List of datasets to watch during training.
        __model__ (Booster): Trained XGBoost model.
    """

    def __init__(self, dataset):
        """
        Initializes the QModel with the given dataset.

        Args:
            dataset (DataFrame): The input dataset to be split and used for training.
        """
        # The default parameters are currently optimal for the VOYAGER dataset.
        self.__params__ = {
            "objective": "binary:logistic",
            "eta": 0.175,
            "max_depth": 10,
            "min_child_weight": 1.0,
            "subsample": 0.95,
            "colsample_bytree": 0.7,
            "gamma": 0.8,
            "eval_metric": "auc",
            "nthread": 12,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "hist",
            "seed": RANDOM_STATE,
        }
        self.__train_df__, self.__test_df__ = train_test_split(
            dataset, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
        self.__train_df__, self.__valid_df__ = train_test_split(
            self.__train_df__,
            test_size=VALID_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        self.__dtrain__ = xgb.DMatrix(
            self.__train_df__[PREDICTORS], label=self.__train_df__[TARGET].values
        )
        self.__dvalid__ = xgb.DMatrix(
            self.__valid_df__[PREDICTORS], label=self.__valid_df__[TARGET].values
        )
        self.__dtest__ = xgb.DMatrix(
            self.__test_df__[PREDICTORS], label=self.__test_df__[TARGET].values
        )
        self.__watchlist__ = [(self.__dtrain__, "train"), (self.__dvalid__, "valid")]
        self.__model__ = None

    def train(self):
        """
        Trains the XGBoost model using the training and validation datasets.
        """
        self.__model__ = xgb.train(
            self.__params__,
            self.__dtrain__,
            MAX_ROUNDS,
            self.__watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )

    def score(self, params):
        self.__model__ = xgb.train(
            params,
            self.__dtrain__,
            MAX_ROUNDS,
            self.__watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )
        predictions = self.__model__.predict(
            self.__dvalid__, iteration_range=range(0, self.__model__.best_iteration + 1)
        )
        score = roc_auc_score(self.__dvalid__.get_label(), predictions)
        loss = 1 - score
        return {"loss": loss, "status": STATUS_OK}

    def tune_hyperopt(self, evaluations=250):
        """
        This is the optimization function that given a space (space here) of
        hyperparameters and a scoring function (score here), finds the best hyperparameters.
        """
        # To learn more about XGBoost parameters, head to this page:
        # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
        space = {
            "eta": hp.quniform("eta", 0.025, 0.5, 0.025),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            "max_depth": hp.choice("max_depth", np.arange(1, 14, dtype=int)),
            "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
            "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
            "gamma": hp.quniform("gamma", 0.5, 1, 0.05),
            "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
            "eval_metric": "auc",
            "objective": "binary:logistic",
            # Increase this number if you have more cores. Otherwise, remove it and it will default
            # to the maxium number.
            "nthread": 12,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "hist",
            "seed": RANDOM_STATE,
        }
        # Use the fmin function from Hyperopt to find the best hyperparameters
        best = fmin(
            self.score,
            space,
            algo=tpe.suggest,
            # trials=trials,
            max_evals=evaluations,
        )
        self.__params__ = best
        print(f"-- best parameters, \n\t{best}")
        return best

    def tune_params(self, verbose=True, tunable=["shape", "sampling", "eta"]):
        """
        Tunes the model's hyperparameters using cross-validation.

        Args:
            verbose (bool, optional): If True, prints the tuning process. Defaults to True.
            tunable (list, optional): List of parameters to tune. Defaults to ["shape", "sampling", "eta"].
        """
        max_auc = 0
        best_params = self.__params__.copy()

        if "shape" in tunable:
            gridsearch_params = [
                (max_depth, min_child_weight)
                for max_depth in range(9, 12)
                for min_child_weight in range(5, 8)
            ]
            if verbose:
                print("[QModel.tune] Cross validation tuning decision tree shape,")
            for max_depth, min_child_weight in gridsearch_params:
                if verbose:
                    print(
                        f"\tCV with max_depth={max_depth}, min_child_weight={min_child_weight}"
                    )
                self.__params__["max_depth"] = max_depth
                self.__params__["min_child_weight"] = min_child_weight
                cv_results = xgb.cv(
                    self.__params__,
                    self.__dtrain__,
                    num_boost_round=MAX_ROUNDS,
                    seed=42,
                    nfold=5,
                    metrics={"auc"},
                    early_stopping_rounds=EARLY_STOP,
                )
                mean_auc = cv_results["test-auc-mean"].max()
                boost_rounds = cv_results["test-auc-mean"].argmax()
                if verbose:
                    print(f"\t\tAUC {mean_auc} for {boost_rounds} rounds.")
                if mean_auc > max_auc:
                    max_auc = mean_auc
                    best_params["max_depth"] = max_depth
                    best_params["min_child_weight"] = min_child_weight

            self.__params__.update(best_params)

        if "sampling" in tunable:
            if verbose:
                print("[QModel.tune] Cross validation tuning sampling,")
            gridsearch_params = [
                (subsample, colsample)
                for subsample in [i / 10.0 for i in range(7, 11)]
                for colsample in [i / 10.0 for i in range(7, 11)]
            ]
            for subsample, colsample in reversed(gridsearch_params):
                if verbose:
                    print(f"\tCV with subsample={subsample}, colsample={colsample}")
                self.__params__["subsample"] = subsample
                self.__params__["colsample_bytree"] = colsample
                cv_results = xgb.cv(
                    self.__params__,
                    self.__dtrain__,
                    num_boost_round=MAX_ROUNDS,
                    seed=42,
                    nfold=5,
                    metrics={"auc"},
                    early_stopping_rounds=EARLY_STOP,
                )
                mean_auc = cv_results["test-auc-mean"].max()
                boost_rounds = cv_results["test-auc-mean"].argmax()
                if verbose:
                    print(f"\t\tAUC {mean_auc} for {boost_rounds} rounds")
                if mean_auc > max_auc:
                    max_auc = mean_auc
                    best_params["subsample"] = subsample
                    best_params["colsample_bytree"] = colsample

            self.__params__.update(best_params)

        if "eta" in tunable:
            if verbose:
                print("[QModel.tune] Cross validation tuning eta,")
            for eta in [0.3, 0.2, 0.1, 0.05, 0.01, 0.005]:
                if verbose:
                    print(f"CV with eta={eta}")
                self.__params__["eta"] = eta
                cv_results = xgb.cv(
                    self.__params__,
                    self.__dtrain__,
                    num_boost_round=MAX_ROUNDS,
                    seed=42,
                    nfold=5,
                    metrics=["auc"],
                    early_stopping_rounds=EARLY_STOP,
                )
                mean_auc = cv_results["test-auc-mean"].max()
                boost_rounds = cv_results["test-auc-mean"].argmax()
                if verbose:
                    print(f"\t\tAUC {mean_auc} for {boost_rounds} rounds\n")
                if mean_auc > max_auc:
                    max_auc = mean_auc
                    best_params["eta"] = eta

            self.__params__.update(best_params)
            if verbose:
                print("---REPORT---")
                print(f"Tuning results,\n\t{best_params}")
                print("Updated model parameters to best parameters")

    def save(self, model_name="QModel"):
        """
        Saves the trained model to a file.

        Args:
            model_name (str, optional): Name of the model file. Defaults to "QModel".
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.json"
        self.__model__.save_model(filename)

    def get_model(self):
        """
        Returns the trained model.

        Returns:
            Booster: The trained XGBoost model.
        """
        return self.__model__


class QModelPredict:
    """
    A class to handle loading a trained model and making predictions on new data.

    Attributes:
        __model__ (Booster): Loaded XGBoost model.
    """

    def __init__(self, modelpath=None):
        """
        Initializes the QModelPredict with a model file.

        Args:
            modelpath (str, optional): Path to the model file. Defaults to None.

        Raises:
            ValueError: If no model path is provided.
        """
        if modelpath is None:
            raise ValueError("[QModelPredict __init__()] No model path given")

        self.__model__ = xgb.Booster()
        self.__model__.load_model(modelpath)

    def noise_filter(self, predictions):
        # PCA Filter
        # predictions = predictions.reshape(-1, 1)
        # pca = PCA(n_components=1)
        # pca.fit(predictions)
        # predictions = pca.inverse_transform(pca.transform(predictions))
        # predictions = [item[0] for item in predictions]
        # return predictions

        # Kalman Filter
        # kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        # measurements = np.asarray(predictions)
        # filtered_state_means, _ = kf.filter(measurements)
        # return filtered_state_means.flatten()

        # Butter Low-Pass
        fs = 20
        normal_cutoff = 2 / (0.5 * fs)
        # Get the filter coefficients
        b, a = butter(2, normal_cutoff, btype="low", analog=False)
        y = filtfilt(b, a, predictions)
        return y

    def normalize(self, predictions):
        return (predictions - np.min(predictions)) / (
            np.max(predictions) - np.min(predictions)
        )

    def predict(self, datapath):
        """
        Makes predictions on the given data.

        Args:
            datapath (str): Path to the CSV file containing data for predictions.

        Raises:
            ValueError: If no data path is provided.

        Returns:
            list: List of indices where the prediction exceeds the threshold.
        """
        if datapath is None:
            raise ValueError("[QModelPredict __init__()] No data path given")
        f_names = self.__model__.feature_names
        dataframe = pd.read_csv(datapath).drop(
            columns=[
                "Date",
                "Time",
                "Ambient",
                "Temperature",
            ]
        )
        dataframe = dataframe[f_names]
        xgb_data = xgb.DMatrix(dataframe)
        predictions = self.__model__.predict(xgb_data)
        predictions = self.normalize(predictions)
        print(min(predictions), max(predictions))
        predictions = self.noise_filter(predictions)

        prediction_threshold = np.mean(predictions)
        height = 0.01
        peaks = []
        while len(peaks) < 6:
            peaks, _ = find_peaks(predictions, height=height)
            height = height / 2
        prominences = peak_prominences(predictions, peaks)
        return predictions, peaks
