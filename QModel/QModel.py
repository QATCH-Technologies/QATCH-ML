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
from scipy.fftpack import fft, fftfreq, ifft


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
POOLING_PREDICTORS = [
    "Relative_time",
    "Dissipation",
    "Resonance_Frequency",
    "Peak Magnitude (RAW)",
    "Difference",
]
DISCRIMINATOR_PREDICTORS = [
    "Relative_time",
    "Dissipation",
    "Resonance_Frequency",
    "Peak Magnitude (RAW)",
    "Difference",
    "Pooling",
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
        self.__pooling_params__ = {
            "objective": "binary:logistic",
            "eta": 0.4,
            "max_depth": 6,
            "min_child_weight": 2.0,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "gamma": 0.8,
            "eval_metric": "auc",
            "nthread": 12,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "hist",
            "seed": RANDOM_STATE,
        }
        self.__discriminator_params__ = {
            "objective": "binary:logistic",
            "eta": 0.325,
            "max_depth": 1,
            "min_child_weight": 5.0,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "gamma": 0.7,
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

        self.__p_dtrain__ = xgb.DMatrix(
            self.__train_df__[POOLING_PREDICTORS],
            label=self.__train_df__[TARGET].values,
        )
        self.__p_dvalid__ = xgb.DMatrix(
            self.__valid_df__[POOLING_PREDICTORS],
            label=self.__valid_df__[TARGET].values,
        )
        self.__p_dtest__ = xgb.DMatrix(
            self.__test_df__[POOLING_PREDICTORS],
            label=self.__test_df__[TARGET].values,
        )
        self.__d_dtrain__ = xgb.DMatrix(
            self.__train_df__[DISCRIMINATOR_PREDICTORS],
            label=self.__train_df__[TARGET].values,
        )
        self.__d_dvalid__ = xgb.DMatrix(
            self.__valid_df__[DISCRIMINATOR_PREDICTORS],
            label=self.__valid_df__[TARGET].values,
        )
        self.__d_dtest__ = xgb.DMatrix(
            self.__test_df__[DISCRIMINATOR_PREDICTORS],
            label=self.__test_df__[TARGET].values,
        )
        self.__pooler_watchlist__ = [
            (self.__p_dtrain__, "train"),
            (self.__p_dvalid__, "valid"),
        ]
        self.__discriminator_watchlist__ = [
            (self.__d_dtrain__, "train"),
            (self.__d_dvalid__, "valid"),
        ]
        self.__pooling_model__ = None
        self.__discriminator_model__ = None

    def train_pooler(self):
        """
        Trains the XGBoost model using the training and validation datasets.
        """
        self.__pooling_model__ = xgb.train(
            self.__pooling_params__,
            self.__p_dtrain__,
            MAX_ROUNDS,
            self.__pooler_watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )

    def train_discriminator(self):
        """
        Trains the XGBoost model using the training and validation datasets.
        """
        self.__discriminator_model__ = xgb.train(
            self.__discriminator_params__,
            self.__d_dtrain__,
            MAX_ROUNDS,
            self.__discriminator_watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )

    def score_pooler(self, params):
        self.__pooling_model__ = xgb.train(
            params,
            self.__p_dtrain__,
            MAX_ROUNDS,
            self.__pooler_watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )
        predictions = self.__pooling_model__.predict(
            self.__p_dvalid__,
            iteration_range=range(0, self.__pooling_model__.best_iteration + 1),
        )
        score = roc_auc_score(self.__p_dvalid__.get_label(), predictions)
        loss = 1 - score
        return {"loss": loss, "status": STATUS_OK}

    def score_discrimintator(self, params):
        self.__discriminator_model__ = xgb.train(
            params,
            self.__d_dtrain__,
            MAX_ROUNDS,
            self.__discriminator_watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )
        predictions = self.__pooling_model__.predict(
            self.__p_dvalid__,
            iteration_range=range(0, self.__pooling_model__.best_iteration + 1),
        )
        score = roc_auc_score(self.__p_dvalid__.get_label(), predictions)
        loss = 1 - score
        return {"loss": loss, "status": STATUS_OK}

    def tune(self, model, evaluations=250):
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
        if model == "p":
            # Use the fmin function from Hyperopt to find the best hyperparameters
            best = fmin(
                self.score_pooler,
                space,
                algo=tpe.suggest,
                # trials=trials,
                max_evals=evaluations,
            )
            self.__pooling_params__ = best
            print(f"-- best pooling parameters, \n\t{best}")
        if model == "d":
            # Use the fmin function from Hyperopt to find the best hyperparameters
            best = fmin(
                self.score_discrimintator,
                space,
                algo=tpe.suggest,
                # trials=trials,
                max_evals=evaluations,
            )
            self.__discriminator_params__ = best
            print(f"-- best discriminator parameters, \n\t{best}")
        return best

    def save_pooler(self, model_name="QModelPooler"):
        """
        Saves the trained model to a file.

        Args:
            model_name (str, optional): Name of the model file. Defaults to "QModel".
        """
        filename = f"QModel/SavedModels/{model_name}.json"
        self.__pooling_model__.save_model(filename)

    def save_discriminator(self, model_name="QModelDiscriminator"):
        """
        Saves the trained model to a file.

        Args:
            model_name (str, optional): Name of the model file. Defaults to "QModel".
        """
        filename = f"QModel/SavedModels/{model_name}.json"
        self.__discriminator_model__.save_model(filename)

    def get_pooler(self):
        """
        Returns the trained model.

        Returns:
            Booster: The trained XGBoost model.
        """
        return self.__pooling_model__

    def get_discriminator(self):
        """
        Returns the trained model.

        Returns:
            Booster: The trained XGBoost model.
        """
        return self.__discriminator_model__


class QModelPredict:
    """
    A class to handle loading a trained model and making predictions on new data.

    Attributes:
        __model__ (Booster): Loaded XGBoost model.
    """

    def __init__(self, pooler_path=None, discriminator_path=None):
        """
        Initializes the QModelPredict with a model file.

        Args:
            modelpath (str, optional): Path to the model file. Defaults to None.

        Raises:
            ValueError: If no model path is provided.
        """
        if pooler_path is None or discriminator_path is None:
            raise ValueError("[QModelPredict __init__()] No model path given")

        self.__pooler__ = xgb.Booster()
        self.__pooler__.load_model(pooler_path)
        self.__discriminator__ = xgb.Booster()
        self.__discriminator__.load_model(discriminator_path)

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
        f_names = self.__pooler__.feature_names
        pooler_df = pd.read_csv(datapath).drop(
            columns=[
                "Date",
                "Time",
                "Ambient",
                "Temperature",
            ]
        )
        pooler_df = pooler_df[f_names]
        pooling_data = xgb.DMatrix(pooler_df)
        pooling_results = self.__pooler__.predict(pooling_data)
        pooling_results = self.normalize(pooling_results)
        pooling_results = self.noise_filter(pooling_results)

        discriminator_df = pooler_df
        discriminator_df["Pooling"] = pooling_results
        discriminator_data = xgb.DMatrix(discriminator_df)
        discriminator_results = self.__discriminator__.predict(discriminator_data)

        discriminator_results = fft(discriminator_results)
        discriminator_results[len(discriminator_results) // 2 :] = np.zeros(
            len(discriminator_results) // 2
        )
        discriminator_results = ifft(discriminator_results)
        # print(max(discriminator_df["Relative_time"]))
        # sample_rate = len(discriminator_results) / max(
        #     discriminator_df["Relative_time"]
        # )
        # print(sample_rate)
        # N = len(discriminator_results)
        # discriminator_results = fftfreq(N, 1 / sample_rate)

        prediction_threshold = np.mean(discriminator_results)
        # discriminator_results = savgol_filter(discriminator_results, 3, 1)
        peaks, _ = find_peaks(discriminator_results)

        # Find the largest peak
        largest_peak = peaks[np.argmax(discriminator_results[peaks])]
        print(f"largets peaks: {largest_peak}")
        height = 0.1
        l_peaks = []
        while len(l_peaks) < 2:
            l_peaks, _ = find_peaks(
                discriminator_results[0:largest_peak],
                height=height,
                distance=None,
            )
            height = height - 0.01
        l_peaks = np.append(l_peaks, largest_peak)
        r_peaks = []
        height = 0.1
        while len(r_peaks) < 3:
            r_peaks, _ = find_peaks(
                savgol_filter(
                    discriminator_results[largest_peak : len(discriminator_results)],
                    20,
                    1,
                ),
                height=height,
            )
            height = height / 2

        # prominences = peak_prominences(pooling_results, peaks)
        peaks = np.concatenate((l_peaks, r_peaks + largest_peak))
        return pooling_results, discriminator_results, peaks
