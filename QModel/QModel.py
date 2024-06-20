import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
from scipy.signal import (
    find_peaks,
    savgol_filter,
    butter,
    filtfilt,
    peak_prominences,
)
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from pykalman import KalmanFilter
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

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

# OUTPUT_PREDICTORS = [
#     "Relative_time",
#     "Dissipation",
#     "Resonance_Frequency",
#     "Peak Magnitude (RAW)",
#     "Difference",
#     "Pooling",
#     "Discrim",
# ]


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
            "eta": 0.375,
            "max_depth": 7,
            "min_child_weight": 2.0,
            "subsample": 0.95,
            "colsample_bytree": 0.7,
            "gamma": 0.65,
            "eval_metric": "auc",
            "nthread": 12,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "hist",
            "seed": RANDOM_STATE,
        }
        self.__discriminator_params__ = {
            "objective": "binary:logistic",
            "eta": 0.475,
            "max_depth": 7,
            "min_child_weight": 2.0,
            "subsample": 0.7,
            "colsample_bytree": 0.55,
            "gamma": 0.7,
            "eval_metric": "auc",
            "nthread": 12,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "hist",
            "seed": RANDOM_STATE,
        }
        # self.__output_params__ = {
        #     "objective": "binary:logistic",
        #     "eta": 0.325,
        #     "max_depth": 1,
        #     "min_child_weight": 5.0,
        #     "subsample": 0.7,
        #     "colsample_bytree": 0.8,
        #     "gamma": 0.7,
        #     "eval_metric": "auc",
        #     "nthread": 12,
        #     "booster": "gbtree",
        #     "device": "cuda",
        #     "tree_method": "hist",
        #     "seed": RANDOM_STATE,
        # }
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
        # self.__o_dtrain__ = xgb.DMatrix(
        #     self.__train_df__[OUTPUT_PREDICTORS],
        #     label=self.__train_df__[TARGET].values,
        # )
        # self.__o_dvalid__ = xgb.DMatrix(
        #     self.__valid_df__[OUTPUT_PREDICTORS],
        #     label=self.__valid_df__[TARGET].values,
        # )
        # self.__o_dtest__ = xgb.DMatrix(
        #     self.__test_df__[OUTPUT_PREDICTORS],
        #     label=self.__test_df__[TARGET].values,
        # )
        self.__pooler_watchlist__ = [
            (self.__p_dtrain__, "train"),
            (self.__p_dvalid__, "valid"),
        ]
        self.__discriminator_watchlist__ = [
            (self.__d_dtrain__, "train"),
            (self.__d_dvalid__, "valid"),
        ]
        # self.__output_watchlist__ = [
        #     (self.__o_dtrain__, "train"),
        #     (self.__o_dvalid__, "valid"),
        # ]
        self.__pooling_model__ = None
        self.__discriminator_model__ = None
        # self.__output_model__ = None

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

    # def train_output(self):
    #     """
    #     Trains the XGBoost model using the training and validation datasets.
    #     """
    #     self.__discriminator_model__ = xgb.train(
    #         self.__output_params__,
    #         self.__o_dtrain__,
    #         MAX_ROUNDS,
    #         self.__output_watchlist__,
    #         early_stopping_rounds=EARLY_STOP,
    #         maximize=True,
    #         verbose_eval=VERBOSE_EVAL,
    #     )

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
        predictions = self.__discriminator_model__.predict(
            self.__d_dvalid__,
            iteration_range=range(0, self.__discriminator_model__.best_iteration + 1),
        )
        score = roc_auc_score(self.__d_dvalid__.get_label(), predictions)
        loss = 1 - score
        return {"loss": loss, "status": STATUS_OK}

    # def score_output(self, params):
    #     self.__output_model__ = xgb.train(
    #         params,
    #         self.__o_dtrain__,
    #         MAX_ROUNDS,
    #         self.__output_watchlist__,
    #         early_stopping_rounds=EARLY_STOP,
    #         maximize=True,
    #         verbose_eval=VERBOSE_EVAL,
    #     )
    #     predictions = self.__output_model__.predict(
    #         self.__o_dvalid__,
    #         iteration_range=range(0, self.__output_model__.best_iteration + 1),
    #     )
    #     score = roc_auc_score(self.__o_dvalid__.get_label(), predictions)
    #     loss = 1 - score
    #     return {"loss": loss, "status": STATUS_OK}

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
        # if model == "o":
        #     # Use the fmin function from Hyperopt to find the best hyperparameters
        #     best = fmin(
        #         self.score_output,
        #         space,
        #         algo=tpe.suggest,
        #         # trials=trials,
        #         max_evals=evaluations,
        #     )
        #     self.__output_params__ = best
        #     print(f"-- best output parameters, \n\t{best}")
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

    # def save_output(self, model_name="QModelOutput"):
    #     """
    #     Saves the trained model to a file.

    #     Args:
    #         model_name (str, optional): Name of the model file. Defaults to "QModel".
    #     """
    #     filename = f"QModel/SavedModels/{model_name}.json"
    #     self.__output_model__.save_model(filename)

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

    # def get_output(self):
    #     """
    #     Returns the trained model.

    #     Returns:
    #         Booster: The trained XGBoost model.
    #     """
    #     return self.__discriminator_model__


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

    def histogram_analysis(self, arr, num_bins, threshold):
        hist, bins = np.histogram(arr, bins=num_bins)
        while np.count_nonzero(hist) != threshold:
            num_bins -= 1
            hist, bins = np.histogram(arr, bins=num_bins)
        return hist, bins

    def largest_value_in_bins(self, values, bin_edges):
        # Combine values with their bin indices
        combined = list(zip(values, range(len(values))))

        # Sort values based on their bin indices
        combined.sort(key=lambda x: x[0])

        # Initialize variables
        max_values = []
        current_bin_index = 0
        current_bin_values = []

        # Iterate through sorted values and assign them to bins
        for value, original_index in combined:
            # Determine current bin index
            while (
                current_bin_index < len(bin_edges) - 1
                and value >= bin_edges[current_bin_index + 1]
            ):
                current_bin_index += 1

            # Add value to current bin
            current_bin_values.append(value)

            # Check if we need to move to the next bin
            if current_bin_index == len(bin_edges) - 1 or (
                current_bin_index < len(bin_edges) - 1
                and value < bin_edges[current_bin_index + 1]
            ):
                # We are at the end of the current bin
                if current_bin_values:
                    # Find maximum value in current bin
                    max_value_in_bin = max(current_bin_values)
                    max_values.append(max_value_in_bin)

                # Move to the next bin
                current_bin_values = []
                current_bin_index += 1

        return max_values

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def noise_filter(self, predictions):
        # Butter Low-Pass
        fs = 20
        normal_cutoff = 2 / (0.5 * fs)
        # Get the filter coefficients
        b, a = butter(2, normal_cutoff, btype="low", analog=False)
        y = filtfilt(b, a, predictions)
        return y

    def interpolate_peaks(self, data, peak_indices):
        """
        Generate a linearly interpolated dataset over the maxima (peaks) of the signal.

        Parameters:
        - signal_data: numpy array or list, the original signal data
        - peak_indices: list of integers, indices of peaks in the signal

        Returns:
        - interpolated_data: numpy array, the interpolated dataset over the peaks
        """
        data = np.asarray(data)  # Convert to numpy array if not already
        interpolated_data = np.zeros_like(
            data, dtype=float
        )  # Initialize interpolated data

        # Iterate over pairs of peak indices
        for i in range(len(peak_indices) - 1):
            start_idx = peak_indices[i]
            end_idx = peak_indices[i + 1]

            # Calculate linear interpolation between start_idx and end_idx
            start_value = data[start_idx]
            end_value = data[end_idx]
            interval_values = np.linspace(
                start_value, end_value, num=end_idx - start_idx + 1
            )

            # Assign interpolated values to the corresponding indices
            interpolated_data[start_idx : end_idx + 1] = interval_values

        # Ensure the last segment is filled to the end of the signal
        last_peak_idx = peak_indices[-1]
        interpolated_data[last_peak_idx:] = data[last_peak_idx]

        return interpolated_data

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
        # Pooler predictions
        pooler_df = pooler_df[f_names]
        pooling_data = xgb.DMatrix(pooler_df)
        pooling_results = self.__pooler__.predict(pooling_data)
        pooling_results = self.normalize(pooling_results)
        pooling_results = self.noise_filter(pooling_results)

        # Discriminator Predictions
        discriminator_df = pooler_df
        discriminator_df["Pooling"] = pooling_results
        discriminator_data = xgb.DMatrix(discriminator_df)
        discriminator_results = self.__discriminator__.predict(discriminator_data)
        # Get all peaks in the dataset
        all_peaks, _ = find_peaks(
            discriminator_results, height=discriminator_results.mean()
        )
        hist, bins = self.histogram_analysis(all_peaks, len(all_peaks), 4)
        s_bound = int(bins[1])
        print(f"-- s_bound={s_bound}")
        # Find the 3 most significant peaks to the right of the largest peak.
        r_peaks, _ = find_peaks(
            discriminator_results[s_bound : len(discriminator_results)],
            height=discriminator_results[s_bound : len(discriminator_results)].mean(),
        )
        r_peaks = r_peaks + s_bound
        hist, bins = self.histogram_analysis(
            r_peaks,
            len(r_peaks),
            3,
        )
        largest = self.largest_value_in_bins(discriminator_results, bins)
        print(largest)
        r_peaks = []
        height = discriminator_results[s_bound : len(discriminator_results)].mean()
        while len(r_peaks) < 3:
            r_peaks, _ = find_peaks(
                discriminator_results[s_bound : len(discriminator_results)],
                height=height,
            )

        r_grouping = KMeans(n_clusters=3, random_state=42)
        r_grouping.fit(r_peaks.reshape(-1, 1))
        r_centroids = r_grouping.cluster_centers_ + s_bound

        # Find the 3 most significant peaks to the left of the largest peak.
        height = 0.1
        l_peaks = []
        while len(l_peaks) < 4:
            l_peaks, _ = find_peaks(
                pooling_results[0:s_bound],
                height=height,
            )

            height = height - 0.01
        if len(l_peaks) < 3:
            l_peaks = np.append(l_peaks, (max(l_peaks) + 1))
        l_grouping = KMeans(n_clusters=3, random_state=42)
        l_grouping.fit(l_peaks.reshape(-1, 1))
        l_centroids = l_grouping.cluster_centers_
        peaks = np.concatenate((l_centroids, r_centroids))

        int_list = list(map(int, peaks.flatten()))

        # Remove duplicates
        pois = list(set(int_list))
        pois.sort()

        return pooling_results, discriminator_results, pois, s_bound
