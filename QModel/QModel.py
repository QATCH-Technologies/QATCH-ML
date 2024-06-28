import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import (
    find_peaks,
    peak_widths,
    peak_prominences,
    savgol_filter,
    butter,
    filtfilt,
)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score
from hyperopt import STATUS_OK, fmin, hp, tpe, Trials
from sklearn.ensemble import RandomForestClassifier
import sys

np.set_printoptions(threshold=sys.maxsize)
try:
    from QDataPipline import QDataPipeline
except:
    from QATCH.QModel.QDataPipline import QDataPipeline

""" The following are parameters for QModel to use during training time. """
""" The percentage of data to include in the validation set. """
VALID_SIZE = 0.20
""" The percentage of data to include in the test set. """
TEST_SIZE = 0.20
""" The number of folds to train. """
NUMBER_KFOLDS = 5
""" A random seed to set the state of QModel to. """
SEED = 2018
""" The number of rounds to boost for. """
MAX_ROUNDS = 1000
OPT_ROUNDS = 1000
""" Acceptable number of early stopping rounds. """
EARLY_STOP = 50
""" The number of rounds after which to print a verbose model evaluation. """
VERBOSE_EVAL = 50
""" The target supervision feature. """
TARGET_FEATURES = "Class"
""" Training features for the pooling model. """
PREDICTORS = [
    # "Difference",
    # "Difference_gradient",
    # "Dissipation_gradient",
    # "Resonance_Frequency_gradient",
    "Cumulative",
    # "Extrema",
]


class QModel:
    def __init__(self, dataset):
        self.__params__ = {
            "objective": "binary:logistic",
            "eta": 0.175,
            "max_depth": 5,
            "min_child_weight": 4.0,
            "subsample": 0.6,
            "colsample_bytree": 0.75,
            "gamma": 0.8,
            "eval_metric": "auc",
            "nthread": 12,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "auto",
            "sampling_method": "gradient_based",
            # "objective": "multi:softprob",
            # "eval_metric": "merror",
            # "num_class": 7,
            "seed": SEED,
        }
        self.__train_df__, self.__test_df__ = train_test_split(
            dataset, test_size=TEST_SIZE, random_state=SEED, shuffle=True
        )
        self.__train_df__, self.__valid_df__ = train_test_split(
            self.__train_df__,
            test_size=VALID_SIZE,
            random_state=SEED,
            shuffle=True,
        )

        self.__dtrain__ = xgb.DMatrix(
            self.__train_df__[PREDICTORS],
            label=self.__train_df__[TARGET_FEATURES].values,
        )
        self.__dvalid__ = xgb.DMatrix(
            self.__valid_df__[PREDICTORS],
            label=self.__valid_df__[TARGET_FEATURES].values,
        )
        self.__dtest__ = xgb.DMatrix(
            self.__test_df__[PREDICTORS],
            label=self.__test_df__[TARGET_FEATURES].values,
        )

        self.__watchlist__ = [
            (self.__dtrain__, "train"),
            (self.__dvalid__, "valid"),
        ]

        # self.__model__ = RandomForestClassifier(verbose=1)
        self.__model__ = None

    def train_model(self):
        # print("Fitting...")
        # self.__model__.fit(
        #     self.__train_df__[PREDICTORS], self.__train_df__[TARGET_FEATURES]
        # )
        self.__model__ = xgb.train(
            self.__params__,
            self.__dtrain__,
            MAX_ROUNDS,
            evals=self.__watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )

    def score_model(self, params):
        self.__model__ = xgb.train(
            params,
            self.__dtrain__,
            MAX_ROUNDS,
            evals=self.__watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )
        predictions = self.__model__.predict(
            self.__dvalid__,
            iteration_range=range(0, self.__model__.best_iteration),
        )

        # score = precision_score(
        #     self.__dvalid__.get_label(),
        #     predictions,
        #     average=None,
        # )
        # score = roc_auc_score(
        #     self.__dvalid__.get_label(), predictions, multi_class="ovr"
        # )
        score = roc_auc_score(
            self.__dvalid__.get_label(), predictions, average="weighted"
        )
        # print(score)
        loss = 1 - score
        return {"loss": loss, "status": STATUS_OK}

    def tune(self, evaluations=250):
        space = {
            "eta": hp.quniform("eta", 0.025, 0.5, 0.025),
            "max_depth": hp.choice("max_depth", np.arange(1, 14, dtype=int)),
            "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
            "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
            "gamma": hp.quniform("gamma", 0.5, 1, 0.05),
            "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
            "eval_metric": "auc",
            "objective": "binary:logistic",
            "nthread": 12,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "auto",
            "sampling_method": "gradient_based",
            # "objective": "multi:softprob",
            # "eval_metric": "merror",
            # "num_class": 7,
            "seed": SEED,
        }
        best = None

        best = fmin(
            self.score_model,
            space,
            algo=tpe.suggest,
            max_evals=evaluations,
            trials=Trials(),
        )
        self.__params__ = best
        print(f"-- best pooling parameters, \n\t{best}")

        return best

    def save_model(self, model_name="QModel"):
        filename = f"QModel/SavedModels/{model_name}.json"
        self.__model__.save_model(filename)

    def get_model(self):
        return self.__model__


class QModelPredict:
    def __init__(self, model_path=None):
        if model_path is None:
            raise ValueError("[QModelPredict __init__()] No model path given")

        # Load the pre-trained models from the specified paths
        self.__model__ = xgb.Booster()

        self.__model__.load_model(model_path)

    def histogram_analysis(self, data, num_bins, threshold):
        # Validate inputs
        if num_bins <= 0:
            raise ValueError(
                "[QModelPredict histrogram_analysis()]: `num_bins` must be greater than zero."
            )
        if threshold <= 0:
            raise ValueError(
                "[QModelPredict histrogram_analysis()]: `threshold` must be greater than zero."
            )
        if num_bins > len(data):
            raise ValueError(
                "[QModelPredict histrogram_analysis()]: `num_bins` cannot be greater than the length of `data`."
            )

        # Make initial partition of data into bins.
        hist, bins = np.histogram(data, bins=num_bins)

        # While the number of bins is less than the threshold number of bins (typically 3), continue
        # creating partitions until there are at least a threshold number of bins with values in them.
        # If there the number of bins gets decremented <= 0, the current number of bins is returned to
        # the caller.
        while np.count_nonzero(hist) != threshold:
            num_bins -= 1
            if num_bins <= 0:
                break
            hist, bins = np.histogram(data, bins=num_bins)
        return hist, bins

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def full_bins(self, bins, hist):
        try:
            if not isinstance(bins, np.ndarray) or not isinstance(hist, np.ndarray):
                raise ValueError(
                    "[QModelPredict full_bins()]: `bins` and `hist` must be numpy arrays."
                )
            if bins.size == 0 or hist.size == 0:
                raise ValueError(
                    "[QModelPredict full_bins()]: `bins` and `hist` cannot be empty arrays."
                )

            full_bins = [
                (int(bins[i]), int(bins[i + 1]))
                for i, height in enumerate(hist)
                if height > 0
            ]
            return full_bins
        except Exception as e:
            raise ValueError(
                "[QModelPredict full_bins()]: Invalid `bins` or `hist` input for `full_bins` function."
            ) from e

    def find_max_indices(self, data, ranges):
        try:
            # Validate `data`
            data = np.array(data)
            if data.size == 0:
                raise ValueError(
                    "[QModelPredict find_max_indices()]: `data` cannot be an empty array."
                )

            # Validate `ranges`
            if not isinstance(ranges, list):
                raise ValueError(
                    "[QModelPredict find_max_indices()]: `ranges` must be a list of tuples."
                )
            for range_tuple in ranges:
                if not isinstance(range_tuple, tuple) or len(range_tuple) != 2:
                    raise ValueError(
                        "[QModelPredict find_max_indices()]: `ranges` must be a list of tuples (start, end)."
                    )
                start, end = range_tuple
                if not (
                    isinstance(start, (int, np.integer))
                    and isinstance(end, (int, np.integer))
                ):
                    raise ValueError(
                        "[QModelPredict find_max_indices()]: Values in each tuple of `ranges` must be integers."
                    )
                if start < 0 or end >= len(data) or start > end:
                    raise ValueError(
                        "[QModelPredict find_max_indices()]: Invalid range (start, end) in `ranges`."
                    )

            # Find max indices
            max_indices = []
            for range_tuple in ranges:
                start, end = range_tuple
                sub_array = data[start : end + 1]
                max_index = np.argmax(sub_array)
                max_index_in_data = start + max_index
                max_indices.append(max_index_in_data)

            return max_indices
        except Exception as e:
            raise ValueError(
                "[QModelPredict find_max_indices()]: Invalid input for `find_max_indices` function."
            ) from e

    def noise_filter(self, data):
        try:
            # Validate `predictions`
            data = np.array(data)
            if data.size == 0:
                raise ValueError(
                    "[QModelPredict noise_filter()]: `predictions` cannot be an empty array."
                )

            # Butterworth low-pass filter parameters
            fs = 20
            normal_cutoff = 2 / (0.5 * fs)

            # Get the filter coefficients
            b, a = butter(2, normal_cutoff, btype="low", analog=False)

            # Apply the filter using filtfilt
            filtered_predictions = filtfilt(b, a, data)

            return filtered_predictions
        except Exception as e:
            raise ValueError(
                "[QModelPredict noise_filter()]: Invalid input for `noise_filter` function."
            ) from e

    def find_top_peaks(self, y, num_peaks=3, prominence=0.1, height=0.1):
        try:
            # Validate input signal `y`
            y = np.asarray(y)
            if y.size == 0:
                raise ValueError(
                    "[QModelPredict find_top_peaks()]: `y` cannot be an empty array."
                )

            # Validate `num_peaks`
            if not isinstance(num_peaks, int) or num_peaks <= 0:
                raise ValueError(
                    "[QModelPredict find_top_peaks()]: `num_peaks` must be a positive integer."
                )
            # Find peaks with their properties

            peaks, properties = find_peaks(y, prominence=prominence, height=height)
            while len(peaks) < num_peaks:
                height -= 0.01
                peaks, properties = find_peaks(y, prominence=prominence, height=height)

            prominences = peak_prominences(y, peaks)[0]
            widths = peak_widths(y, peaks, rel_height=0.5)[0]
            heights = y[peaks]

            # Define the weights for height, prominence, and width
            w_h = 0.5
            w_p = 0.25
            w_w = 0.25

            # Normalize the properties
            H_min, H_max = heights.min(), heights.max()
            P_min, P_max = prominences.min(), prominences.max()
            W_min, W_max = widths.min(), widths.max()

            # Compute the scores for each peak
            scores = (
                w_h * (heights - H_min) / (H_max - H_min)
                + w_p * (prominences - P_min) / (P_max - P_min)
                + w_w * (widths - W_min) / (W_max - W_min)
            )

            # Find the indices of the top `num_peaks` peaks with the highest scores
            if len(peaks) > 0:
                top_peak_indices = np.argsort(scores)[-min(num_peaks, len(peaks)) :]
            else:
                top_peak_indices = []

            # Extract the top peaks and their properties
            top_peaks = peaks[top_peak_indices]
            top_properties = {
                key: properties[key][top_peak_indices] for key in properties
            }

            return top_peaks, top_properties

        except Exception as e:
            raise ValueError(
                "[QModelPredict find_top_peaks()]: Invalid input for `find_top_peaks` function."
            ) from e

    def find_start_stop(self, data):
        """
        Finds the start and stop indices of a critical region in the normalized data.

        Args:
            data (array-like): The input data to analyze.

        Returns:
            tuple: A tuple containing the start and stop indices of the critical region.

        Raises:
            ValueError: If `data` is not a valid array-like object or if `data` is empty.

        Notes:
            - Normalizes `data` to facilitate slope calculations.
            - Uses a Savitzky-Golay filter for smoothing the normalized data.
            - Determines the start of the critical region based on significant positive slope changes.
            - Determines the stop of the critical region based on significant negative slope changes.

        Example:
            >>> from QModel import QModelPredict
            >>> analyzer = QModelPredict()
            >>> data = [0.1, 0.2, 0.3, 0.4, 0.5]
            >>> start_index, stop_index = analyzer.find_start_stop(data)
        """
        # Validate `data`
        data = np.asarray(data)
        if data.size == 0:
            raise ValueError(
                "[QModelPredict find_start_stop()]: `data` cannot be an empty array."
            )
        data = self.normalize(data)
        savgol_filter(data, int(len(data) * 0.01), 1)

        # The start of the critical region for initial fill.
        start = 0
        # Slopes to determine the start of this region.
        start_slopes = []

        # A buffer for the first 1% of the input data set.
        # Ensures any sensor interference does not affect the slope
        # calculation.
        buffer = int(len(data) * 0.01)
        start_slopes = np.where(
            np.arange(start + 1, len(data)) < buffer,
            0,
            (data[start + 1 :] - data[start]) / np.arange(start + 1, len(data) - start),
        )

        # Compute where there is a significant positive change in the start slopes.
        # This index gets returned as the start index of the critical region.
        start_slopes = self.normalize(start_slopes)
        start_tmp = 1
        for i in range(1, len(start_slopes)):
            if start_slopes[i] < start_slopes[start] + 0.1:
                start_tmp = i

        # Compute where there is a significant negative change in the stop slopes.
        # This index gets returned as the stop index of the critical region.
        stop_tmp = start_tmp + buffer
        stop_slopes = [0]
        for i in range(start_tmp, len(start_slopes)):
            if not (np.isnan(start_slopes[i]) or np.isnan(start_slopes[start_tmp])):
                stop_slopes.append(
                    (start_slopes[i] - start_slopes[start_tmp]) / (i - start_tmp)
                )
            if stop_slopes[-1] < stop_slopes[-2] - np.mean(stop_slopes):
                stop_tmp = i
                break
        return (start + start_tmp - buffer, start + stop_tmp)

    def classify(self, predictions):
        # Check if the input is a numpy array
        if not isinstance(predictions, np.ndarray):
            raise TypeError("Input must be a numpy array")

        # Check if each element in the array is also a numpy array with 6 elements
        for inner_array in predictions:
            if not isinstance(inner_array, np.ndarray) or len(inner_array) != 7:
                raise ValueError(
                    "Each inner array must be a numpy array with 7 elements"
                )

        # Initialize an array to store indices of maximum values
        max_indices = []
        # Loop through each inner array to find the index of the maximum value
        for inner_array in predictions:
            max_indices.append(np.argmax(inner_array))

        return max_indices

    def predict(self, file_buffer):
        # Load CSV data and drop unnecessary columns
        df = pd.read_csv(file_buffer)
        columns_to_drop = ["Date", "Time", "Ambient", "Temperature"]
        if not all(col in df.columns for col in columns_to_drop):
            raise ValueError(
                f"[QModelPredict predict()]: Input data must contain the following columns: {columns_to_drop}"
            )

        df = df.drop(columns=columns_to_drop)

        file_buffer_2 = file_buffer
        if not isinstance(file_buffer_2, str):
            if hasattr(file_buffer_2, "seekable") and file_buffer_2.seekable():
                file_buffer_2.seek(0)  # reset ByteIO buffer to beginning of stream
            else:
                # ERROR: 'file_buffer_2' must be 'BytesIO' type here, but it's not seekable!
                raise Exception(
                    "Cannot 'seek' stream prior to passing to 'QDataPipeline'."
                )
        else:
            # Assuming 'file_buffer_2' is a string to a file path, this will work fine as-is
            pass

        # Process data using QDataPipeline
        qdp = QDataPipeline(file_buffer_2)
        qdp.compute_difference()
        qdp.noise_filter(column="Cumulative")
        qdp.compute_smooth(column="Dissipation", winsize=25, polyorder=1)
        qdp.compute_smooth(column="Difference", winsize=25, polyorder=1)
        qdp.compute_smooth(column="Resonance_Frequency", winsize=25, polyorder=1)

        # qdp.interpolate()
        qdp.compute_gradient(column="Dissipation")
        qdp.compute_gradient(column="Difference")
        qdp.compute_gradient(column="Resonance_Frequency")
        qdp.fill_nan()
        qdp.trim_head()
        qdp.scale("Cumulative")
        df = qdp.get_dataframe()
        dissipation = df["Dissipation"]
        # Ensure feature names match for pooling predictions
        f_names = self.__model__.feature_names
        # df.drop(
        #     columns=[
        #         "Relative_time",
        #         "Resonance_Frequency",
        #         "Dissipation",
        #         "Date",
        #         "Time",
        #         "Ambient",
        #         "Temperature",
        #         "Peak Magnitude (RAW)",
        #     ],
        #     inplace=True,
        # )
        df = df[f_names]
        d_data = xgb.DMatrix(df)

        results = np.concatenate(
            (
                np.zeros(df.index.min()),
                self.__model__.predict(
                    d_data,
                ),
            )
        )

        # results = self.__model__.predict(d_data)
        # results = self.classify(results)
        # for r in results:
        #     print(r)
        plt.figure(figsize=(10, 10))
        plt.plot(self.normalize(results), c="r")
        plt.plot(self.normalize(dissipation), c="b")
        plt.show()
        return results
