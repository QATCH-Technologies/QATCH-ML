import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.early_stop import no_progress_loss
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from ModelData import ModelData
from QConstants import *
import pickle
from scipy.signal import find_peaks
import random
from sklearn.mixture import GaussianMixture

np.set_printoptions(threshold=sys.maxsize)

# ModelData_found = False
# try:
#     if not ModelData_found:
#         import QModel.ModelData
#     ModelData_found = True
# except:
#     ModelData_found = False
# try:
#     if not ModelData_found:
#         from QATCH.models.ModelData import ModelData
#     ModelData_found = True
# except:
#     ModelData_found = False
# if not ModelData_found:
#     raise ImportError("Cannot find 'ModelData' in any expected location.")

QDataPipeline_found = False
try:
    if not QDataPipeline_found:
        from QDataPipeline import QDataPipeline
    QDataPipeline_found = True
except:
    QDataPipeline_found = False
try:
    if not QDataPipeline_found:
        from QModel.QDataPipeline import QDataPipeline
    QDataPipeline_found = True
except:
    QDataPipeline_found = False
try:
    if not QDataPipeline_found:
        from QATCH.QModel.QDataPipeline import QDataPipeline
    QDataPipeline_found = True
except:
    QDataPipeline_found = False
if not QDataPipeline_found:
    raise ImportError("Cannot find 'QDataPipeline' in any expected location.")


class QMultiModel:
    """
    A class used to represent and manage an XGBoost model for multi-target classification tasks.

    This class handles the initialization, training, hyperparameter tuning, and management of an XGBoost model.
    It supports multi-target classification with specific hyperparameters and uses cross-validation to optimize
    the model's performance.

    Attributes:
        __params__ (dict): A dictionary of hyperparameters used for training the XGBoost model.
        __train_df__ (pd.DataFrame): Training subset of the dataset.
        __valid_df__ (pd.DataFrame): Validation subset of the dataset.
        __test_df__ (pd.DataFrame): Test subset of the dataset.
        __dtrain__ (xgb.DMatrix): DMatrix object for the training data.
        __dvalid__ (xgb.DMatrix): DMatrix object for the validation data.
        __dtest__ (xgb.DMatrix): DMatrix object for the test data.
        __watchlist__ (list): List of DMatrix objects to monitor during training, containing tuples of the form (DMatrix, "name").
        __model__ (xgb.Booster or None): The trained XGBoost model, initially set to None.

    Methods:
        __init__(self, dataset, predictors, target_features):
            Initializes the XGBoost model with the specified dataset, predictors, and target features.

        train_model(self):
            Trains the multi-target XGBoost model using the training dataset.

        objective(self, params):
            Evaluates the performance of the XGBoost model using cross-validation and returns the best AUC score.

        tune(self, evaluations=250):
            Tunes the XGBoost model's hyperparameters using Bayesian optimization with Tree-structured Parzen Estimator (TPE).

        save_model(self, model_name="QMultiModel"):
            Saves the trained XGBoost model to a specified file.

        get_model(self):
            Retrieves the trained XGBoost model.
    """

    def __init__(
        self,
        dataset: pd.DataFrame = None,
        predictors: list = None,
        target_features: str = "Class",
    ) -> None:
        """
        Initializes the XGBoost model with the specified dataset, predictors, and target features.

        Args:
            dataset (pd.DataFrame): The complete dataset containing both predictors and target features.
            predictors (list[str]): A list of column names in `dataset` that will be used as features for model training.
            target_features (list[str]): A list of column names in `dataset` that will be used as target variables for the model.

        Attributes:
            __params__ (dict): A dictionary of hyperparameters used for training the XGBoost model. These include:
                - objective (str): The learning task and objective ("multi:softprob" for multi-class classification).
                - eval_metric (str): Evaluation metric ("auc" for Area Under the Curve).
                - eta (float): Step size shrinkage used in updates to prevent overfitting (0.175).
                - max_depth (int): Maximum depth of a tree (5).
                - min_child_weight (float): Minimum sum of instance weight (hessian) needed in a child (4.0).
                - subsample (float): Subsample ratio of the training instances (0.6).
                - colsample_bytree (float): Subsample ratio of columns when constructing each tree (0.75).
                - gamma (float): Minimum loss reduction required to make a further partition on a leaf node (0.8).
                - nthread (int): Number of threads used for training (NUM_THREADS).
                - booster (str): Type of booster to use ("gbtree").
                - device (str): Device to run on ("cuda").
                - tree_method (str): Tree construction algorithm ("auto").
                - sampling_method (str): Sampling method ("gradient_based").
                - seed (int): Random seed for reproducibility (SEED).
                - num_class (int): Number of classes (7).

            __train_df__ (pd.DataFrame): Training subset of the dataset.
            __valid_df__ (pd.DataFrame): Validation subset of the dataset.
            __test_df__ (pd.DataFrame): Test subset of the dataset.

            __dtrain__ (xgb.DMatrix): DMatrix object for the training data.
            __dvalid__ (xgb.DMatrix): DMatrix object for the validation data.
            __dtest__ (xgb.DMatrix): DMatrix object for the test data.

            __watchlist__ (list): List of DMatrix objects to watch during training, containing tuples of the form (DMatrix, "name").

            __model__ (xgb.Booster or None): The trained XGBoost model, initially set to None.
        """
        self.__params__ = {
            "objective": "multi:softprob",
            "eval_metric": "auc",
            "eta": 0.175,
            "max_depth": 5,
            "min_child_weight": 4.0,
            "subsample": 0.6,
            "colsample_bytree": 0.75,
            "gamma": 0.8,
            "nthread": NUM_THREADS,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "auto",
            "sampling_method": "gradient_based",
            "seed": SEED,
            # "multi_strategy": "multi_output_tree",
            "num_class": 7,
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
            data=self.__train_df__[predictors],
            label=self.__train_df__[target_features].values,
        )
        self.__dvalid__ = xgb.DMatrix(
            data=self.__valid_df__[predictors],
            label=self.__valid_df__[target_features].values,
        )
        self.__dtest__ = xgb.DMatrix(
            data=self.__test_df__[predictors],
            label=self.__test_df__[target_features].values,
        )

        self.__watchlist__ = [
            (self.__dtrain__, "train"),
            (self.__dvalid__, "valid"),
        ]

        self.__model__ = None

    def train_model(self) -> None:
        """
        Trains the multi-target XGBoost model using the training dataset.

        This method initializes the training process for the XGBoost model with the previously defined parameters and datasets.
        The model is trained over a number of rounds with early stopping if the performance does not improve on the validation set.

        During the training process, the model's performance is evaluated on both the training and validation datasets,
        and the best model based on the validation performance is saved.

        Prints a status message indicating the start of the training process.

        Attributes:
            __model__ (xgb.Booster): The trained XGBoost model after completion of the training process.

        Raises:
            ValueError: If any of the necessary parameters or datasets have not been initialized prior to calling this method.
        """
        print(f"[STATUS] Training multi-target model")
        self.__model__ = xgb.train(
            self.__params__,
            self.__dtrain__,
            MAX_ROUNDS,
            evals=self.__watchlist__,
            early_stopping_rounds=EARLY_STOP,
            maximize=True,
            verbose_eval=VERBOSE_EVAL,
        )

    def update_model(self) -> None:
        # TODO: Implement functionality to pass a model path and load it then update the model with a new dataset.
        print(f"[STATUS] Updating multi-target model")
        pass

    def objective(self, params: dict = None) -> None:
        """
        Evaluates the performance of the XGBoost model using cross-validation and returns the best AUC score.

        This method performs k-fold cross-validation on the training dataset using the provided hyperparameters.
        The method returns the best AUC score from the cross-validation process, which can be used as the objective function
        in hyperparameter optimization.

        Args:
            params (dict): Dictionary of hyperparameters to be used in the XGBoost model during cross-validation.

        Returns:
            dict: A dictionary containing the following keys:
                - "loss" (float): The best mean AUC score on the test set obtained during cross-validation.
                - "status" (str): The status of the evaluation, typically "ok".

        Raises:
            ValueError: If any of the necessary parameters or datasets have not been initialized prior to calling this method.
        """
        results = xgb.cv(
            params,
            self.__dtrain__,
            MAX_ROUNDS,
            nfold=NUMBER_KFOLDS,
            stratified=True,
            early_stopping_rounds=20,
            metrics=["auc"],
            verbose_eval=VERBOSE_EVAL,
        )
        best_score = results["test-auc-mean"].max()
        return {"loss": best_score, "status": STATUS_OK}

    def tune(self, evaluations: int = 250) -> dict:
        """
        Tunes the XGBoost model's hyperparameters using Bayesian optimization with Tree-structured Parzen Estimator (TPE).

        This method runs the hyperparameter tuning process for a specified number of iterations.
        It searches for the best hyperparameters within the defined search space by minimizing the loss function.
        The search is based on the performance of the model as evaluated by cross-validation.

        Args:
            evaluations (int, optional): The maximum number of iterations for hyperparameter optimization. Default is 250.

        Returns:
            dict: A dictionary of the best hyperparameters found during the tuning process.

        Raises:
            ValueError: If any of the necessary parameters or datasets have not been initialized prior to calling this method.

        Example:
            best_hyperparams = self.tune(evaluations=300)
        """
        print(
            f"[STATUS] Running model tuning for {evaluations} max iterations")
        space = {
            "max_depth": hp.choice("max_depth", np.arange(1, 20, 1, dtype=int)),
            "eta": hp.uniform("eta", 0, 1),
            "gamma": hp.uniform("gamma", 0, 10e1),
            "reg_alpha": hp.uniform("reg_alpha", 10e-7, 10),
            "reg_lambda": hp.uniform("reg_lambda", 0, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "colsample_bynode": hp.uniform("colsample_bynode", 0.5, 1),
            "colsample_bylevel": hp.uniform("colsample_bylevel", 0.5, 1),
            "min_child_weight": hp.choice(
                "min_child_weight", np.arange(1, 10, 1, dtype="int")
            ),
            "max_delta_step": hp.choice(
                "max_delta_step", np.arange(1, 10, 1, dtype="int")
            ),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "eval_metric": "auc",
            "objective": "multi:softprob",
            "nthread": NUM_THREADS,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "auto",
            "sampling_method": "gradient_based",
            "seed": SEED,
            # "multi_strategy": "multi_output_tree",
            "num_class": 7,
        }
        trials = Trials()
        best_hyperparams = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=evaluations,
            trials=trials,
            return_argmin=False,
            early_stop_fn=no_progress_loss(10),
        )

        self.__params__ = best_hyperparams.copy()
        if "eval_metric" in self.__params__:
            self.__params__ = {
                key: self.__params__[key]
                for key in self.__params__
                if key != "eval_metric"
            }

        print(f"-- best parameters, \n\t{best_hyperparams}")

        return best_hyperparams

    def save_model(self, model_name: str = "QMultiModel") -> None:
        """
        Saves the trained XGBoost model to a specified file.

        This method saves the current XGBoost model in JSON format to the specified location.
        The model is saved in the "QModel/SavedModels/" directory with the provided model name.

        Args:
            model_name (str, optional): The name of the model file to be saved. Default is "QMultiModel".

        Returns:
            None

        Example:
            self.save_model(model_name="MyModel")
        """
        filename = f"QModel/SavedModels/{model_name}.json"
        print(f"[INFO] Saving model {model_name}")
        self.__model__.save_model(filename)

    def get_model(self) -> xgb.Booster:
        """
        Retrieves the trained XGBoost model.

        This method returns the trained XGBoost model, which can be used for further predictions or analysis.

        Returns:
            xgb.Booster: The trained XGBoost model.

        Example:
            model = self.get_model()
        """
        return self.__model__


class QPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            raise ValueError("[QModelPredict __init__()] No model path given")

        # Load the pre-trained models from the specified paths
        self.__model__ = xgb.Booster()

        self.__model__.load_model(model_path)
        with open("QModel/SavedModels/label_0.pkl", "rb") as file:
            self.__label_0__ = pickle.load(file)
        with open("QModel/SavedModels/label_1.pkl", "rb") as file:
            self.__label_1__ = pickle.load(file)
        with open("QModel/SavedModels/label_2.pkl", "rb") as file:
            self.__label_2__ = pickle.load(file)

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def compute_bounds(self, indices):
        bounds = []
        start = indices[0]
        end = indices[0]

        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                end = indices[i]
            else:
                bounds.append((start, end))
                start = indices[i]
                end = indices[i]

        # Append the last run
        bounds.append((start, end))

        return bounds

    def extract_results(self, results):

        num_indices = len(results[0])
        extracted = [[] for _ in range(num_indices)]

        for sublist in results:
            for idx in range(num_indices):
                extracted[idx].append(sublist[idx])

        return extracted

    def adjust_predictions(self, prediction, rel_time, poi_num, type, i, j):
        rel_time = rel_time[i:j]
        rel_time_norm = self.normalize(rel_time)
        bounds = None

        if type == 0:
            bounds = self.__label_0__["Class_" + str(poi_num)]
        elif type == 1:
            bounds = self.__label_1__["Class_" + str(poi_num)]
        elif type == 2:
            bounds = self.__label_2__["Class_" + str(poi_num)]

        lq = bounds["lq"]
        uq = bounds["uq"]
        adj = np.where((rel_time_norm >= lq) & (rel_time_norm <= uq), 1, 0)
        adjustment = np.concatenate(
            (np.zeros(i), np.array(adj), (np.zeros(len(prediction) - j)))
        )

        adj_prediction = prediction * adjustment
        lq_idx = next((i for i, x in enumerate(adj) if x == 1), -1) + i
        uq_idx = next((i for i, x in reversed(
            list(enumerate(adj))) if x == 1), -1) + i
        return adj_prediction, (lq_idx, uq_idx)

    def find_and_sort_peaks(self, signal):
        # Find peaks
        peaks, properties = find_peaks(signal)
        # Get the peak heights
        peak_heights = []
        for p in peaks:
            peak_heights.append(signal[p])

        # Sort peaks by height in descending order
        sorted_indices = np.argsort(peak_heights)[::-1]
        sorted_peaks = peaks[sorted_indices]
        return sorted_peaks

    def peak_density(self, signal, peaks, segment_le):

        density = len(peaks) / len(signal)

        return density

    def adjustment_poi_2(self, df, candidates, guess, emp, actual, bounds):
        # CONSIDER JUST ADJUSTING POI 1-3 for bad examples only.  Very good fo good examples.
        # diss = df["Dissipation"]
        # rf = df["Resonance_Frequency"]
        # diff = df["Difference"]
        # diss_peaks, _ = find_peaks(diss)
        # rf_peaks, _ = find_peaks(rf)
        # diff_peaks, _ = find_peaks(diff)
        # initial_guess = np.array(guess)
        # candidate_points = np.array(candidates)
        # rf_points = np.array(rf_peaks)
        # diss_points = np.array(diss_peaks)
        # diff_points = np.array(diff_peaks)
        # x_min, x_max = bounds
        # candidates = np.append(candidates, guess)
        # candidate_density = len(candidates) / (x_max - x_min)
        # fig, ax = plt.subplots()
        # ax.plot(diff, color="grey")
        # ax.fill_betweenx(
        #     [0, max(diss)], bounds[0], bounds[1], color=f"yellow", alpha=0.5
        # )
        # ax.axvline(guess, color="green", linestyle='dotted', label="guess")
        # ax.axvline(emp, color="purple", label="emp")
        # ax.axvline(actual, color="orange", linestyle="--", label="actual")
        # # ax.axvline(adjusted_point, color="brown", label="adjusted")
        # plt.legend()
        # plt.show()
        return np.average((guess, emp))

    def adjustmet_poi_4(self, df, candidates, guess, actual, bounds):
        diss = df["Dissipation"]
        rf = df["Resonance_Frequency"]
        diff = df["Difference"]
        diss_peaks, _ = find_peaks(diss)
        rf_peaks, _ = find_peaks(rf)
        diff_peaks, _ = find_peaks(diff)
        initial_guess = np.array(guess)
        candidate_points = np.array(candidates)
        rf_points = np.array(rf_peaks)
        diss_points = np.array(diss_peaks)
        diff_points = np.array(diff_peaks)
        x_min, x_max = bounds
        candidates = np.append(candidates, guess)
        candidate_density = len(candidates) / (x_max - x_min)
        if candidate_density < 0.02 or (guess < x_min or guess > x_max):
            # Filter RF points within the bounds
            rf_points = np.concatenate((rf_points, diss_points))
            within_bounds = (rf_points >= x_min) & (rf_points <= x_max)
            filtered_rf_points = rf_points[within_bounds]

            # If no RF points within the bounds, return None or handle accordingly
            if filtered_rf_points.size == 0:
                print("[INFO] No RF Peaks found")
                return guess

            # Calculate proximity weight for each RF point based on diff and diss points
            def calculate_weight(rf_point):
                multiplier = 2
                diff_in_proximity = np.sum(
                    np.abs(np.array(diff_points) - rf_point) < 0.02 * len(rf)
                )
                weight = diff_in_proximity

                # Apply multiplier if a diss point is nearby
                if np.any(np.abs(np.array(diss_points) - rf_point) < 0.02 * len(rf)):
                    weight *= multiplier
                return weight

            # Calculate weights for filtered RF points
            weights = np.array(
                [calculate_weight(rf_point) for rf_point in filtered_rf_points]
            )

            # Calculate weighted distances from initial guess to each filtered RF point
            distances_to_rf = np.abs(filtered_rf_points - initial_guess)
            weighted_distances = distances_to_rf / weights  # Adjust distance by weight

            # Find the closest RF point to the initial guess, considering weights
            closest_rf_idx = np.argmin(weighted_distances)
            closest_rf_point = filtered_rf_points[closest_rf_idx]

            # Calculate distances from candidate points to the closest RF point
            distances_to_closest_rf = np.abs(
                candidate_points - closest_rf_point)

            # Find the closest candidate point to the closest RF point
            closest_candidate_idx = np.argmin(distances_to_closest_rf)
            adjusted_point = candidate_points[closest_candidate_idx]
            # fig, ax = plt.subplots()
            # ax.plot(diss, color="grey")
            # ax.fill_betweenx(
            #     [0, max(diss)], bounds[0], bounds[1], color=f"yellow", alpha=0.5
            # )
            # # ax.scatter(diss_peaks, diss[diss_peaks], color="red", label="diss peaks")
            # ax.scatter(diff_peaks, diss[diff_peaks], color="green", label="diff peaks")
            # ax.scatter(rf_points, diss[rf_points], color="blue", label="rf peaks")
            # ax.scatter(candidates, diss[candidates], color="black", label="candidates")
            # ax.axvline(guess, color="orange", label="guess")
            # ax.axvline(actual, color="orange", linestyle="--", label="actual")
            # ax.axvline(adjusted_point, color="brown", label="adjusted")
            # plt.legend()
            # plt.show()
            return adjusted_point
        else:
            # fig, ax = plt.subplots()
            # ax.plot(diss, color="grey")
            # ax.fill_betweenx(
            #     [0, max(diss)], bounds[0], bounds[1], color=f"yellow", alpha=0.5
            # )
            # # ax.scatter(diss_peaks, diss[diss_peaks], color="red", label="diss peaks")
            # ax.scatter(diff_peaks, diss[diff_peaks], color="green", label="diff peaks")
            # ax.scatter(rf_peaks, diss[rf_peaks], color="blue", label="rf peaks")
            # ax.scatter(candidates, diss[candidates], color="black", label="candidates")
            # ax.axvline(guess, color="orange", label="guess")
            # ax.axvline(actual, color="orange", linestyle="--", label="actual")
            # plt.legend()
            # plt.show()
            return guess

    def find_zero_slope_regions(self, data, threshold=1e-3, min_region_length=1):
        # Calculate the first derivative (approximate slope)
        slopes = np.diff(data)

        # Identify regions where the slope is approximately zero
        zero_slope_mask = np.abs(slopes) < threshold

        # Find consecutive regions of near-zero slope
        regions = []
        start_idx = None

        for i, is_zero_slope in enumerate(zero_slope_mask):
            if is_zero_slope and start_idx is None:
                start_idx = i
            elif not is_zero_slope and start_idx is not None:
                if i - start_idx >= min_region_length:
                    # Store the region as (start, end)
                    regions.append((start_idx, i))
                start_idx = None

        # Handle case where the last region extends to the end of the array
        if start_idx is not None and len(data) - start_idx - 1 >= min_region_length:
            regions.append((start_idx, len(data) - 1))

        return regions

    def adjustmet_poi_5(self, df, candidates, emp_guess, guess, actual, bounds):
        diss = df["Dissipation"]
        rf = df["Resonance_Frequency"]

        diff = df["Difference"]

        diss_peaks, _ = find_peaks(diss)
        rf_peaks, _ = find_peaks(rf)
        diff_peaks, _ = find_peaks(diff)
        initial_guess = np.array(guess)
        candidate_points = np.array(candidates)
        rf_points = np.array(rf_peaks)

        diss_points = np.array(diss_peaks)
        diff_points = np.array(diff_peaks)
        np.concatenate((rf_points, diff_points, diss_points))
        x_min, x_max = bounds
        candidate_density = len(candidates) / (x_max - x_min)
        if candidate_density < 0.01:
            zero_slope = self.find_zero_slope_regions(rf)
            # Filter RF points within the bounds
            # rf_points = np.concatenate((rf_points, diss_points))
            candidates = np.append(candidates, guess)
            within_bounds = (rf_points >= x_min) & (rf_points <= x_max)
            filtered_rf_points = rf_points[within_bounds]

            for (l, r) in zero_slope:
                np.append(rf_points, l)
                np.append(rf_points, r)
            # If no RF points within the bounds, return None or handle accordingly
            if filtered_rf_points.size == 0:
                print("[INFO] No RF Peaks found")
                return guess

            # Calculate proximity weight for each RF point based on diff and diss points
            def calculate_weight(rf_point):
                # multiplier = 2
                # diff_in_proximity = np.sum(
                #     np.abs(np.array(diff_points) - rf_point) < 0.02 * len(rf)
                # )
                # weight = diff_in_proximity

                # # Apply multiplier if a diss point is nearby
                # if np.any(np.abs(np.array(diss_points) - rf_point) < 0.02 * len(rf)):
                #     weight *= multiplier
                return 1

            # Calculate weights for filtered RF points
            weights = np.array(
                [calculate_weight(rf_point) for rf_point in filtered_rf_points]
            )

            # Calculate weighted distances from initial guess to each filtered RF point
            distances_to_rf = np.abs(filtered_rf_points - initial_guess)
            weighted_distances = distances_to_rf / weights  # Adjust distance by weight

            # Find the closest RF point to the initial guess, considering weights
            closest_rf_idx = np.argmin(weighted_distances)
            closest_rf_point = filtered_rf_points[closest_rf_idx]

            # Calculate distances from candidate points to the closest RF point
            distances_to_closest_rf = np.abs(
                candidate_points - closest_rf_point)

            # Find the closest candidate point to the closest RF point
            closest_candidate_idx = np.argmin(distances_to_closest_rf)
            adjusted_point = candidate_points[closest_candidate_idx]
            # fig, ax = plt.subplots()
            # ax.plot(diss, color="grey")
            # ax.fill_betweenx(
            #     [0, max(diss)], bounds[0], bounds[1], color=f"yellow", alpha=0.5
            # )
            # ax.scatter(diss_peaks, diss[diss_peaks],
            #         color="red", label="diss peaks")
            # ax.scatter(diff_peaks, diss[diff_peaks],
            #         color="green", label="diff peaks")
            # ax.scatter(rf_points, diss[rf_points], color="blue", label="rf peaks")
            # ax.scatter(
            #     emp_guess, diss[emp_guess], color="pink", marker="x", label="emp guess"
            # )
            # ax.scatter(candidates, diss[candidates],
            #         color="black", label="candidates")
            # ax.axvline(guess, color="purple", linestyle='dotted', label="guess")
            # ax.axvline(actual, color="orange", linestyle="--", label="actual")
            # ax.axvline(adjusted_point, color="brown", label="adjusted")
            # plt.legend()
            # plt.show()
            return adjusted_point
        else:
            return guess

    def predict(self, file_buffer, type=-1, start=-1, stop=-1, act=None):
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
                # reset ByteIO buffer to beginning of stream
                file_buffer_2.seek(0)
            else:
                # ERROR: 'file_buffer_2' must be 'BytesIO' type here, but it's not seekable!
                raise Exception(
                    "Cannot 'seek' stream prior to passing to 'QDataPipeline'."
                )
        else:
            # Assuming 'file_buffer_2' is a string to a file path, this will work fine as-is
            pass
        if not isinstance(file_buffer, str):
            csv_headers = next(file_buffer)

            if isinstance(csv_headers, bytes):
                csv_headers = csv_headers.decode()

            if "Ambient" in csv_headers:
                csv_cols = (2, 4, 6, 7)
            else:
                csv_cols = (2, 3, 5, 6)

            file_data = np.loadtxt(
                file_buffer.readlines(), delimiter=",", skiprows=0, usecols=csv_cols
            )
            data_path = "QModel Passthrough"
            relative_time = file_data[:, 0]
            # temperature = file_data[:,1]
            resonance_frequency = file_data[:, 2]
            data = file_data[:, 3]

            emp_predictions = ModelData().IdentifyPoints(
                data_path=data_path,
                times=relative_time,
                freq=resonance_frequency,
                diss=data,
            )
        else:
            emp_predictions = ModelData().IdentifyPoints(file_buffer)
        emp_points = []
        start_bound = -1
        if isinstance(emp_predictions, list):
            for pt in emp_predictions:
                if isinstance(pt, int):
                    emp_points.append(pt)
                elif isinstance(pt, list):
                    max_pair = max(pt, key=lambda x: x[1])
                    emp_points.append(max_pair[0])
            start_bound = emp_points[0]
        ############################
        # MAIN PREDICTION PIPELINE #
        ############################
        # Process data using QDataPipeline
        qdp = QDataPipeline(file_buffer_2)
        df2 = qdp.__dataframe__
        rel_time = qdp.__dataframe__["Relative_time"]
        qdp.preprocess(poi_filepath=None)
        df = qdp.get_dataframe()
        inc_regions = qdp.common_increases()
        zeroes = qdp.common_zeroes()

        f_names = self.__model__.feature_names
        df = df[f_names]
        d_data = xgb.DMatrix(df)

        results = self.__model__.predict(d_data)
        normalized_results = self.normalize(results)
        extracted_results = self.extract_results(normalized_results)

        # extracted_1 = emp_points[0]
        extracted_1 = np.argmax(extracted_results[1])
        extracted_2 = np.argmax(extracted_results[2])
        extracted_3 = np.argmax(extracted_results[3])
        extracted_4 = np.argmax(extracted_results[4])
        extracted_5 = np.argmax(extracted_results[5])
        extracted_6 = np.argmax(extracted_results[6])

        if not isinstance(emp_predictions, list):
            extracted_1 = np.argmax(extracted_results[1])

        if start > -1:
            extracted_1 = start
        if stop > -1:
            extracted_6 = stop

        adj_1 = emp_points[0]
        adj_2, bounds_2 = self.adjust_predictions(
            prediction=extracted_results[2],
            rel_time=rel_time,
            poi_num=2,
            type=type,
            i=start_bound,
            j=extracted_6,
        )
        adj_3, bounds_3 = self.adjust_predictions(
            prediction=extracted_results[3],
            rel_time=rel_time,
            poi_num=3,
            type=type,
            i=start_bound,
            j=extracted_6,
        )
        adj_4, bounds_4 = self.adjust_predictions(
            prediction=extracted_results[4],
            rel_time=rel_time,
            poi_num=4,
            type=type,
            i=start_bound,
            j=extracted_6,
        )
        adj_5, bounds_5 = self.adjust_predictions(
            prediction=extracted_results[5],
            rel_time=rel_time,
            poi_num=5,
            type=type,
            i=start_bound,
            j=extracted_6,
        )
        adj_6 = extracted_results[6]

        candidates_2 = self.find_and_sort_peaks(adj_2)
        candidates_3 = self.find_and_sort_peaks(adj_3)
        candidates_4 = self.find_and_sort_peaks(adj_4)
        candidates_5 = self.find_and_sort_peaks(adj_5)
        candidates_6 = self.find_and_sort_peaks(adj_6)
        poi_2 = self.adjustment_poi_2(
            df2, candidates_2, extracted_2, emp_points[1], act[1], bounds_2)
        poi_4 = self.adjustmet_poi_4(
            df2, candidates_4, extracted_4, act[3], bounds_4)
        poi_5 = self.adjustmet_poi_5(
            df2, candidates_5, extracted_5, emp_points[4], act[4], bounds_5
        )

        poi_1 = adj_1
        poi_2 = np.argmax(adj_2)
        poi_3 = np.argmax(adj_3)
        poi_6 = np.argmax(adj_6)
        pois = [
            poi_1,
            poi_2,
            poi_3,
            poi_4,
            poi_5,
            poi_6,
        ]
        candidates = [
            poi_1,
            candidates_2,
            candidates_3,
            candidates_4,
            candidates_5,
            candidates_6,
        ]
        return pois, candidates
