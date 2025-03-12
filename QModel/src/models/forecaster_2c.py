from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy.signal import hilbert, savgol_filter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

FEATURES = [
    'Dissipation',
    'Dissipation_rolling_mean',
    'Dissipation_rolling_median',
    'Dissipation_ewm',
    'Dissipation_rolling_std',
    'Dissipation_diff',
    'Dissipation_pct_change',
    'Dissipation_ratio_to_mean',
    'Dissipation_ratio_to_ewm',
    'Dissipation_envelope'
]
NUM_CLASSES = 4
TARGET = "Fill"
DOWNSAMPLE_FACTOR = 5
SPIN_UP_TIME = (1.2, 1.4)
BASELINE_WINDOW = 100


class QForecasterDataprocessor:
    @staticmethod
    def load_content(data_dir: str) -> list:
        """
        Walk through data_dir and return a list of tuples.
        Each tuple contains the path to a CSV file (excluding those ending in "_poi.csv" or "_lower.csv")
        and its corresponding POI file (with '_poi.csv' replacing '.csv').
        """
        print(f"[INFO] Loading content from {data_dir}")
        loaded_content = []
        for root, _, files in tqdm(os.walk(data_dir), desc='Loading files...'):
            for f in files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    poi_file = f.replace(".csv", "_poi.csv")
                    loaded_content.append(
                        (os.path.join(root, f), os.path.join(root, poi_file))
                    )
        return loaded_content

    @staticmethod
    def find_time_delta(df: pd.DataFrame) -> int:
        """
        Compute the first index at which the difference in Relative_time
        deviates significantly from its expanding rolling mean.
        Returns -1 if no significant change is found.
        """
        time_df = pd.DataFrame()
        time_df["Delta"] = df["Relative_time"].diff()
        threshold = 0.032
        rolling_avg = time_df["Delta"].expanding(min_periods=2).mean()
        time_df["Significant_change"] = (
            time_df["Delta"] - rolling_avg).abs() > threshold
        change_indices = time_df.index[time_df["Significant_change"]].tolist()
        return change_indices[0] if change_indices else -1

    @staticmethod
    def reassign_region(fill):
        """
        Reassign numeric fill values to string region labels.
        """
        if fill == 0:
            return 'no_fill'
        elif fill in [1, 2, 3]:
            return 'init_fill'
        elif fill == 4:
            return 'ch_1'
        elif fill == 5:
            return 'ch_2'
        elif fill == 6:
            return 'full_fill'
        else:
            return fill  # fallback if needed

    @staticmethod
    def compute_additional_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute additional features for the Dissipation column and live compute the difference curve.
        This includes rolling statistics, differences, ratios, signal envelope, and a Difference curve.
        """
        try:
            required_columns = ["Dissipation",
                                "Resonance_Frequency", "Relative_time"]
            if not all(column in df.columns for column in required_columns):
                raise ValueError(
                    f"Input DataFrame must contain the following columns: {required_columns}")

            window = 10
            span = 10
            run_length = len(df)
            # Ensure window_length is at least 3
            window_length = max(3, int(np.ceil(0.01 * run_length)))

            # Adjust window_length if it's larger than the Dissipation data size
            diss_length = len(df['Dissipation'])
            if window_length > diss_length:
                window_length = diss_length if diss_length % 2 == 1 else diss_length - 1

            polyorder = 2 if window_length > 2 else 1
            df['Dissipation'] = savgol_filter(
                df['Dissipation'].values, window_length=window_length, polyorder=polyorder)
            df['Dissipation_rolling_mean'] = df['Dissipation'].rolling(
                window=window, min_periods=1).mean()
            df['Dissipation_rolling_median'] = df['Dissipation'].rolling(
                window=window, min_periods=1).median()
            df['Dissipation_ewm'] = df['Dissipation'].ewm(
                span=span, adjust=False).mean()
            df['Dissipation_rolling_std'] = df['Dissipation'].rolling(
                window=window, min_periods=1).std()
            df['Dissipation_diff'] = df['Dissipation'].diff()
            df['Dissipation_pct_change'] = df['Dissipation'].pct_change()
            df['Dissipation_ratio_to_mean'] = df['Dissipation'] / \
                df['Dissipation_rolling_mean']
            df['Dissipation_ratio_to_ewm'] = df['Dissipation'] / \
                df['Dissipation_ewm']
            df['Dissipation_envelope'] = np.abs(
                hilbert(df['Dissipation'].values))

            # Compute difference curve
            xs = df["Relative_time"]
            i = next((x for x, t in enumerate(xs) if t > 0.5), None)
            j = next((x for x, t in enumerate(xs) if t > 2.5), None)

            if i is not None and j is not None:
                avg_resonance_frequency = df["Resonance_Frequency"][i:j].mean()
                avg_dissipation = df["Dissipation"][i:j].mean()

                df["ys_diss"] = (df["Dissipation"] -
                                 avg_dissipation) * avg_resonance_frequency / 2
                df["ys_freq"] = avg_resonance_frequency - \
                    df["Resonance_Frequency"]
                difference_factor = 1  # Default value; can be made dynamic
                df["Difference"] = df["ys_freq"] - \
                    difference_factor * df["ys_diss"]

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

            return df
        except Exception as e:
            raise RuntimeError(f"Error computing features: {e}")

    @staticmethod
    def _process_fill(df: pd.DataFrame, poi_file: str) -> pd.DataFrame:
        """
        Helper to process fill information from the poi_file.
        Reads the poi CSV (with no header) and adds a Fill column to df.
        If the poi file does not contain a header, the method treats the first column
        as change indices (adding 1 to Fill from that index onward).
        Optionally, if check_unique is True, the method returns None when the number
        of unique fill values is not 7.
        """
        fill_df = pd.read_csv(poi_file, header=None)
        if "Fill" in fill_df.columns:
            df["Fill"] = fill_df["Fill"]
        else:
            df["Fill"] = 0
            change_indices = sorted(fill_df.iloc[:, 0].values)
            for idx in change_indices:
                df.loc[idx:, "Fill"] += 1

        df["Fill"] = pd.Categorical(df["Fill"]).codes
        df["Fill"] = df["Fill"].apply(QForecasterDataprocessor.reassign_region)
        mapping = {'no_fill': 0, 'init_fill': 1,
                   'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
        df["Fill"] = df["Fill"].map(mapping)
        return df

    @staticmethod
    def load_and_preprocess_data(data_dir: str, num_datasets: int):
        runs = []
        content = QForecasterDataprocessor.load_content(data_dir)
        random.shuffle(content)

        if num_datasets < len(content):
            content = content[:num_datasets]

        for file, poi_file in content:
            df = pd.read_csv(file)
            required_cols = ["Relative_time", "Dissipation"]
            if df.empty or not all(col in df.columns for col in required_cols):
                continue

            df = df[required_cols]

            try:
                df = QForecasterDataprocessor._process_fill(
                    df, poi_file=poi_file)
            except FileNotFoundError:
                df = None
            if df is None:
                continue

            df = df[df["Relative_time"] >= random.uniform(
                SPIN_UP_TIME[0], SPIN_UP_TIME[1])]
            df = df.iloc[::DOWNSAMPLE_FACTOR]
            df = df.reset_index()
            init_fill_point = QForecasterDataprocessor.init_fill_point(
                df, BASELINE_WINDOW, 10)
            df = df.iloc[init_fill_point:]
            if df is None or df.empty:
                continue
            df.loc[df['Fill'] == 0, 'Fill'] = 1
            # Method 2 (in-place subtraction):
            df['Fill'] -= 1
            df = QForecasterDataprocessor.compute_additional_features(df)
            df.reset_index(inplace=True)

            runs.append(df)

        training_data = pd.concat(runs).sort_values(
            "Relative_time").reset_index(drop=True)
        training_data.drop(columns=['Relative_time'], inplace=True)
        return training_data

    @staticmethod
    def viterbi_decode(prob_matrix, transition_matrix):
        T, N = prob_matrix.shape
        dp = np.full((T, N), -np.inf)
        backpointer = np.zeros((T, N), dtype=int)
        dp[0, 0] = np.log(prob_matrix[0, 0])
        for t in range(1, T):
            for j in range(N):
                allowed_prev = [0] if j == 0 else [j-1, j]
                best_state = allowed_prev[0]
                best_score = dp[t-1, best_state] + \
                    np.log(transition_matrix[best_state, j])
                for i in allowed_prev:
                    if transition_matrix[i, j] <= 0:
                        continue
                    score = dp[t-1, i] + np.log(transition_matrix[i, j])
                    if score > best_score:
                        best_score = score
                        best_state = i
                dp[t, j] = np.log(prob_matrix[t, j]) + best_score
                backpointer[t, j] = best_state
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(dp[T-1])
        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]
        return best_path

    @staticmethod
    def compute_dynamic_transition_matrix(training_data, num_states=5, smoothing=1e-6):
        states = training_data["Fill"].values
        transition_counts = np.zeros((num_states, num_states))
        for i in range(num_states):
            transition_counts[i, i] = smoothing
            if i + 1 < num_states:
                transition_counts[i, i+1] = smoothing
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            if next_state == current_state or next_state == current_state + 1:
                transition_counts[current_state, next_state] += 1
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return transition_counts / row_sums

    @staticmethod
    def init_fill_point(
        df: pd.DataFrame, baseline_window: int = 10, threshold_factor: float = 3.0
    ) -> int:
        """
        Identify the first index in the Dissipation column where a significant increase
        occurs relative to the initial baseline noise.

        The baseline is estimated using the first `baseline_window` samples.
        A significant increase is defined as a Dissipation value exceeding:
            baseline_mean + threshold_factor * baseline_std

        Args:
            df (pd.DataFrame): DataFrame containing a 'Dissipation' column.
            baseline_window (int): Number of initial samples used to compute the baseline.
            threshold_factor (float): Multiplier for the baseline standard deviation.

        Returns:
            int: The index of the first significant increase, or -1 if not found.
        """
        if 'Resonance_Frequency' not in df.columns:
            raise ValueError(
                "Resonance_Frequency column not found in DataFrame.")
        if len(df) < baseline_window:
            return -1

        if df['Relative_time'].max() < 3.0:
            return -1

        baseline_values = df['Resonance_Frequency'].iloc[:baseline_window]
        baseline_mean = baseline_values.mean()
        baseline_std = baseline_values.std()
        threshold = baseline_mean - threshold_factor * baseline_std
        dissipation = df['Resonance_Frequency'].values

        for idx, value in enumerate(dissipation):
            if value < threshold:
                return idx
        return -1

    @staticmethod
    def ch1_point(
        df: pd.DataFrame,
        init_fill_idx: int,
        min_offset_ratio: float = 0.1
    ) -> int:
        """
        Find the index in the 'Dissipation' column where the slope first begins to decrease,
        provided that the candidate is at least min_offset_ratio (default 10%) ahead of the init_fill_idx.

        The slope is approximated by the difference between successive Dissipation values.
        We return the first index (after the offset) where the slope decreases compared to the previous slope,
        which indicates a turning point in the slope.

        Args:
            df (pd.DataFrame): DataFrame containing a 'Dissipation' column.
            init_fill_idx (int): Starting index from which to begin the search.
            min_offset_ratio (float): Minimum offset as a fraction of the remaining data length (default is 0.1).

        Returns:
            int: The index where the slope first decreases (i.e. the turning point), or -1 if none is found.
        """
        if 'Dissipation' not in df.columns:
            raise ValueError("Dissipation column not found in DataFrame.")

        dissipation = df['Dissipation'].values
        n = len(dissipation)

        if init_fill_idx < 0:
            return -1
        if init_fill_idx >= n - 1:
            raise ValueError(
                "init_fill_idx is out of bounds or too close to the end of the array.")
        # Ensure we start at least 10% of the remaining length ahead of init_fill_idx.
        min_idx = init_fill_idx + \
            max(1, int(min_offset_ratio * (n - init_fill_idx)))
        if min_idx >= n - 1:
            return -1

        # Compute the slope (first difference) array.
        slopes = np.diff(dissipation)

        # Iterate over candidate indices (using dissipation indices)
        # Note: slope[i] corresponds to the difference dissipation[i+1]-dissipation[i].
        for i in range(min_idx, n - 1):
            # Check if the slope decreases relative to the previous slope.
            if slopes[i] < slopes[i - 1]:
                # i is the index in dissipation where the slope starts to drop.
                return i

        return -1

    @staticmethod
    def ch2_point(df, init_fill_point, ch1_fill_point, rolling_window=20, std_threshold=1e-7):
        """
        Finds the index where ch2_fill_point (blue line) occurs after ch1_fill_point (red line).
        - ch2_fill_point is detected at the start of the next plateau after an increase.
        - Ensures it occurs near 3x the 'Relative_time' difference between init_fill_point and ch1_fill_point.
        - Checks if the dataframe has even reached the expected Relative_time before searching.

        Parameters:
            df (pd.DataFrame): Dataframe with 'Dissipation' and 'Resonance_Frequency'.
            init_fill_point (int): Index of init_fill_point (initial fill location).
            ch1_fill_point (int): Index of ch1_fill_point (red line).
            rolling_window (int): Window size for detecting plateaus.
            std_threshold (float): Standard deviation threshold for plateau detection.

        Returns:
            int: Index of ch2_fill_point (blue line) or -1 if not found.
        """
        if ch1_fill_point == -1 or init_fill_point == -1:
            print(
                "ch1_fill_point or init_fill_point not found. Cannot detect ch2_fill_point.")
            return -1

        # Compute expected Relative_time for ch2_fill_point
        time_diff = df.loc[ch1_fill_point, "Relative_time"] - \
            df.loc[init_fill_point, "Relative_time"]
        expected_relative_time = df.loc[ch1_fill_point,
                                        "Relative_time"] + 3 * time_diff

        # Check if the dataframe has reached the expected Relative_time
        max_relative_time = df["Relative_time"].max()
        if max_relative_time < expected_relative_time:
            print(
                f"Dataframe has not reached expected Relative_time {expected_relative_time:.2f}. Returning -1.")
            return -1

        # Compute rate of change and rolling statistics
        df["Dissipation_diff"] = df["Dissipation"].diff()
        df["Dissipation_rolling_std"] = df["Dissipation"].rolling(
            rolling_window).std()

        # Search AFTER ch1_fill_point
        post_ch1_df = df.loc[ch1_fill_point+1:].copy()

        # Find the end of the first plateau
        first_plateau_end = post_ch1_df[post_ch1_df["Dissipation_rolling_std"]
                                        < std_threshold].index.min()

        if first_plateau_end is None or pd.isna(first_plateau_end):
            print("No first plateau detected. Returning -1.")
            return df[df["Relative_time"] >= expected_relative_time].index.min()

        # Find the first significant increase after the plateau
        threshold_increase = post_ch1_df["Dissipation_diff"].abs().mean() * 5
        increase_index = post_ch1_df[post_ch1_df["Dissipation_diff"]
                                     > threshold_increase].index.min()

        if increase_index is None or pd.isna(increase_index) or increase_index < first_plateau_end:
            print(
                "No significant increase detected after the first plateau. Returning -1.")
            return df[df["Relative_time"] >= expected_relative_time].index.min()

        # Fix: Apply the filter BEFORE slicing to avoid reindexing warnings
        plateau_candidates = post_ch1_df[post_ch1_df["Dissipation_rolling_std"] < std_threshold]
        plateau_candidates = plateau_candidates.loc[increase_index:]

        if plateau_candidates.empty:
            print("No second plateau detected. Returning -1.")
            return df[df["Relative_time"] >= expected_relative_time].index.min()

        next_plateau_start = plateau_candidates.index.min()

        if next_plateau_start is None or pd.isna(next_plateau_start):
            print("No valid next plateau start detected. Returning -1.")
            return df[df["Relative_time"] >= expected_relative_time].index.min()
        # Ensure ch2_fill_point occurs near the expected Relative_time
        relative_time_diff = abs(
            df.loc[next_plateau_start, "Relative_time"] - expected_relative_time)

        # Define a tolerance window (adjustable)
        time_tolerance = time_diff * 0.5  # Allow 50% deviation from the expected time

        if relative_time_diff > time_tolerance:
            print(
                f"Detected ch2_fill_point at index {next_plateau_start}, but it's too far from expected time. Returning expected index.")

            return df[df["Relative_time"] >= expected_relative_time].index.min()

        return next_plateau_start


###############################################################################
# Trainer Class: Handles model training, hyperparameter tuning, and saving.
###############################################################################


class QForecasterTrainer:
    def __init__(self, features, target='Fill', save_dir=None):
        """
        Args:
            numerical_features (list): List of numerical feature names.
            target (str): Target column name.
            save_dir (str): Directory to save trained objects.
        """
        self.features = features
        self.target = target
        self.save_dir = save_dir

        self.model = None
        self.preprocessors = None
        self.params = None
        self.transition_matrix = None

    def _build_preprocessors(self, X):
        """Fit and return preprocessors for numerical data."""
        num_imputer = SimpleImputer(strategy='mean')
        X_num = num_imputer.fit_transform(X[self.features])
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
        preprocessors = {'num_imputer': num_imputer, 'scaler': scaler}
        X_processed = X_num
        return X_processed, preprocessors

    def _tune_parameters(self, dtrain, base_params, max_evals=10):
        """Tune XGBoost hyperparameters using Hyperopt."""
        space = {
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1)
        }

        def objective(hyperparams):
            params = base_params.copy()
            params['max_depth'] = int(hyperparams['max_depth'])
            params['min_child_weight'] = int(hyperparams['min_child_weight'])
            params['learning_rate'] = hyperparams['learning_rate']
            params['subsample'] = hyperparams['subsample']
            params['colsample_bytree'] = hyperparams['colsample_bytree']

            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=200,
                nfold=5,
                metrics={'auc'},
                early_stopping_rounds=10,
                seed=42,
                verbose_eval=False
            )
            best_score = cv_results['test-auc-mean'].max()
            best_rounds = len(cv_results)
            return {'loss': -best_score, 'status': STATUS_OK, 'num_rounds': best_rounds}

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest,
                    max_evals=max_evals, trials=trials)
        best['max_depth'] = int(best['max_depth'])
        best['min_child_weight'] = int(best['min_child_weight'])
        params = base_params.copy()
        params.update(best)
        best_trial = min(trials.results, key=lambda x: x['loss'])
        optimal_rounds = best_trial['num_rounds']
        return params, optimal_rounds

    def train_model_native(self, training_data, tune=True):
        """
        Train an XGBoost model on the provided training data.

        Returns:
            model: Trained XGBoost model.
            preprocessors (dict): Fitted preprocessors.
            params (dict): Model parameters.
        """
        features = self.features
        X = training_data[features].copy()
        y = training_data[self.target].values

        X_processed, preprocessors = self._build_preprocessors(X)
        dtrain = xgb.DMatrix(X_processed, label=y)

        base_params = {
            'objective': 'multi:softprob',
            'num_class': NUM_CLASSES,
            'eval_metric': 'auc',
            'seed': 42,
            'device': 'cuda',
        }

        if tune:
            params, optimal_rounds = self._tune_parameters(dtrain, base_params)
        else:
            params = base_params.copy()
            params.update({'max_depth': 5, 'learning_rate': 0.1})
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=200,
                nfold=5,
                metrics={'auc'},
                early_stopping_rounds=10,
                seed=42,
                verbose_eval=False
            )
            optimal_rounds = len(cv_results)

        model = xgb.train(params, dtrain, num_boost_round=optimal_rounds)
        return model, preprocessors, params

    def _apply_preprocessors(self, X, preprocessors):
        """
        Apply loaded preprocessors to input data.
        """
        X_copy = X.copy()
        X_num = preprocessors['num_imputer'].transform(
            X_copy[self.features])
        X_num = preprocessors['scaler'].transform(X_num)
        return X_num

    def predict_native(self, model: xgb.Booster, preprocessors, X):
        """
        Generate predictions using a loaded XGBoost model.
        """
        X_processed = self._apply_preprocessors(X, preprocessors)
        dmatrix = xgb.DMatrix(X_processed)
        return model.predict(dmatrix)

    def train_model(self, training_data, tune=True):

        self.transition_matrix = QForecasterDataprocessor.compute_dynamic_transition_matrix(
            training_data)

        self.model, self.preprocessors, self.params = self.train_model_native(
            training_data, tune=tune)
        print("[INFO] Base model training complete.")

    def save_models(self):
        """
        Save base models, meta classifier, preprocessors, parameters, and transition matrices.
        """
        if not self.save_dir:
            raise ValueError("Save directory not specified.")
        os.makedirs(self.save_dir, exist_ok=True)

        # Save short model and its objects.
        self.model.save_model(os.path.join(
            self.save_dir, "model.json"))
        with open(os.path.join(self.save_dir, "preprocessors.pkl"), "wb") as f:
            pickle.dump(self.preprocessors, f)
        with open(os.path.join(self.save_dir, "params.pkl"), "wb") as f:
            pickle.dump(self.params, f)
        with open(os.path.join(self.save_dir, "transition_matrix.pkl"), "wb") as f:
            pickle.dump(self.transition_matrix, f)

        print("[INFO] All models and associated objects have been saved successfully.")

    def load_models(self):
        if not self.save_dir:
            raise ValueError("Save directory not specified.")

        self.model = xgb.Booster()
        self.model.load_model(os.path.join(
            self.save_dir, "model.json"))
        with open(os.path.join(self.save_dir, "preprocessors.pkl"), "rb") as f:
            self.preprocessors = pickle.load(f)
        with open(os.path.join(self.save_dir, "params.pkl"), "rb") as f:
            self.params = pickle.load(f)
        with open(os.path.join(self.save_dir, "transition_matrix.pkl"), "rb") as f:
            self.transition_matrix = pickle.load(f)

        print("[INFO] All models and associated objects have been loaded successfully.")


class QForecasterPredictor:
    """Predictor class for QForecaster that handles model loading, data accumulation, and prediction.

    This class loads a pre-trained XGBoost model along with its associated preprocessors and transition
    matrix. It provides methods to apply preprocessing to input data, generate predictions using the
    model and Viterbi decoding, and maintain a history of predictions to assess stability.
    """

    def __init__(
        self,
        numerical_features: List[str] = FEATURES,
        target: str = 'Fill',
        save_dir: Optional[str] = None,
        batch_threshold: int = 60
    ) -> None:
        """Initializes the QForecasterPredictor.

        Args:
            numerical_features (List[str], optional): List of feature names used for numerical processing.
                Defaults to FEATURES.
            target (str, optional): The target variable name. Defaults to 'Fill'.
            save_dir (Optional[str], optional): Directory path from which to load model files and preprocessors.
                Defaults to None.
            batch_threshold (int, optional): The rate at which to process batches of data. Defaults to 60.
        """
        self.numerical_features: List[str] = numerical_features
        self.target: str = target
        self.save_dir: Optional[str] = save_dir
        self.batch_threshold: int = batch_threshold

        self.model: Optional[xgb.Booster] = None
        self.preprocessors: Optional[Dict[str, Any]] = None
        self.transition_matrix: Optional[np.ndarray] = None

        self.accumulated_data: Optional[pd.DataFrame] = None
        self.batch_num: int = 0
        self.prediction_history: Dict[int, List[Tuple[int, int]]] = {}

    def load_models(self) -> None:
        """Loads the XGBoost model, preprocessors, and transition matrix from the specified directory.

        The method expects to find:
          - "model.json" for the XGBoost model,
          - "preprocessors.pkl" for the preprocessing objects, and
          - "transition_matrix.pkl" for the transition matrix.

        Raises:
            ValueError: If `save_dir` is not specified.
        """
        if not self.save_dir:
            raise ValueError("Save directory not specified.")

        model_path = os.path.join(self.save_dir, "model.json")
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        preprocessors_path = os.path.join(self.save_dir, "preprocessors.pkl")
        with open(preprocessors_path, "rb") as f:
            self.preprocessors = pickle.load(f)

        transition_matrix_path = os.path.join(
            self.save_dir, "transition_matrix.pkl")
        with open(transition_matrix_path, "rb") as f:
            self.transition_matrix = pickle.load(f)

    def _apply_preprocessors(
        self, X: pd.DataFrame, preprocessors: Dict[str, Any]
    ) -> np.ndarray:
        """Applies the loaded preprocessors to the input data.

        The method copies the input DataFrame, extracts the numerical features, applies the numerical
        imputer, and then scales the features.

        Args:
            X (pd.DataFrame): Input DataFrame containing the numerical features.
            preprocessors (Dict[str, Any]): Dictionary of preprocessor objects. Must include:
                - 'num_imputer': for imputing missing values.
                - 'scaler': for scaling the data.

        Returns:
            np.ndarray: The preprocessed numerical features as a NumPy array.
        """
        X_copy = X.copy()
        X_num = preprocessors['num_imputer'].transform(
            X_copy[self.numerical_features])
        X_num = preprocessors['scaler'].transform(X_num)
        return X_num

    def predict_native(
        self, model: xgb.Booster, preprocessors: Dict[str, Any], X: pd.DataFrame
    ) -> np.ndarray:
        """Generates predictions using the provided XGBoost model.

        The input DataFrame is filtered to contain only numerical features, preprocessed, converted to an
        XGBoost DMatrix, and then used to produce predictions.

        Args:
            model (xgb.Booster): The pre-trained XGBoost model.
            preprocessors (Dict[str, Any]): Dictionary containing preprocessing objects.
            X (pd.DataFrame): Input DataFrame from which predictions are to be generated.

        Returns:
            np.ndarray: An array of prediction probabilities or scores.
        """
        X = X[self.numerical_features]
        X_processed = self._apply_preprocessors(X, preprocessors)
        dmatrix = xgb.DMatrix(X_processed)
        return model.predict(dmatrix)

    def reset_accumulator(self) -> None:
        """Resets the accumulated data and batch counter.

        This method clears the internal DataFrame that accumulates new data and resets the batch number to zero.
        """
        self.accumulated_data = None
        self.batch_num = 0

    def is_prediction_stable(
        self,
        pred_list: List[Tuple[int, int]],
        stability_window: int = 5,
        frequency_threshold: float = 0.8,
        location_tolerance: int = 0
    ) -> Tuple[bool, Optional[int]]:
        """Determines if the prediction is stable based on its occurrence frequency and location consistency.

        Each element in `pred_list` is a tuple consisting of (location, prediction). A prediction is deemed
        stable if, among the last `stability_window` entries:
          - It appears with a frequency at least equal to `frequency_threshold`, and
          - The difference between the maximum and minimum locations does not exceed `location_tolerance`.

        Args:
            pred_list (List[Tuple[int, int]]): List of tuples where each tuple is (location, prediction).
            stability_window (int, optional): Number of recent predictions to consider for stability. Defaults to 5.
            frequency_threshold (float, optional): Required frequency fraction for stability. Defaults to 0.8.
            location_tolerance (int, optional): Maximum allowed difference between locations for stability.
                Defaults to 0.

        Returns:
            Tuple[bool, Optional[int]]:
                - A boolean indicating if a stable prediction was found.
                - The stable prediction class if stable, otherwise None.
        """
        if len(pred_list) < stability_window:
            return False, None

        recent = pred_list[-stability_window:]
        groups: Dict[int, Dict[str, Any]] = {}
        for loc, pred in recent:
            if pred not in groups:
                groups[pred] = {'count': 0, 'locations': []}
            groups[pred]['count'] += 1
            groups[pred]['locations'].append(loc)

        for pred, data in groups.items():
            frequency = data['count'] / stability_window
            if frequency >= frequency_threshold:
                if max(data['locations']) - min(data['locations']) <= location_tolerance:
                    return True, pred
        return False, None

    def update_predictions(
        self,
        new_data: pd.DataFrame,
        stability_window: int = 5,
        frequency_threshold: float = 0.8,
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """Accumulates new data, generates predictions, and assesses prediction stability."""

        # Accumulate new data
        if self.accumulated_data is None:
            self.accumulated_data = pd.DataFrame(columns=new_data.columns)
        self.accumulated_data = pd.concat(
            [self.accumulated_data, new_data], ignore_index=True)
        current_count = len(self.accumulated_data)

        # Compute additional features
        self.accumulated_data = QForecasterDataprocessor.compute_additional_features(
            self.accumulated_data)

        # Identify initial fill points
        init_fill_point = QForecasterDataprocessor.init_fill_point(
            self.accumulated_data, baseline_window=100, threshold_factor=10)
        ch1_fill_point = QForecasterDataprocessor.ch1_point(
            self.accumulated_data, init_fill_point)
        ch2_fill_point = QForecasterDataprocessor.ch2_point(
            self.accumulated_data, init_fill_point, ch1_fill_point)

        # If no fill has started, return waiting status
        if init_fill_point == -1:
            return self._return_waiting_state(current_count)

        # Perform predictions on live data
        predictions, conf = self._generate_predictions(
            init_fill_point, ch1_fill_point, ch2_fill_point)

        # Update prediction history and check stability
        stable_predictions = self._check_stability(
            predictions, conf, init_fill_point, stability_window, frequency_threshold, confidence_threshold)

        self.batch_num += 1

        return {
            "status": "completed",
            "pred": predictions,
            "conf": conf,
            "stable_predictions": stable_predictions,
            "accumulated_data": self.accumulated_data,
            "accumulated_count": current_count,
            "init_fill_point": init_fill_point,
            "ch1_fill_point": ch1_fill_point,
            "ch2_fill_point": ch2_fill_point,
        }

    # ------------------ Helper Methods ------------------

    def _return_waiting_state(self, current_count: int) -> Dict[str, Any]:
        """Handles case when fill has not started."""
        predictions = np.zeros(current_count, dtype=int)
        conf = np.zeros(current_count)
        for i in range(current_count):
            if i not in self.prediction_history:
                self.prediction_history[i] = []
            self.prediction_history[i].append((i, 0))

        return {
            "status": "waiting",
            "pred": predictions,
            "conf": conf,
            "stable_predictions": {},
            "accumulated_data": self.accumulated_data,
            "accumulated_count": current_count
        }

    def _generate_predictions(self, init_fill_point: int, ch1_fill_point: int, ch2_fill_point: int):
        """Runs predictions and applies Viterbi decoding."""
        predictions = np.zeros(len(self.accumulated_data), dtype=int)
        conf = np.zeros(len(self.accumulated_data))

        # Apply no_fill (0) before init_fill_point
        predictions[:init_fill_point] = 0

        # Get live data for prediction
        X_live = self.accumulated_data[self.numerical_features].iloc[init_fill_point:]
        prob_matrix = self.predict_native(
            self.model, self.preprocessors, X_live)

        # Perform Viterbi decoding
        ml_pred = QForecasterDataprocessor.viterbi_decode(
            prob_matrix, self.transition_matrix)
        ml_pred_offset = ml_pred + 1  # Offset if needed

        predictions[init_fill_point:] = ml_pred_offset
        conf[init_fill_point:] = [prob_matrix[i, ml_pred[i]]
                                  for i in range(len(ml_pred))]

        # Adjust predictions based on fill points
        self._apply_fill_point_constraints(
            predictions, ch1_fill_point, ch2_fill_point)

        return predictions, conf

    def _apply_fill_point_constraints(self, predictions: np.ndarray, ch1_fill_point: int, ch2_fill_point: int):
        """Adjusts predictions based on fill points."""
        if ch1_fill_point > -1:
            predictions[ch1_fill_point:] = 2  # Assign 2 after ch1_fill_point
        if ch2_fill_point > -1:
            predictions[ch2_fill_point:] = 3  # Assign 3 after ch2_fill_point

    def _check_stability(self, predictions: np.ndarray, conf: np.ndarray, init_fill_point: int,
                         stability_window: int, frequency_threshold: float, confidence_threshold: float) -> Dict[int, int]:
        """Checks stability of predictions over a given window."""
        stable_predictions = {}

        for idx, p in enumerate(predictions[init_fill_point:], start=init_fill_point):
            if idx not in self.prediction_history:
                self.prediction_history[idx] = []
            self.prediction_history[idx].append((idx, p))

        for idx, history in self.prediction_history.items():
            stable, stable_class = self.is_prediction_stable(
                history, stability_window, frequency_threshold, int(confidence_threshold))
            if stable:
                stable_predictions[idx] = stable_class

        return stable_predictions

# -----------------------------
# Helper: Simulate Serial Stream
# -----------------------------


def simulate_serial_stream_from_loaded(loaded_data, batch_size=100):
    """
    Yields sequential batches of rows from the loaded data.
    """
    num_rows = len(loaded_data)
    for start_idx in range(0, num_rows, batch_size):
        yield loaded_data.iloc[start_idx:start_idx+batch_size]

# -----------------------------
# Live Prediction Simulator Class
# -----------------------------


def simulate_serial_stream_from_loaded(loaded_data):
    """
    Simulate a serial data stream by yielding random chunks from loaded_data.

    Each chunk will have a random size between 100 and 200 rows.

    Args:
        loaded_data (pd.DataFrame): The loaded dataset to stream.

    Yields:
        pd.DataFrame: A chunk of data with a random number of rows.
    """
    num_rows = len(loaded_data)
    start_idx = 0
    while start_idx < num_rows:
        batch_size = random.randint(10, 30)
        yield loaded_data.iloc[start_idx:start_idx + batch_size]
        start_idx += batch_size


class QForecasterSimulator:
    def __init__(self, predictor: QForecasterPredictor, dataset: pd.DataFrame, poi_file: pd.DataFrame = None, ignore_before=0, delay=1.0):
        """
        Simulator class to stream data in random chunk sizes and update predictions live.
        Args:
            predictor (QForecasterPredictor): Predictor object.
            dataset (pd.DataFrame): Dataset to simulate streaming.
            poi_file (pd.DataFrame, optional): DataFrame containing Points Of Interest indices.
            ignore_before (int): Number of initial rows to ignore.
            delay (float): Delay (in seconds) between processing batches.
        """
        self.predictor = predictor
        self.dataset = dataset
        self.ignore_before = ignore_before
        self.delay = delay

        # If a POI file is provided, extract the actual indices.
        if poi_file is not None:
            self.actual_poi_indices = poi_file
        else:
            self.actual_poi_indices = np.array([])

        # Setup a live plot with a single axis.
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.ion()  # Enable interactive mode.
        plt.show()

    def run(self):
        """
        Runs the simulation by iterating through the dataset in random chunk sizes,
        updating predictions, and plotting the current state.
        """
        self.predictor.reset_accumulator()
        batch_number = 0
        for batch_data in simulate_serial_stream_from_loaded(self.dataset):
            batch_number += 1
            print(
                f"[INFO] Processing batch {batch_number} with {len(batch_data)} rows.")

            # Update predictions using the new batch.
            results = self.predictor.update_predictions(batch_data)

            # Update the live plot.
            self.plot_results(results, batch_number=batch_number)

            # Simulate processing delay.
            plt.pause(self.delay)

        plt.ioff()  # Turn off interactive mode.
        plt.show()

    def plot_results(self, results: dict, batch_number: int):
        """
        Updates the live plot with the normalized dissipation curve overlaid with
        background shading corresponding to contiguous predicted class regions.
        """
        self.ax.cla()  # Clear previous plot

        accumulated_data = results.get("accumulated_data")
        if accumulated_data is None or accumulated_data.empty:
            print("[WARN] No accumulated data available for plotting.")
            return

        x_dissipation = np.arange(len(accumulated_data))

        if results.get("status") == "waiting":
            self.ax.set_title(
                f"Waiting for more data (Batch {batch_number})", fontsize=14)
            self.ax.plot(
                x_dissipation,
                accumulated_data["Dissipation"],
                linestyle=":",
                color="#333333",
                linewidth=1.5,
                label="Dissipation Curve",
            )
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            return

        init_fill_point = results["init_fill_point"]
        ch1_fill_point = results["ch1_fill_point"]
        ch2_fill_point = results["ch2_fill_point"]

        # Vertical lines for fill points
        if init_fill_point > -1:
            self.ax.axvline(init_fill_point, color="orange",
                            label="Initial fill location")
        if ch1_fill_point > -1:
            self.ax.axvline(ch1_fill_point, color="brown",
                            label="Channel 1 fill location")
        if ch2_fill_point > -1:
            self.ax.axvline(ch2_fill_point, color="red",
                            label="Channel 2 fill location")

        # Plot Dissipation Curve
        self.ax.plot(
            x_dissipation,
            accumulated_data["Difference"],
            linestyle=":",
            color="#333333",
            linewidth=1.5,
            label="Dissipation Curve",
        )

        preds = np.array(results["pred"])
        stable_preds = results.get("stable_predictions", {})

        # Define class name and color mappings
        class_names = {0: "no_fill", 1: "initial_fill",
                       2: "channel_1", 3: "channel_2", 4: "full_fill"}
        class_colors = {0: "#d3d3d3", 1: "#a6cee3",
                        2: "#b2df8a", 3: "#fdbf6f", 4: "#fb9a99"}

        # Overlay shaded regions for predicted class regions
        if len(preds) > 0:
            already_labeled = {}
            current_class = preds[0]
            start_idx = 0
            for i in range(1, len(preds)):
                if preds[i] != current_class:
                    end_idx = i - 1
                    label = class_names[current_class] if current_class not in already_labeled else None
                    already_labeled[current_class] = True
                    self.ax.axvspan(
                        start_idx,
                        end_idx,
                        color=class_colors.get(current_class, "#cccccc"),
                        alpha=0.3,
                        label=label,
                    )
                    current_class = preds[i]
                    start_idx = i
            # Shade the final segment
            end_idx = len(preds) - 1
            label = class_names[current_class] if current_class not in already_labeled else None
            self.ax.axvspan(
                start_idx,
                end_idx,
                color=class_colors.get(current_class, "#cccccc"),
                alpha=0.3,
                label=label,
            )

        # Scatter plot stable predictions
        if stable_preds:
            stable_indices = list(stable_preds.keys())
            stable_labels = list(stable_preds.values())

            # Ensure indices are within bounds
            stable_indices = [
                idx for idx in stable_indices if idx < len(accumulated_data)]
            stable_labels = [stable_preds[idx] for idx in stable_indices]

            if stable_indices:
                y_stable = accumulated_data["Dissipation"].iloc[stable_indices]
                self.ax.scatter(
                    stable_indices,
                    y_stable,
                    color="blue",
                    marker="s",
                    s=50,
                    label="Stable Predictions",
                    edgecolor="black",
                    linewidth=0.8,
                )

        # Scatter actual POI indices
        if self.actual_poi_indices.size > 0:
            valid_actual_indices = [
                int(idx) for idx in self.actual_poi_indices if idx < len(accumulated_data)]
            if valid_actual_indices:
                y_actual = accumulated_data["Dissipation"].iloc[valid_actual_indices]
                self.ax.scatter(
                    valid_actual_indices,
                    y_actual,
                    color="#2ca02c",
                    marker="o",
                    s=50,
                    label=f"Actual POI {self.actual_poi_indices.size}",
                )

        self.ax.set_title(
            f"Dissipation Curve (Batch {batch_number})", fontsize=14, weight="medium")
        self.ax.set_xlabel("Data Index", fontsize=12)
        self.ax.set_ylabel(self.predictor.target, fontsize=12)
        self.ax.tick_params(axis="both", which="major", labelsize=10)
        self.ax.legend(frameon=False, fontsize=10)

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


TESTING = True
TRAINING = False
# Main execution block.
if __name__ == '__main__':
    SAVE_DIR = r'QModel\SavedModels\forecaster_v3'
    if TRAINING:
        train_dir = r'content\training_data'
        training_data = QForecasterDataprocessor.load_and_preprocess_data(
            train_dir, num_datasets=100)

        qft = QForecasterTrainer(FEATURES, TARGET, SAVE_DIR)
        qft.train_model(training_data=training_data, tune=True)
        qft.save_models()

    if TESTING:
        test_dir = r"content\test_data"
        test_content = QForecasterDataprocessor.load_content(test_dir)
        random.shuffle(test_content)
        for data_file, poi_file in test_content:
            dataset = pd.read_csv(data_file)
            end = np.random.randint(0, len(dataset))
            end = random.randint(min(end, len(dataset)),
                                 max(end, len(dataset)))
            random_slice = dataset.iloc[0:end]

            # Load the POI file which is a flat list of 6 indices from the original dataset.
            poi_df = pd.read_csv(poi_file, header=None)
            poi_indices_original = poi_df.to_numpy().flatten()

            # Use these indices to extract the corresponding relative times from the original dataset.
            original_relative_times = dataset.loc[poi_indices_original,
                                                  'Relative_time'].values

            predictor = QForecasterPredictor(
                FEATURES, target=TARGET, save_dir=SAVE_DIR)
            # Ensure models and preprocessors are loaded.
            predictor.load_models()

            delay = dataset['Relative_time'].max() / len(random_slice)
            random_slice = random_slice[random_slice["Relative_time"] >= random.uniform(
                SPIN_UP_TIME[0], SPIN_UP_TIME[1])]
            # Downsample the data.
            random_slice = random_slice.iloc[::DOWNSAMPLE_FACTOR]

            # Get the downsampled relative times.
            downsampled_times = random_slice["Relative_time"].values

            # For each original relative time, find the index in the downsampled data
            # that has the closest relative time.
            mapped_poi_indices = []
            for orig_time in original_relative_times:
                idx = np.abs(downsampled_times - orig_time).argmin()
                mapped_poi_indices.append(idx)
            mapped_poi_indices = np.array(mapped_poi_indices)

            # Create the simulator, passing the mapped POI indices.
            simulator = QForecasterSimulator(
                predictor,
                random_slice,
                poi_file=mapped_poi_indices,
                ignore_before=50,
                delay=1
            )

            # Run the simulation.
            simulator.run()
