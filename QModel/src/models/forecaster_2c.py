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
        Compute a series of additional features (e.g., rolling statistics,
        differences, ratios, and the signal envelope) for the Dissipation column.
        Also computes a 'Time_shift' column based on the first significant change in time.
        """
        window = 10
        span = 10
        run_length = len(df)
        window_length = int(np.ceil(0.01 * run_length))
        # Ensure window_length is odd and at least 3
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 3:
            window_length = 3
        polyorder = 2 if window_length > 2 else 1

        df['Dissipation'] = savgol_filter(df['Dissipation'].values,
                                          window_length=window_length,
                                          polyorder=polyorder)
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
        # df['Relative_time_diff'] = df['Relative_time'].diff().replace(0, np.nan)
        # df['Dissipation_rate'] = df['Dissipation_diff'] / df['Relative_time_diff']
        df['Dissipation_ratio_to_mean'] = df['Dissipation'] / \
            df['Dissipation_rolling_mean']
        df['Dissipation_ratio_to_ewm'] = df['Dissipation'] / df['Dissipation_ewm']
        df['Dissipation_envelope'] = np.abs(hilbert(df['Dissipation'].values))

        # Drop the Resonance_Frequency column if present.
        if 'Resonance_Frequency' in df.columns:
            df.drop(columns=['Resonance_Frequency'], inplace=True)

        return df

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
        if 'Dissipation' not in df.columns:
            raise ValueError("Dissipation column not found in DataFrame.")

        if len(df) < baseline_window:
            return -1

        baseline_values = df['Dissipation'].iloc[:baseline_window]
        baseline_mean = baseline_values.mean()
        baseline_std = baseline_values.std()
        threshold = baseline_mean + threshold_factor * baseline_std
        dissipation = df['Dissipation'].values
        for idx, value in enumerate(dissipation):
            if value > threshold:
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


###############################################################################
# Predictor Class: Handles generating predictions (including live predictions).
###############################################################################
class QForecasterPredictor:
    def __init__(self, numerical_features: list, target: str = 'Fill',
                 save_dir: str = None, batch_threshold: int = 60):
        self.numerical_features = numerical_features
        self.target = target
        self.save_dir = save_dir
        self.batch_threshold = batch_threshold

        self.model = None
        self.preprocessors = None
        self.transition_matrix = None

        self.accumulated_data = None
        self.batch_num = 0
        self.fig = None
        self.axes = None

        self.prediction_history = {}

    def load_models(self):
        if not self.save_dir:
            raise ValueError("Save directory not specified.")

        self.model = xgb.Booster()
        self.model.load_model(os.path.join(self.save_dir, "model.json"))
        with open(os.path.join(self.save_dir, "preprocessors.pkl"), "rb") as f:
            self.preprocessors = pickle.load(f)
        with open(os.path.join(self.save_dir, "transition_matrix.pkl"), "rb") as f:
            self.transition_matrix = pickle.load(f)

        print("[INFO] All models and associated objects have been loaded successfully.")

    def _apply_preprocessors(self, X, preprocessors):
        """
        Apply loaded preprocessors to input data.
        """
        X_copy = X.copy()
        X_num = preprocessors['num_imputer'].transform(
            X_copy[self.numerical_features])
        X_num = preprocessors['scaler'].transform(X_num)
        return X_num

    def predict_native(self, model, preprocessors, X):
        """
        Generate predictions using a loaded XGBoost model.
        """
        X = X[FEATURES]
        X_processed = self._apply_preprocessors(X, preprocessors)
        dmatrix = xgb.DMatrix(X_processed)
        return model.predict(dmatrix)

    def reset_accumulator(self):
        """
        Reset the internal accumulated dataframe and batch counter.
        """
        self.accumulated_data = None
        self.batch_num = 0

    def is_prediction_stable(self, pred_list, stability_window=5, frequency_threshold=0.8, confidence_threshold=0.9):
        """
        Determine if the predictions in pred_list are stable.

        Args:
            pred_list (list): A list of tuples (prediction, confidence) for recent updates.
            stability_window (int): The number of most recent predictions to consider.
            frequency_threshold (float): Fraction of predictions that must agree.
            confidence_threshold (float): Minimum average confidence required.

        Returns:
            (bool, stable_class): Tuple where bool indicates stability and stable_class is the converged prediction.
        """
        if len(pred_list) < stability_window:
            return False, None

        recent = pred_list[-stability_window:]
        counts = {}
        confidences = {}
        for pred, conf in recent:
            counts[pred] = counts.get(pred, 0) + 1
            confidences.setdefault(pred, []).append(conf)

        for pred, count in counts.items():
            if count / stability_window >= frequency_threshold:
                avg_conf = np.mean(confidences[pred])
                if avg_conf >= confidence_threshold:
                    return True, pred
        return False, None

    def enforce_fill_distribution(self, prob_matrix, transition_matrix):
        """
        Replaces the original boosting method.

        Steps:
          1. Use Viterbi decoding to obtain raw predictions (for the live data portion).
          2. Compute the baseline "init_fill" duration from the init_fill_point to the last contiguous
             prediction of raw class 1. This duration is defined as:

                 baseline_duration = (Relative_time at last raw 1 in the contiguous block)
                                    - (Relative_time at the init_fill_point)

          3. Enforce that any contiguous segment of raw class 1 (channel_1 fill) has a duration of at least
             4×baseline_duration, and any contiguous segment of raw class 2 (channel_2 fill) lasts at least
             9×baseline_duration.

        Returns:
            Tuple: (adjusted predictions, original probability matrix, transition_matrix)
        """
        # Obtain raw predictions via Viterbi decoding.
        pred = QForecasterDataprocessor.viterbi_decode(
            prob_matrix, transition_matrix)

        # Determine the init_fill_point from the accumulated data.
        init_fill_point = QForecasterDataprocessor.init_fill_point(
            self.accumulated_data, 100, threshold_factor=10)
        if init_fill_point == -1:
            return pred, prob_matrix, transition_matrix

        # Get the live data times corresponding to predictions (from init_fill_point onward).
        live_times = self.accumulated_data['Relative_time'].iloc[init_fill_point:].values

        # Ensure that there is at least one prediction.
        if len(pred) == 0:
            return pred, prob_matrix, transition_matrix

        # Compute the baseline duration from the init_fill block.
        # We define the baseline as the contiguous block of raw 1's starting at the beginning.
        if pred[0] != 1:
            # If the first live prediction isn't a 1, we cannot compute a baseline; return as is.
            return pred, prob_matrix, transition_matrix

        block_end = 0
        while block_end < len(pred) and pred[block_end] == 1:
            block_end += 1

        # Baseline duration is the difference between the first live time and the last time in the contiguous block of 1's.
        baseline_duration = live_times[block_end - 1] - live_times[0]
        # If baseline_duration is zero (or negative), skip enforcement.
        if baseline_duration <= 0:
            return pred, prob_matrix, transition_matrix

        # Set required durations:
        # For channel_1 fill (raw class 1).
        required_duration_ch1 = 4 * baseline_duration
        # For channel_2 fill (raw class 2).
        required_duration_ch2 = 9 * baseline_duration

        # Enforce the minimum duration requirements on live predictions.
        adjusted_pred = pred.copy()
        n = len(pred)
        i = 0
        while i < n:
            current_state = pred[i]
            start_time = live_times[i]
            j = i + 1
            # Find contiguous segment with the same raw prediction.
            while j < n and pred[j] == current_state:
                j += 1
            segment_duration = live_times[j - 1] - start_time

            if current_state == 1 and segment_duration < required_duration_ch1:
                # Extend the segment to reach the required duration.
                k = j
                while k < n and (live_times[k] - start_time) < required_duration_ch1:
                    adjusted_pred[k] = 1
                    k += 1
                i = k
            elif current_state == 2 and segment_duration < required_duration_ch2:
                k = j
                while k < n and (live_times[k] - start_time) < required_duration_ch2:
                    adjusted_pred[k] = 2
                    k += 1
                i = k
            else:
                i = j

        return adjusted_pred, prob_matrix, transition_matrix

    def update_predictions(self, new_data, stability_window=5,
                           frequency_threshold=0.8, confidence_threshold=0.9):
        """
        Accumulate new data and run predictions in batches.
        - Data before the init_fill_point are set to 0 (no_fill).
        - ML prediction is applied only to data from the init_fill_point onward.
        - Raw ML predictions (0-3) are adjusted via Viterbi decoding and the fill distribution enforcement.
        - Final predictions are offset by 1 (yielding classes 1-4).
        """
        if self.accumulated_data is None:
            self.accumulated_data = pd.DataFrame(columns=new_data.columns)
        self.accumulated_data = pd.concat(
            [self.accumulated_data, new_data], ignore_index=True)
        current_count = len(self.accumulated_data)

        # Compute additional features.
        self.accumulated_data = QForecasterDataprocessor.compute_additional_features(
            self.accumulated_data)

        # Identify the initial fill point and the ch₁ point.
        init_fill_point = QForecasterDataprocessor.init_fill_point(
            self.accumulated_data, 100, threshold_factor=20)
        ch1_fill_point = QForecasterDataprocessor.ch1_point(
            self.accumulated_data, init_fill_point)

        # If fill hasn't started, report all data as no_fill (0).
        if init_fill_point == -1:
            print("[INFO] Fill not yet started; reporting all data as no_fill (0).")
            predictions = np.zeros(current_count, dtype=int)
            conf = np.zeros(current_count)
            for i in range(current_count):
                if i not in self.prediction_history:
                    self.prediction_history[i] = []
                self.prediction_history[i].append((0, 1.0))
            return {
                "status": "waiting",
                "pred": predictions,
                "conf": conf,
                "stable_predictions": {},
                "accumulated_data": self.accumulated_data,
                "accumulated_count": current_count
            }

        predictions = np.zeros(current_count, dtype=int)
        conf = np.zeros(current_count)

        # Pre-fill data (before init_fill_point) are set to 0.
        for i in range(init_fill_point):
            if i not in self.prediction_history:
                self.prediction_history[i] = []
            self.prediction_history[i].append((0, 1.0))

        if current_count > init_fill_point:
            features = self.numerical_features
            X_live = self.accumulated_data[features].iloc[init_fill_point:]
            prob_matrix = self.predict_native(
                self.model, self.preprocessors, X_live)

            # Apply Viterbi decoding and enforce fill duration based on the init_fill block.
            pred_ml, prob_matrix, _ = self.enforce_fill_distribution(
                prob_matrix, self.transition_matrix)
            # Offset raw predictions by 1 (raw 0-3 become final 1-4).
            ml_pred_offset = pred_ml + 1

            # ----------------------------------------------------------
            # (1) Restrict predictions: Ensure that label 2 only starts at ch₁ point.
            # Since ml_pred_offset corresponds to data starting at init_fill_point,
            # for indices before ch1_fill_point, force any label 2 to label 1.
            if ch1_fill_point > init_fill_point:
                region_length = ch1_fill_point - init_fill_point
                ml_pred_offset[:region_length] = np.where(
                    ml_pred_offset[:region_length] == 2,
                    1,
                    ml_pred_offset[:region_length]
                )

                # ------------------------------------------------------
                # (2) Enforce minimum region length on predictions 2 and 3.
                # For any contiguous block of predictions that are 2 or 3,
                # if its length is shorter than the length of the init fill region,
                # change the block (here we set it to 1; alternatively, merge/extend as needed).
                def enforce_min_region_length(preds, min_length, allowed_labels=(2, 3)):
                    start = None
                    for i in range(len(preds)):
                        if preds[i] in allowed_labels:
                            if start is None:
                                start = i
                        else:
                            if start is not None:
                                if i - start < min_length:
                                    # Adjust the region to label 1.
                                    preds[start:i] = 1
                                start = None
                    # Handle the case where the allowed region extends to the end.
                    if start is not None and len(preds) - start < min_length:
                        preds[start:] = 1
                    return preds

                ml_pred_offset = enforce_min_region_length(
                    ml_pred_offset, region_length)

            predictions[init_fill_point:] = ml_pred_offset

            ml_conf = np.array([prob_matrix[i, pred_ml[i]]
                                for i in range(len(pred_ml))])
            conf[init_fill_point:] = ml_conf

            for idx, (p, c) in enumerate(zip(ml_pred_offset, ml_conf), start=init_fill_point):
                if idx not in self.prediction_history:
                    self.prediction_history[idx] = []
                self.prediction_history[idx].append((p, c))

        stable_predictions = {}
        for idx, history in self.prediction_history.items():
            stable, stable_class = self.is_prediction_stable(
                history, stability_window, frequency_threshold, confidence_threshold)
            if stable:
                stable_predictions[idx] = stable_class

        self.batch_num += 1
        print(f"\n[INFO] Running predictions on batch {self.batch_num} with {current_count} entries. "
              f"ML predictions are offset by 1, and pre-fill data are marked as no_fill (0).")

        return {
            "status": "completed",
            "pred": predictions,
            "conf": conf,
            "stable_predictions": stable_predictions,
            "accumulated_data": self.accumulated_data,
            "accumulated_count": current_count
        }


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
        batch_size = random.randint(100, 200)
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
        Stable prediction regions (as determined by the model selection system) are overlaid
        with darker shading. The plot is styled with a clean, minimalist aesthetic.
        """
        self.ax.cla()  # Clear previous plot

        # Check if there is enough data to run predictions.
        if results.get("status") == "waiting":
            self.ax.set_title(
                f"Waiting for more data (Batch {batch_number})", fontsize=14)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            return

        # Otherwise, proceed with plotting.
        accumulated_data = results.get("accumulated_data")
        if accumulated_data is None or accumulated_data.empty:
            print("[WARN] No accumulated data available for plotting.")
            return

        x_dissipation = np.arange(len(accumulated_data))
        init_fill_point = QForecasterDataprocessor.init_fill_point(
            accumulated_data, 100, threshold_factor=20)
        ch1_fill_point = QForecasterDataprocessor.ch1_point(
            accumulated_data, init_fill_point)

        if init_fill_point > -1:
            self.ax.axvline(init_fill_point, color='orange',
                            label='Initial fill location')

        if ch1_fill_point > -1:
            self.ax.axvline(ch1_fill_point, color='brown',
                            label='Channel 1 fill location')
        self.ax.plot(
            x_dissipation,
            accumulated_data["Dissipation"],
            linestyle=':',
            color='#333333',
            linewidth=1.5,
            label='Dissipation Curve'
        )

        preds = np.array(results["pred"])

        # Define mappings for class IDs to names and colors.
        class_names = {
            0: "no_fill",
            1: "initial_fill",
            2: "channel_1",
            3: "channel_2",
            4: "full_fill"
        }
        class_colors = {
            0: "#d3d3d3",    # light grey for no_fill
            1: "#a6cee3",    # pastel blue for initial_fill
            2: "#b2df8a",    # pastel green for channel_1
            3: "#fdbf6f",    # pastel orange for channel_2
            4: "#fb9a99"     # pastel red/pink for full_fill
        }

        # Overlay shaded regions for predicted class regions.
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
                        start_idx, end_idx,
                        color=class_colors.get(current_class, '#cccccc'),
                        alpha=0.3,
                        label=label
                    )
                    current_class = preds[i]
                    start_idx = i
            # Shade the final segment.
            end_idx = len(preds) - 1
            label = class_names[current_class] if current_class not in already_labeled else None
            self.ax.axvspan(
                start_idx, end_idx,
                color=class_colors.get(current_class, '#cccccc'),
                alpha=0.3,
                label=label
            )

        # Scatter the actual POI indices on the dissipation curve.
        if self.actual_poi_indices.size > 0:
            valid_actual_indices = [
                int(idx) for idx in self.actual_poi_indices if idx < len(accumulated_data)]
            if valid_actual_indices:
                y_actual = accumulated_data["Dissipation"].iloc[valid_actual_indices]
                self.ax.scatter(
                    valid_actual_indices,
                    y_actual,
                    color='#2ca02c',
                    marker='o',
                    s=50,
                    label='Actual POI'
                )

       # Overlay stable prediction regions for the short model as darker shaded areas.
        if "stable_predictions" in results and results["stable_predictions"]:
            stable_dict = results["stable_predictions"]
            stable_idxs = sorted(stable_dict.keys())
            if stable_idxs:
                segments = []
                current_class = stable_dict[stable_idxs[0]]
                seg_start = stable_idxs[0]
                prev = stable_idxs[0]
                # Group contiguous indices with the same stable class.
                for idx in stable_idxs[1:]:
                    if idx == prev + 1 and stable_dict[idx] == current_class:
                        prev = idx
                    else:
                        segments.append((seg_start, prev, current_class))
                        seg_start = idx
                        prev = idx
                        current_class = stable_dict[idx]
                segments.append((seg_start, prev, current_class))
                already_labeled = {}
                for seg_start, seg_end, cls in segments:
                    label = f"Stable Short: {class_names[cls]}" if cls not in already_labeled else None
                    already_labeled[cls] = True
                    self.ax.axvspan(
                        seg_start, seg_end,
                        color=class_colors.get(cls, '#000000'),
                        alpha=0.6,  # darker for short model
                        label=label
                    )

        self.ax.set_title(
            f'Dissipation Curve (Batch {batch_number})', fontsize=14, weight='medium')
        self.ax.set_xlabel('Data Index', fontsize=12)
        self.ax.set_ylabel(self.predictor.target, fontsize=12)
        self.ax.tick_params(axis='both', which='major', labelsize=10)
        self.ax.legend(frameon=False, fontsize=10)

        # Redraw the figure.
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
                delay=delay * 5
            )

            # Run the simulation.
            simulator.run()
