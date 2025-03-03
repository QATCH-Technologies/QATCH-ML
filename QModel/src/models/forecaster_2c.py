from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.linear_model import LogisticRegression
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy.signal import hilbert, savgol_filter
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle

FEATURES = [
    'Relative_time',
    'Dissipation',
    'Dissipation_rolling_mean',
    'Dissipation_rolling_median',
    'Dissipation_ewm',
    'Dissipation_rolling_std',
    'Dissipation_diff',
    'Dissipation_pct_change',
    'Dissipation_rate',
    'Dissipation_ratio_to_mean',
    'Dissipation_ratio_to_ewm',
    'Dissipation_envelope',
    'Time_shift'
]
TARGET = "Fill"
DATA_TO_LOAD = 50
IGNORE_BEFORE = 50


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
        df['Relative_time_diff'] = df['Relative_time'].diff().replace(0, np.nan)
        df['Dissipation_rate'] = df['Dissipation_diff'] / df['Relative_time_diff']
        df['Dissipation_ratio_to_mean'] = df['Dissipation'] / \
            df['Dissipation_rolling_mean']
        df['Dissipation_ratio_to_ewm'] = df['Dissipation'] / df['Dissipation_ewm']
        df['Dissipation_envelope'] = np.abs(hilbert(df['Dissipation'].values))

        # Drop the Resonance_Frequency column if present.
        if 'Resonance_Frequency' in df.columns:
            df.drop(columns=['Resonance_Frequency'], inplace=True)

        t_delta = QForecasterDataprocessor.find_time_delta(df)
        if t_delta == -1:
            df['Time_shift'] = 0
        else:
            df.loc[t_delta:, 'Time_shift'] = 1

        return df

    @staticmethod
    def _process_fill(df: pd.DataFrame, poi_file: str, check_unique: bool = False, file_name: str = None) -> pd.DataFrame:
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
    def load_and_preprocess_data_split(data_dir: str):
        """
        Load and preprocess data from all files in data_dir. Each file (and its matching
        POI file) is processed to compute additional features and fill information.
        Files are then categorized into 'short_runs' and 'long_runs' based on whether
        a significant time delta is detected. Before final concatenation, if the number
        of rows in a run exceeds IGNORE_BEFORE, the first IGNORE_BEFORE rows are dropped.
        Additionally, each DataFrame is filtered to remove rows with Relative_time < 1.2
        and downsampled by a factor of 5.
        Returns two DataFrames: one for short runs and one for long runs.
        """
        runs = []
        content = QForecasterDataprocessor.load_content(data_dir)
        random.shuffle(content)

        for file, poi_file in content:

            df = pd.read_csv(file)
            required_cols = ["Relative_time", "Dissipation"]
            if df.empty or not all(col in df.columns for col in required_cols):
                continue

            df = df[required_cols]
            df = QForecasterDataprocessor.compute_additional_features(df)
            try:
                df = QForecasterDataprocessor._process_fill(
                    df, poi_file, check_unique=True, file_name=file)
            except FileNotFoundError:
                df = None
            if df is None:
                continue
            df = df[df["Relative_time"] >= 1.2]
            df = df.iloc[::5]
            runs.append(df)

        training_data = pd.concat(runs).sort_values(
            "Relative_time").reset_index(drop=True)

        return training_data

    @staticmethod
    def load_and_preprocess_single(data_file: str, poi_file: str):
        """
        Load and preprocess a single data file (and its corresponding POI file).
        The method ensures required sensor columns exist, processes the fill information,
        and prints a head sample of the resulting DataFrame.
        """
        df = pd.read_csv(data_file)
        required_cols = ["Relative_time", "Resonance_Frequency", "Dissipation"]
        if df.empty or not all(col in df.columns for col in required_cols):
            raise ValueError(
                "Data file is empty or missing required sensor columns.")
        df = df[required_cols]
        df = QForecasterDataprocessor._process_fill(df, poi_file)
        df = df[df["Relative_time"] >= 1.2]

        df = df.iloc[::5]
        if df is None:
            raise ValueError("Error processing fill information.")
        print("[INFO] Preprocessed single-file data sample:")
        print(df.head())
        return df

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


###############################################################################
# Trainer Class: Handles model training, hyperparameter tuning, and saving.
###############################################################################

class QForecasterTrainer:
    def __init__(self, numerical_features, target='Fill', save_dir=None):
        """
        Args:
            numerical_features (list): List of numerical feature names.
            target (str): Target column name.
            save_dir (str): Directory to save trained objects.
        """
        self.numerical_features = numerical_features
        self.target = target
        self.save_dir = save_dir

        self.model = None
        self.preprocessors = None
        self.params = None
        self.transition_matrix = None

    def _build_preprocessors(self, X):
        """Fit and return preprocessors for numerical data."""
        num_imputer = SimpleImputer(strategy='mean')
        X_num = num_imputer.fit_transform(X[self.numerical_features])
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
                metrics={'aucpr'},
                early_stopping_rounds=10,
                seed=42,
                verbose_eval=False
            )
            best_score = cv_results['test-aucpr-mean'].max()
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
        features = self.numerical_features
        X = training_data[features].copy()
        y = training_data[self.target].values

        X_processed, preprocessors = self._build_preprocessors(X)
        dtrain = xgb.DMatrix(X_processed, label=y)

        base_params = {
            'objective': 'multi:softprob',
            'num_class': 5,
            'eval_metric': 'aucpr',
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
                metrics={'aucpr'},
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
            X_copy[self.numerical_features])
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
        self.model.load_model(os.path.join(
            self.save_dir, "model.json"))
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

        # Consider only the last 'stability_window' predictions.
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

    def update_predictions(self, new_data, ignore_before=0, stability_window=5,
                           frequency_threshold=0.8, confidence_threshold=0.9, current_state=None):
        """
        Accumulate new data and run predictions in batches. The method updates prediction
        histories for stability detection and uses the meta-model for base model selection.
        """
        # Initialize accumulator if needed.
        if self.accumulated_data is None:
            self.accumulated_data = pd.DataFrame(columns=new_data.columns)

        # Append the new data.
        self.accumulated_data = pd.concat(
            [self.accumulated_data, new_data], ignore_index=True)
        current_count = len(self.accumulated_data)
        self.accumulated_data = QForecasterDataprocessor.compute_additional_features(
            self.accumulated_data)
        if current_count < self.batch_threshold:
            print(
                f"[INFO] Accumulated {current_count} entries so far; waiting for {self.batch_threshold} entries to run predictions.")
            return {
                "status": "waiting",
                "accumulated_count": current_count,
                "accumulated_data": self.accumulated_data,
                "predictions": None
            }

        self.batch_num += 1
        print(
            f"\n[INFO] Running predictions on batch {self.batch_num} with {current_count} entries.")

        # Optionally ignore initial rows for stabilization (only on the first run).
        if self.batch_num == 1 and current_count > ignore_before:
            self.accumulated_data = self.accumulated_data.iloc[ignore_before:]

        features = self.numerical_features
        X_live = self.accumulated_data[features]

        prob_matrix = self.predict_native(
            self.model, self.preprocessors, X_live)
        pred = QForecasterDataprocessor.viterbi_decode(
            prob_matrix, self.transition_matrix)

        conf = np.array([prob_matrix[i, pred[i]]
                         for i in range(len(pred))])

        for i, (p_s, c_s) in enumerate(zip(pred, conf)):
            if i not in self.prediction_history:
                self.prediction_history[i] = []
            self.prediction_history[i].append((p_s, c_s))

        stable_predictions = {}
        for idx, history in self.prediction_history.items():
            stable, stable_class = self.is_prediction_stable(
                history, stability_window, frequency_threshold, confidence_threshold)
            if stable:
                stable_predictions[idx] = stable_class

        # Capture the accumulated data for plotting before resetting.
        data_for_plot = self.accumulated_data.copy()
        return {
            "status": "completed",
            "pred": pred,
            "conf": conf,
            "stable_predictions": stable_predictions,
            "accumulated_data": data_for_plot,
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
            self.actual_poi_indices = poi_file.values
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
            results = self.predictor.update_predictions(
                batch_data, ignore_before=self.ignore_before)

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
    if TRAINING:
        train_dir = r'content\training_data'
        training_data = QForecasterDataprocessor.load_and_preprocess_data_split(
            train_dir)

        qft = QForecasterTrainer(FEATURES, TARGET,
                                 r'QModel\SavedModels\forecaster')
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
            poi_file = pd.read_csv(poi_file, header=None)

            predictor = QForecasterPredictor(
                FEATURES, target='Fill', save_dir='QModel/SavedModels/forecaster/')
            # Ensure models and preprocessors are loaded.
            predictor.load_models()

            delay = dataset['Relative_time'].max() / len(random_slice)
            random_slice = random_slice[random_slice["Relative_time"] >= 1.2]
            random_slice = random_slice.iloc[::5]
            # Create the simulator, now passing the poi_file.
            simulator = QForecasterSimulator(
                predictor,
                random_slice,
                poi_file=poi_file,
                ignore_before=50,
                delay=delay
            )

            # Run the simulation.
            simulator.run()
