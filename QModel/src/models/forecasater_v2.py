from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
import xgboost as xgb
import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# Assumes definitions for NUM_THREADS, SEED, TRAINING, TESTING, etc.
from q_constants import *


# =============================================================================
# Data Processing Helpers
# =============================================================================
# =============================================================================
# Data Processing Helpers (Extended with Live Feature Computation)
# =============================================================================
class DataProcessor:
    """Helper class to process and transform data."""

    @staticmethod
    def apply_smoothing(df: pd.DataFrame, smoothing: float = 0.01) -> pd.DataFrame:
        for col in ["Resonance_Frequency", "Dissipation"]:
            data = df[col].values
            n_points = len(data)
            window_length = max(5, int(n_points * smoothing))
            if window_length % 2 == 0:
                window_length += 1
            polyorder = 2
            if window_length <= polyorder:
                window_length = polyorder + 2
                if window_length % 2 == 0:
                    window_length += 1
            df[col] = savgol_filter(
                data, window_length=window_length, polyorder=polyorder)
        return df

    @staticmethod
    def reassign_region(Fill) -> str:
        if Fill == 0:
            return 'no_fill'
        elif Fill in [1, 2, 3]:
            return 'init_fill'
        elif Fill == 4:
            return 'ch_1'
        elif Fill == 5:
            return 'ch_2'
        elif Fill == 6:
            return 'full_fill'
        else:
            return Fill

    @staticmethod
    def balance_classes(combined_data: pd.DataFrame) -> pd.DataFrame:
        class_counts = combined_data['Fill'].value_counts()
        max_count = class_counts.max()
        balanced_dfs = []
        for label, group in combined_data.groupby('Fill'):
            if len(group) < max_count:
                group_balanced = group.sample(
                    max_count, replace=True, random_state=42)
            else:
                group_balanced = group
            balanced_dfs.append(group_balanced)
        combined_data_balanced = pd.concat(balanced_dfs).sort_values(
            by='Relative_time').reset_index(drop=True)
        return combined_data_balanced

    @staticmethod
    def compute_dynamic_transition_matrix(training_data: pd.DataFrame, num_states: int = 5,
                                          smoothing: float = 1e-6) -> np.ndarray:
        states = training_data["Fill"].values
        transition_counts = np.full((num_states, num_states), smoothing)
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            if next_state == current_state or next_state == current_state + 1:
                transition_counts[current_state, next_state] += 1
        transition_matrix = transition_counts / \
            transition_counts.sum(axis=1, keepdims=True)
        return transition_matrix

    @staticmethod
    def add_live_features(df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
        """
        Compute additional signal processing features live.
        Features include:
          - Rolling mean, std, and median for Dissipation and Resonance_Frequency.
          - First and second derivatives.
          - Signal energy (rolling mean of squared values).
          - Rolling range (max-min).
        """
        # Rolling statistics for Dissipation
        df['Dissipation_roll_mean'] = df['Dissipation'].rolling(
            window=window_size, min_periods=1).mean()
        df['Dissipation_roll_std'] = df['Dissipation'].rolling(
            window=window_size, min_periods=1).std().fillna(0)
        df['Dissipation_roll_median'] = df['Dissipation'].rolling(
            window=window_size, min_periods=1).median()

        # Rolling statistics for Resonance_Frequency
        df['RF_roll_mean'] = df['Resonance_Frequency'].rolling(
            window=window_size, min_periods=1).mean()
        df['RF_roll_std'] = df['Resonance_Frequency'].rolling(
            window=window_size, min_periods=1).std().fillna(0)
        df['RF_roll_median'] = df['Resonance_Frequency'].rolling(
            window=window_size, min_periods=1).median()

        # First derivatives
        df['RF_derivative'] = df['Resonance_Frequency'].diff().fillna(0)
        df['Dissipation_derivative'] = df['Dissipation'].diff().fillna(0)

        # Second derivative (acceleration) for Resonance_Frequency
        df['RF_second_derivative'] = df['RF_derivative'].diff().fillna(0)

        # Signal energy: rolling mean of squared signal
        df['RF_energy'] = (df['Resonance_Frequency'] **
                           2).rolling(window=window_size, min_periods=1).mean()
        df['Dissipation_energy'] = (
            df['Dissipation'] ** 2).rolling(window=window_size, min_periods=1).mean()

        # Rolling range (max - min)
        df['RF_range'] = df['Resonance_Frequency'].rolling(window=window_size, min_periods=1)\
            .apply(lambda x: x.max() - x.min(), raw=True)
        df['Dissipation_range'] = df['Dissipation'].rolling(window=window_size, min_periods=1)\
            .apply(lambda x: x.max() - x.min(), raw=True)

        return df

# =============================================================================
# Data Loader
# =============================================================================


class DataLoader:
    """
    Handles loading and preprocessing of training/test data from CSV files.
    """

    def __init__(self, data_to_load: int = np.inf):
        self.data_to_load = data_to_load

    def load_content(self, data_dir: str) -> list:
        print(f"[INFO] Loading content from {data_dir}")
        loaded_content = []
        for data_root, _, data_files in tqdm(os.walk(data_dir), desc='Loading files...'):
            for f in data_files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    matched_poi_file = f.replace(".csv", "_poi.csv")
                    loaded_content.append(
                        (os.path.join(data_root, f), os.path.join(data_root, matched_poi_file)))
        return loaded_content

    def load_and_preprocess_data(self, data_dir: str) -> pd.DataFrame:
        """
        Loads multiple CSV files, applies smoothing, region reassignment,
        and computes extra signal processing features. Finally, it balances the classes.
        """
        all_data = []
        content = self.load_content(data_dir)
        num_samples = self.data_to_load
        sampled_content = random.sample(
            content, k=min(num_samples, len(content)))

        for data_file, poi_path in sampled_content:
            df = pd.read_csv(data_file)
            if not df.empty and all(col in df.columns for col in ["Relative_time", "Resonance_Frequency", "Dissipation"]):
                df = df[["Relative_time", "Resonance_Frequency", "Dissipation"]]
                # Apply smoothing
                df = DataProcessor.apply_smoothing(df, smoothing=0.01)
                Fill_df = pd.read_csv(poi_path, header=None)
                if "Fill" in Fill_df.columns:
                    df["Fill"] = Fill_df["Fill"]
                else:
                    df["Fill"] = 0
                    change_indices = sorted(Fill_df.iloc[:, 0].values)
                    for idx in change_indices:
                        df.loc[idx:, "Fill"] += 1
                df["Fill"] = pd.Categorical(df["Fill"]).codes
                all_data.append(df)

        if not all_data:
            raise ValueError("No valid data found in the specified directory.")

        combined_data = pd.concat(all_data).sort_values(by='Relative_time')
        unique_Fill = sorted(combined_data["Fill"].unique())
        if len(unique_Fill) != 7:
            raise ValueError(
                f"Expected 7 unique Fill values, but found {len(unique_Fill)}: {unique_Fill}")

        # Reassign regions and map to numeric values.
        combined_data['Fill'] = combined_data['Fill'].apply(
            DataProcessor.reassign_region)
        mapping = {'no_fill': 0, 'init_fill': 1,
                   'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
        combined_data['Fill'] = combined_data['Fill'].map(mapping)

        # Balance the classes
        combined_data_balanced = DataProcessor.balance_classes(combined_data)

        # *** New: Add additional signal processing features for training ***
        combined_data_balanced = DataProcessor.add_live_features(
            combined_data_balanced, window_size=10)

        print("[INFO] Combined balanced data sample:")
        print(combined_data_balanced.head())
        print("[INFO] Class distribution after balancing:")
        print(combined_data_balanced['Fill'].value_counts())
        return combined_data_balanced

    def load_and_preprocess_single(self, data_file: str, poi_file: str) -> pd.DataFrame:
        df = pd.read_csv(data_file)
        required_cols = ["Relative_time", "Resonance_Frequency", "Dissipation"]
        if df.empty or not all(col in df.columns for col in required_cols):
            raise ValueError(
                "Data file is empty or missing required sensor columns.")
        df = df[required_cols]

        poi_df = pd.read_csv(poi_file, header=None)
        if "Fill" in poi_df.columns:
            df["Fill"] = poi_df["Fill"]
        else:
            df["Fill"] = 0
            change_indices = sorted(poi_df.iloc[:, 0].values)
            for idx in change_indices:
                df.loc[idx:, "Fill"] += 1
        df["Fill"] = pd.Categorical(df["Fill"]).codes
        df["Fill"] = df["Fill"].apply(DataProcessor.reassign_region)
        mapping = {'no_fill': 0, 'init_fill': 1,
                   'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
        df["Fill"] = df["Fill"].map(mapping)

        # Also add live features here if desired.
        df = DataProcessor.add_live_features(df, window_size=10)

        print("[INFO] Preprocessed single-file data sample:")
        print(df.head())
        return df


# =============================================================================
# XGBoost Forecaster with Tuning and Live Prediction
# =============================================================================
class XGBForecaster:
    """
    Wraps the XGBoost model, hyperparameter tuning (via hyperopt), training,
    prediction (with Viterbi decoding), and live prediction loop.
    """

    def __init__(self, seed: int = SEED, num_threads: int = NUM_THREADS):
        self.seed = seed
        self.num_threads = num_threads
        self.model = None

    def tune_model(self, X: pd.DataFrame, y: pd.Series, max_evals: int = 10) -> dict:
        """
        Uses hyperopt to tune hyperparameters with cross-validation.
        """
        dtrain = xgb.DMatrix(X, label=y)

        def objective(params):
            params_copy = params.copy()
            # Fix parameters that remain constant
            params_copy.update({
                "objective": "multi:softprob",
                "num_class": 5,
                "eval_metric": "mlogloss",
                "eta": 0.175,
                "max_depth": 5,
                "min_child_weight": 4.0,
                "subsample": 0.6,
                "colsample_bytree": 0.75,
                "gamma": 0.8,
                "nthread": self.num_threads,
                "booster": "gbtree",
                "device": "cuda",
                "tree_method": "auto",
                "sampling_method": "gradient_based",
                "seed": self.seed,
            })
            cv_results = xgb.cv(
                params_copy,
                dtrain,
                num_boost_round=1000,
                nfold=3,
                metrics='auc',
                seed=42,
                early_stopping_rounds=10,
                verbose_eval=False
            )
            best_iteration = len(cv_results)
            best_loss = cv_results['test-auc-mean'].max()
            return {'loss': -best_loss, 'status': STATUS_OK, 'best_iteration': best_iteration, 'params': params_copy}

        space = {
            "max_depth": hp.choice("max_depth", np.arange(1, 20, 1, dtype=int)),
            "eta": hp.uniform("eta", 0, 1),
            "gamma": hp.uniform("gamma", 0, 100),
            "reg_alpha": hp.uniform("reg_alpha", 1e-7, 10),
            "reg_lambda": hp.uniform("reg_lambda", 0, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "colsample_bynode": hp.uniform("colsample_bynode", 0.5, 1),
            "colsample_bylevel": hp.uniform("colsample_bylevel", 0.5, 1),
            "min_child_weight": hp.choice("min_child_weight", np.arange(1, 10, 1, dtype=int)),
            "max_delta_step": hp.choice("max_delta_step", np.arange(1, 10, 1, dtype=int)),
            "subsample": hp.uniform("subsample", 0.5, 1),
            # Some parameters below are redundant because they are fixed in the objective.
            "eval_metric": "auc",
            "objective": "multi:softprob",
            "nthread": self.num_threads,
            "booster": "gbtree",
            "device": "cuda",
            "tree_method": "auto",
            "sampling_method": "gradient_based",
            "seed": self.seed,
        }

        trials = Trials()
        rng = np.random.default_rng(self.seed)

        best_result = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=rng,
            early_stop_fn=no_progress_loss(10, percent_increase=0.1),
        )

        best_trial = trials.best_trial['result']
        best_params = best_trial['params']
        best_num_round = best_trial['best_iteration']

        print("Best hyperparameters found:", best_params)
        print("Best cross-validation mlogloss:", best_trial['loss'])

        return {"params": best_params, "num_round": best_num_round}

    def train(self, training_data: pd.DataFrame):
        """
        Extracts features and labels, performs tuning, and then trains the final model.
        """
        features = ["Relative_time", "Resonance_Frequency", "Dissipation"]
        X = training_data[features]
        y = training_data["Fill"]
        tuning_result = self.tune_model(X, y, max_evals=10)
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(
            tuning_result["params"], dtrain, num_boost_round=tuning_result["num_round"])
        print("[INFO] Model training complete.")
        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns predicted probabilities for input features.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        ddata = xgb.DMatrix(X[self.model.feature_names])
        return self.model.predict(ddata)

    def save_model(self, save_path: str):
        """
        Saves the model to a file.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save_model(save_path)
        print(f"[INFO] Model saved to {save_path}")

    def load_model(self, load_path: str):
        """
        Loads the model from a file.
        """
        self.model = xgb.Booster()
        self.model.load_model(load_path)
        print(f"[INFO] Model loaded from {load_path}")

    @staticmethod
    def viterbi_decode(prob_matrix: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
        """
        Applies Viterbi decoding to a matrix of probabilities given a transition matrix.
        """
        T, N = prob_matrix.shape
        dp = np.full((T, N), -np.inf)
        backpointer = np.zeros((T, N), dtype=int)
        # Assume the first sample is in state 0
        dp[0, 0] = np.log(prob_matrix[0, 0])

        for t in range(1, T):
            for j in range(N):
                allowed_prev = [0] if j == 0 else [j - 1, j]
                best_state = allowed_prev[0]
                best_score = dp[t - 1, best_state] + \
                    np.log(transition_matrix[best_state, j])
                for i in allowed_prev:
                    if transition_matrix[i, j] <= 0:
                        continue
                    score = dp[t - 1, i] + np.log(transition_matrix[i, j])
                    if score > best_score:
                        best_score = score
                        best_state = i
                dp[t, j] = np.log(prob_matrix[t, j]) + best_score
                backpointer[t, j] = best_state

        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = np.argmax(dp[T - 1])
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]
        return best_path

    @staticmethod
    def simulate_stream(loaded_data: pd.DataFrame, batch_size: int = 100):
        """
        Generator that yields successive batches of data.
        """
        num_rows = len(loaded_data)
        for start_idx in range(0, num_rows, batch_size):
            yield loaded_data.iloc[start_idx:start_idx + batch_size]

    def _update_live_monitor(self, accumulated_data: pd.DataFrame, predictions: np.ndarray, axes, delay: float):
        """
        Updates the live monitor plots.
        """
        indices = np.arange(len(accumulated_data))
        axes[0].cla()
        axes[0].plot(indices, accumulated_data["Fill"], label="Actual", color='blue',
                     marker='o', linestyle='--', markersize=3)
        axes[0].plot(indices, predictions, label="Predicted", color='red',
                     marker='x', linestyle='-', markersize=3)
        axes[0].set_title("Predicted vs Actual Fill")
        axes[0].set_xlabel("Data Point Index")
        axes[0].set_ylabel("Fill Category (numeric)")
        axes[0].legend()
        axes[0].set_ylim(-0.5, 4.5)

        axes[1].cla()
        axes[1].plot(indices, accumulated_data["Dissipation"],
                     label="Dissipation", color='green')
        axes[1].set_title("Dissipation Curve")
        axes[1].set_xlabel("Data Point Index")
        axes[1].set_ylabel("Dissipation")
        axes[1].legend()
        plt.pause(delay)

    def live_prediction_loop(self, loaded_data: pd.DataFrame, transition_matrix: np.ndarray,
                             batch_size: int = 100, delay: float = 0.1):
        """
        Simulate live predictions by streaming data in batches, applying smoothing,
        computing additional live features, generating predictions, decoding with Viterbi,
        and updating live plots.
        """
        accumulated_data = pd.DataFrame(columns=loaded_data.columns)
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        batch_num = 0

        for batch in self.simulate_stream(loaded_data, batch_size=batch_size):
            batch_num += 1
            print(
                f"\n[INFO] Streaming batch {batch_num} with {len(batch)} data points...")
            time.sleep(delay)  # Simulate delay

            # Accumulate data
            accumulated_data = pd.concat(
                [accumulated_data, batch], ignore_index=True)

            # Apply smoothing to sensor columns
            for col in ["Resonance_Frequency", "Dissipation"]:
                data = accumulated_data[col].values
                n = len(data)
                window_length = int(np.ceil(0.01 * n))
                window_length = max(window_length, 3)
                if window_length % 2 == 0:
                    window_length += 1
                if window_length > n:
                    window_length = n if n % 2 == 1 else n - 1
                accumulated_data[col] = savgol_filter(
                    data, window_length=window_length, polyorder=2)

            # *** NEW: Compute live features ***
            accumulated_data = DataProcessor.add_live_features(
                accumulated_data)

            # Select features for prediction (update feature list if necessary)
            features = ["Relative_time", "Resonance_Frequency", "Dissipation",
                        "Dissipation_roll_mean", "RF_derivative"]
            X_live = accumulated_data[features]
            # Model must have been trained with these features.
            prob_matrix = self.predict(X_live)

            predicted_sequence = self.viterbi_decode(
                prob_matrix, transition_matrix)
            self._update_live_monitor(
                accumulated_data, predicted_sequence, axes, delay)
            print(
                f"[INFO] Updated live monitor after batch {batch_num}. Total points: {len(accumulated_data)}")

        plt.ioff()
        plt.show()
        return predicted_sequence


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    save_path = r"QModel\SavedModels\xgb_forecaster_new_ft.json"
    test_dir = r"content\test_data"
    training_data_dir = r"content/training_data/full_fill"  # Update as needed

    # 1. Load and preprocess training data from multiple files.
    loader = DataLoader(data_to_load=10)
    training_data = loader.load_and_preprocess_data(training_data_dir)
    transition_matrix = DataProcessor.compute_dynamic_transition_matrix(
        training_data)

    # 2. Train the model if in training mode.
    forecaster = XGBForecaster(seed=SEED, num_threads=NUM_THREADS)
    if TRAINING:
        model = forecaster.train(training_data)
        forecaster.save_model(save_path)

    # 3. For testing, load a saved model and run live prediction simulation.
    if TESTING:
        forecaster.load_model(save_path)
        test_content = loader.load_content(test_dir)
        random.shuffle(test_content)
        for data_file, poi_file in test_content:
            df_test = pd.read_csv(data_file)
            delay = df_test['Relative_time'].max(
            ) / len(df_test["Relative_time"].values)
            loaded_data = loader.load_and_preprocess_single(
                data_file, poi_file)
            print(
                f"[INFO] Loaded single run data with {len(loaded_data)} data points.\n")
            forecaster.live_prediction_loop(
                loaded_data, transition_matrix, batch_size=100, delay=delay/2)
            print("\n[INFO] Live prediction simulation complete.")
