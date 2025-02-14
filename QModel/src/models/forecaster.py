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
from scipy.ndimage import uniform_filter1d


# =============================================================================
# Data Processing Helpers
# =============================================================================
class DataProcessor:
    """Helper class to process and transform data."""

    @staticmethod
    def apply_smoothing(df: pd.DataFrame, smoothing: float = 0.01) -> pd.DataFrame:
        """
        Applies Savitzky–Golay smoothing to specific sensor columns.
        """
        for col in ["Resonance_Frequency", "Dissipation"]:
            data = df[col].values
            n_points = len(data)
            # Calculate window_length as a proportion of the data length (at least 5 points).
            window_length = max(5, int(n_points * smoothing))
            # Window length must be odd.
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
        """
        Reassign the raw fill value to one of five region labels.
        """
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
            return Fill  # fallback

    @staticmethod
    def balance_classes(combined_data: pd.DataFrame) -> pd.DataFrame:
        """
        Oversample the data so that every class has the same number of samples.
        """
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
        """
        Compute a transition probability matrix from the training data.
        """
        states = training_data["Fill"].values
        transition_counts = np.full((num_states, num_states), smoothing)
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            # Only count transitions that are staying in the same state or moving to the next.
            if next_state == current_state or next_state == current_state + 1:
                transition_counts[current_state, next_state] += 1
        transition_matrix = transition_counts / \
            transition_counts.sum(axis=1, keepdims=True)
        return transition_matrix


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
        """
        Walks through the directory and returns a list of tuples
        (data_file, poi_file) for CSV files.
        """
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
        Loads multiple CSV files, applies smoothing and region reassignment,
        and then balances the classes.
        """
        all_data = []
        content = self.load_content(data_dir)
        num_samples = self.data_to_load
        sampled_content = random.sample(
            content, k=min(num_samples, len(content)))

        for data_file, poi_path in sampled_content:
            df = pd.read_csv(data_file)
            # Ensure required sensor columns exist
            if not df.empty and all(col in df.columns for col in ["Relative_time", "Resonance_Frequency", "Dissipation"]):
                df = df[["Relative_time", "Resonance_Frequency", "Dissipation"]]
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
        # Ensure exactly 7 unique fill values
        unique_Fill = sorted(combined_data["Fill"].unique())
        if len(unique_Fill) != 7:
            raise ValueError(
                f"Expected 7 unique Fill values, but found {len(unique_Fill)}: {unique_Fill}")

        # Reassign regions and then map string labels to numeric values.
        combined_data['Fill'] = combined_data['Fill'].apply(
            DataProcessor.reassign_region)
        mapping = {'no_fill': 0, 'init_fill': 1,
                   'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
        combined_data['Fill'] = combined_data['Fill'].map(mapping)

        combined_data_balanced = DataProcessor.balance_classes(combined_data)

        print("[INFO] Combined balanced data sample:")
        print(combined_data_balanced.head())
        print("[INFO] Class distribution after balancing:")
        print(combined_data_balanced['Fill'].value_counts())
        return combined_data_balanced

    def load_and_preprocess_single(self, data_file: str, poi_file: str) -> pd.DataFrame:
        """
        Load and preprocess a single run from a pair of CSV files.
        """
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
    def viterbi_decode(prob_matrix: np.ndarray,
                       transition_matrix: np.ndarray,
                       smooth_window: int = 3,
                       initial_state: int = None) -> np.ndarray:
        """
        Applies Viterbi decoding with an optional initial state.
        If initial_state is provided, the first observation is initialized using
        the transition probabilities from that state; otherwise, it assumes state 0.
        """
        # Smooth the prediction probabilities along the time axis using a uniform filter.
        if smooth_window > 1:
            prob_matrix = uniform_filter1d(
                prob_matrix, size=smooth_window, axis=0, mode='nearest')

        T, N = prob_matrix.shape
        dp = np.full((T, N), -np.inf)
        backpointer = np.zeros((T, N), dtype=int)

        # Initialize: if an initial_state is provided, incorporate it.
        if initial_state is not None:
            for j in range(N):
                dp[0, j] = np.log(prob_matrix[0, j]) + \
                    np.log(transition_matrix[initial_state, j])
        else:
            # Default: assume starting in state 0.
            dp[0, 0] = np.log(prob_matrix[0, 0])
            for j in range(1, N):
                dp[0, j] = -np.inf  # disallow starting in other states

        # Standard Viterbi dynamic programming
        for t in range(1, T):
            for j in range(N):
                # Allowed previous states: if j==0, only state 0; else, allow staying or advancing from j-1.
                allowed_prev = [0] if j == 0 else [j - 1, j]
                best_state = allowed_prev[0]
                best_score = dp[t - 1, allowed_prev[0]] + \
                    np.log(transition_matrix[allowed_prev[0], j])
                for i in allowed_prev:
                    if transition_matrix[i, j] <= 0:
                        continue
                    score = dp[t - 1, i] + np.log(transition_matrix[i, j])
                    if score > best_score:
                        best_score = score
                        best_state = i
                dp[t, j] = np.log(prob_matrix[t, j]) + best_score
                backpointer[t, j] = best_state

        # Backtrack to retrieve the best path.
        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = np.argmax(dp[T - 1])
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]
        return best_path
    import numpy as np

    @staticmethod
    def greedy_decode(prob_matrix: np.ndarray,
                      transition_matrix: np.ndarray,
                      smooth_window: int = 3,
                      initial_state: int = None) -> np.ndarray:
        """
        Applies Greedy decoding with an optional initial state.
        Allowed transitions: from state i, the next state can be i or i+1.
        """
        # Optionally smooth the probabilities.
        if smooth_window > 1:
            prob_matrix = uniform_filter1d(
                prob_matrix, size=smooth_window, axis=0, mode='nearest')

        T, N = prob_matrix.shape
        best_path = np.zeros(T, dtype=int)

        # Initialize the first state.
        if initial_state is not None:
            # Pick the best state at time 0 using the provided initial state's transitions.
            first_scores = np.log(prob_matrix[0]) + \
                np.log(transition_matrix[initial_state])
            best_path[0] = int(np.argmax(first_scores))
        else:
            best_path[0] = 0  # default starting state

        # Greedily choose the next state.
        for t in range(1, T):
            current_state = best_path[t - 1]
            # Allowed next states: stay in the current state or advance by one (if possible).
            allowed_next = [current_state]
            if current_state + 1 < N:
                allowed_next.append(current_state + 1)

            best_score = -np.inf
            best_next = current_state
            for j in allowed_next:
                if transition_matrix[current_state, j] <= 0:
                    continue
                score = np.log(prob_matrix[t, j]) + \
                    np.log(transition_matrix[current_state, j])
                if score > best_score:
                    best_score = score
                    best_next = j
            best_path[t] = best_next

        return best_path

    @staticmethod
    def beam_search_decode(prob_matrix: np.ndarray,
                           transition_matrix: np.ndarray,
                           beam_width: int = 3,
                           smooth_window: int = 3,
                           initial_state: int = None) -> np.ndarray:
        """
        Applies Beam Search decoding with an optional initial state.
        Allowed transitions: from state i, next state can be i or i+1.
        """
        if smooth_window > 1:
            prob_matrix = uniform_filter1d(
                prob_matrix, size=smooth_window, axis=0, mode='nearest')

        T, N = prob_matrix.shape
        beams = []  # Each beam is a tuple (score, path)

        # Initialize beams.
        if initial_state is not None:
            initial_beams = []
            for j in range(N):
                score = np.log(prob_matrix[0, j]) + \
                    np.log(transition_matrix[initial_state, j])
                initial_beams.append((score, [j]))
            beams = sorted(initial_beams, key=lambda x: x[0], reverse=True)[
                :beam_width]
        else:
            beams = [(np.log(prob_matrix[0, 0]), [0])]

        # Process each time step.
        for t in range(1, T):
            new_beams = []
            for score, path in beams:
                current_state = path[-1]
                # Allowed transitions: current_state or current_state+1 (if within bounds).
                allowed_next = [current_state]
                if current_state + 1 < N:
                    allowed_next.append(current_state + 1)
                for j in allowed_next:
                    if transition_matrix[current_state, j] <= 0:
                        continue
                    new_score = score + \
                        np.log(transition_matrix[current_state, j]
                               ) + np.log(prob_matrix[t, j])
                    new_path = path + [j]
                    new_beams.append((new_score, new_path))
            # Keep only the top beam_width paths.
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[
                :beam_width]

        # Return the best beam’s path.
        best_path = max(beams, key=lambda x: x[0])[1]
        return np.array(best_path)

    @staticmethod
    def posterior_decode(prob_matrix: np.ndarray,
                         transition_matrix: np.ndarray,
                         smooth_window: int = 3,
                         initial_state: int = None) -> np.ndarray:
        """
        Applies Posterior decoding using the forward-backward algorithm.
        Allowed transitions (for the forward/backward passes):
        - For time t, state j: if j==0, allowed previous state is [0];
            else, allowed previous states are [j-1, j].
        - In the backward pass, from state j the allowed next states are:
            [j] (and [j+1] if j+1 exists).
        """
        if smooth_window > 1:
            prob_matrix = uniform_filter1d(
                prob_matrix, size=smooth_window, axis=0, mode='nearest')
        T, N = prob_matrix.shape

        # Forward pass.
        alpha = np.zeros((T, N))
        if initial_state is not None:
            for j in range(N):
                alpha[0, j] = prob_matrix[0, j] * \
                    transition_matrix[initial_state, j]
        else:
            alpha[0, 0] = prob_matrix[0, 0]
            alpha[0, 1:] = 0.0

        for t in range(1, T):
            for j in range(N):
                allowed_prev = [0] if j == 0 else [j - 1, j]
                sum_val = 0.0
                for i in allowed_prev:
                    sum_val += alpha[t - 1, i] * transition_matrix[i, j]
                alpha[t, j] = prob_matrix[t, j] * sum_val

        # Backward pass.
        beta = np.zeros((T, N))
        beta[T - 1, :] = 1.0
        for t in range(T - 2, -1, -1):
            for j in range(N):
                # Allowed next states: from state j, if j < N-1 then allowed are [j, j+1]; else [j].
                allowed_next = [j] if j == N - 1 else [j, j + 1]
                sum_val = 0.0
                for k in allowed_next:
                    sum_val += transition_matrix[j, k] * \
                        prob_matrix[t + 1, k] * beta[t + 1, k]
                beta[t, j] = sum_val

        # Compute posterior marginals and select the state with maximum probability at each time step.
        best_path = np.zeros(T, dtype=int)
        for t in range(T):
            gamma = alpha[t, :] * beta[t, :]
            best_path[t] = int(np.argmax(gamma))
        return best_path

    @staticmethod
    def simulate_stream(loaded_data: pd.DataFrame, batch_size: int = 200):
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
        Simulate live predictions by streaming data in batches.
        Each new batch is decoded using the last predicted state from the previous batch.
        """
        accumulated_data = pd.DataFrame(columns=loaded_data.columns)
        full_predictions = []  # to store all predictions
        last_state = None      # will hold the final state from the previous batch

        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        batch_num = 0

        for batch in self.simulate_stream(loaded_data, batch_size=batch_size):
            batch_num += 1
            print(
                f"\n[INFO] Streaming batch {batch_num} with {len(batch)} data points...")
            time.sleep(delay)  # simulate delay

            # Append the new batch to accumulated data (for display purposes)
            accumulated_data = pd.concat(
                [accumulated_data, batch], ignore_index=True)
            # Optionally, you can smooth only the new batch and then merge with previous predictions.
            for col in ["Resonance_Frequency", "Dissipation"]:
                data = batch[col].values
                n = len(data)
                window_length = int(np.ceil(0.01 * n))
                window_length = max(window_length, 3)
                if window_length % 2 == 0:
                    window_length += 1
                if window_length > n:
                    window_length = n if n % 2 == 1 else n - 1
                batch[col] = savgol_filter(
                    data, window_length=window_length, polyorder=1)

            features = ["Relative_time", "Resonance_Frequency", "Dissipation"]
            X_live_batch = batch[features]
            prob_matrix_batch = self.predict(
                X_live_batch)  # shape: (batch_size, 5)

            # Decode the new batch using the last predicted state as initial condition.
            # predicted_batch = self.viterbi_decode(
            #     prob_matrix_batch,
            #     transition_matrix,
            #     smooth_window=5,   # adjust as needed
            #     initial_state=last_state
            # )
            # predicted_batch = self.greedy_decode(prob_matrix_batch,
            #                                      transition_matrix,
            #                                      smooth_window=5,   # adjust as needed
            #                                      initial_state=last_state)
            predicted_batch = self.posterior_decode(prob_matrix_batch,
                                                    transition_matrix,
                                                    smooth_window=5,   # adjust as needed
                                                    initial_state=last_state)
            # Update last_state for next batch
            last_state = predicted_batch[-1]
            # Append new predictions to full list
            full_predictions.extend(predicted_batch)

            # Update the live monitor plot using the full accumulated predictions.
            indices = np.arange(len(accumulated_data))
            axes[0].cla()
            axes[0].plot(indices, accumulated_data["Fill"], label="Actual", color='blue',
                         marker='o', linestyle='--', markersize=3)
            axes[0].plot(indices, full_predictions, label="Predicted", color='red',
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

            print(
                f"[INFO] Updated live monitor after batch {batch_num}. Total points: {len(accumulated_data)}")

        plt.ioff()
        plt.show()
        return full_predictions  # returns the full sequence of predictions


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    save_path = r"QModel\SavedModels\xgb_forecaster.json"
    test_dir = r"content\test_data"
    training_data_dir = r"content/training_data/full_fill"  # Update as needed

    # 1. Load and preprocess training data from multiple files.
    loader = DataLoader(data_to_load=np.inf)
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
