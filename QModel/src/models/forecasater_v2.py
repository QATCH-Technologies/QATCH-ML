from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xgboost as xgb
import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from q_constants import *
from hyperopt.early_stop import no_progress_loss

# ---------------------------
# New Helper Function: Downsampling Based on Time Delta
# ---------------------------


def downsample_before_time_delta(df: pd.DataFrame, factor: int = 20, threshold: float = 0.032) -> pd.DataFrame:
    """
    Downsamples all rows before the first significant time delta change in the 'Relative_time' column by the given factor.

    Parameters:
      df : pd.DataFrame
          DataFrame with a 'Relative_time' column.
      factor : int
          The downsampling factor (e.g., 20 means keep every 20th row before the change).
      threshold : float
          The threshold to detect a significant time delta.

    Returns:
      pd.DataFrame
          DataFrame with the early portion downsampled if a significant time delta is found.
    """
    df = df.copy()
    df["Delta"] = df["Relative_time"].diff()
    rolling_avg = df["Delta"].expanding(min_periods=2).mean()
    significant_change = (df["Delta"] - rolling_avg).abs() > threshold
    change_indices = df.index[significant_change].tolist()
    if change_indices:
        first_change_index = change_indices[0]
        # Convert the label to a positional index
        pos = df.index.get_loc(first_change_index)
        if pos > 0:
            df_before = df.iloc[:pos]
            df_after = df.iloc[pos:]
            df_before_downsampled = df_before.iloc[::factor]
            df = pd.concat([df_before_downsampled, df_after])
            df = df.sort_index().reset_index(drop=True)
    df = df.drop(columns=["Delta"])
    return df

# ---------------------------
# Data Handling Functions
# ---------------------------


DATA_TO_LOAD = 100  # adjust as needed

# For example, define a transition matrix that gives high probability to remaining in the same state,
# and a small probability to moving to the next state.
transition_matrix = None


def reassign_region(Fill):
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
        return Fill  # fallback if needed


def load_content(data_dir: str) -> list:
    print(f"[INFO] Loading content from {data_dir}")
    loaded_content = []
    for data_root, _, data_files in tqdm(os.walk(data_dir), desc='Loading files...'):
        for f in data_files:
            # Only process CSV files that are not already POI or lower threshold files
            if (
                f.endswith(".csv")
                and not f.endswith("_poi.csv")
                and not f.endswith("_lower.csv")
            ):
                matched_poi_file = f.replace(".csv", "_poi.csv")
                loaded_content.append(
                    (os.path.join(data_root, f), os.path.join(
                        data_root, matched_poi_file))
                )
    return loaded_content


def load_and_preprocess_data(data_dir: str):
    """
    Load sensor features and associated Fill information from multiple files.
    After merging, the 7 Fill labels are reassigned into 5 regions,
    class imbalances are corrected via oversampling, and any significant
    time delta in 'Relative_time' triggers downsampling of earlier data.
    """
    all_data = []
    content = load_content(data_dir)
    num_samples = DATA_TO_LOAD
    sampled_content = random.sample(content, k=min(num_samples, len(content)))

    for file, poi_path in sampled_content:
        df = pd.read_csv(file)
        if not df.empty and all(col in df.columns for col in ["Relative_time", "Resonance_Frequency", "Dissipation"]):
            df = df[["Relative_time", "Resonance_Frequency", "Dissipation"]]
            # Apply downsampling if a significant time delta is detected.
            df = downsample_before_time_delta(df, factor=20, threshold=0.032)

            Fill_df = pd.read_csv(poi_path, header=None)
            if "Fill" in Fill_df.columns:
                df["Fill"] = Fill_df["Fill"]
            else:
                # Start with 0 everywhere.
                df["Fill"] = 0
                change_indices = sorted(Fill_df.iloc[:, 0].values)
                for idx in change_indices:
                    df.loc[idx:, "Fill"] += 1
            df["Fill"] = pd.Categorical(df["Fill"]).codes
            all_data.append(df)

    if all_data:
        combined_data = pd.concat(all_data).sort_values(by='Relative_time')
    else:
        raise ValueError("No valid data found in the specified directory.")

    # --- Ensure exactly 7 unique Fill values ---
    unique_Fill = sorted(combined_data["Fill"].unique())
    if len(unique_Fill) != 7:
        raise ValueError(
            f"Expected 7 unique Fill values in combined data, but found {len(unique_Fill)}: {unique_Fill}")

    # After concatenating and sorting:
    combined_data['Fill'] = combined_data['Fill'].apply(reassign_region)

    # Map string labels to numeric values
    mapping = {'no_fill': 0, 'init_fill': 1,
               'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
    combined_data['Fill'] = combined_data['Fill'].map(mapping)

    # --- Correct Class Imbalances via Oversampling ---
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

    print("[INFO] Combined balanced data sample:")
    print(combined_data_balanced.head())
    print("[INFO] Class distribution after balancing:")
    print(combined_data_balanced['Fill'].value_counts())

    return combined_data_balanced


def load_and_preprocess_single(data_file: str, poi_file: str):
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

    # Reassign regions
    df["Fill"] = df["Fill"].apply(reassign_region)

    # Map string labels to numeric values
    mapping = {'no_fill': 0, 'init_fill': 1,
               'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
    df["Fill"] = df["Fill"].map(mapping)

    print("[INFO] Preprocessed single-file data sample:")
    print(df.head())
    return df


def tune_model_xgb(X, y):
    """
    Perform hyperparameter tuning using hyperopt to minimize mlogloss
    via xgb.cv and then train the final booster.
    """
    dtrain = xgb.DMatrix(X, label=y)

    def objective(params):
        params_copy = params.copy()
        params_copy.update({
            "objective": "multi:softprob",
            "num_class": 5,  # Add this line
            "eval_metric": "mlogloss",
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
    }

    trials = Trials()
    rng = np.random.default_rng(SEED)  # Use the new NumPy random generator

    best_result = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials,
        rstate=rng,
        early_stop_fn=no_progress_loss(10, percent_increase=0.1),
    )

    best_trial = trials.best_trial['result']
    best_params = best_trial['params']
    best_num_round = best_trial['best_iteration']

    print("Best hyperparameters found:", best_params)
    print("Best cross-validation mlogloss:", best_trial['loss'])

    final_model = xgb.train(best_params, dtrain,
                            num_boost_round=best_num_round)
    return final_model


def train_model(training_data):
    """
    Train the XGBoost model using the learning API.
    """
    features = ["Relative_time", "Resonance_Frequency", "Dissipation"]
    X = training_data[features]
    y = training_data["Fill"]

    # Use the custom tuning function to get a Booster model.
    model = tune_model_xgb(X, y)
    return model


# ---------------------------
# Live Monitor Plotting and Simulation Functions
# ---------------------------

def update_live_monitor(accumulated_data, predictions, axes, delay):
    """
    Update the live monitor plots:
      - Left: Predicted vs Actual fill values (numeric).
      - Right: Dissipation curve over the accumulated data.
    """
    indices = np.arange(len(accumulated_data))

    axes[0].cla()  # Clear previous plot
    axes[0].plot(indices, accumulated_data["Fill"], label="Actual", color='blue', marker='o',
                 linestyle='--', markersize=3)
    axes[0].plot(indices, predictions, label="Predicted", color='red', marker='x',
                 linestyle='-', markersize=3)
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


def simulate_serial_stream_from_loaded(loaded_data, batch_size=100):
    """
    Simulate a serial stream by yielding successive batches from the loaded single data.
    """
    num_rows = len(loaded_data)
    for start_idx in range(0, num_rows, batch_size):
        yield loaded_data.iloc[start_idx:start_idx+batch_size]


def viterbi_decode(prob_matrix, transition_matrix):
    """
    Decodes the most likely sequence of states given a probability matrix and a transition matrix.

    prob_matrix: shape (T, N) -- probability for each of the N states for each time step T.
    transition_matrix: shape (N, N) with allowed transitions.
       For our case, only transitions from state i to i or i+1 are allowed.
    """
    T, N = prob_matrix.shape
    dp = np.full((T, N), -np.inf)  # dynamic programming table (in log space)
    backpointer = np.zeros((T, N), dtype=int)

    # Initialization: assume the first sample is in state 0 (no_fill)
    dp[0, 0] = np.log(prob_matrix[0, 0])

    for t in range(1, T):
        for j in range(N):
            # Allowed transitions: for j==0, only state 0; otherwise, only from state j-1 and j.
            if j == 0:
                allowed_prev = [0]
            else:
                allowed_prev = [j - 1, j]

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


def live_prediction_loop_from_loaded_data(model, loaded_data, batch_size=100, delay=0.1):
    """
    Simulate live predictions by streaming data from the loaded data in batches.
    This version uses the model's probability outputs and applies Viterbi decoding
    to enforce the sequential constraints.

    Additionally, if a significant time delta is detected in the accumulated data,
    the data collected prior to that point is downsampled by a factor of 20.
    """
    accumulated_data = pd.DataFrame(columns=loaded_data.columns)
    predictions_history = []

    # Setup live monitor figure with two subplots.
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    stream_generator = simulate_serial_stream_from_loaded(
        loaded_data, batch_size=batch_size)
    batch_num = 0
    # Ensure we downsample only once when a time delta is detected.
    downsample_applied = False
    for batch in stream_generator:
        batch_num += 1
        print(
            f"\n[INFO] Streaming batch {batch_num} with {len(batch)} data points...")
        time.sleep(delay)  # Simulate delay between batches

        # Accumulate data
        accumulated_data = pd.concat(
            [accumulated_data, batch], ignore_index=True)

        # Check for significant time delta in the accumulated data if not already applied.
        if not downsample_applied:
            new_accumulated = downsample_before_time_delta(
                accumulated_data, factor=20, threshold=0.032)
            if len(new_accumulated) < len(accumulated_data):
                print(
                    "[INFO] Significant time delta detected. Downsampling early data by factor 20.")
                accumulated_data = new_accumulated
                downsample_applied = True

        features = ["Relative_time", "Resonance_Frequency", "Dissipation"]
        X_live = accumulated_data[features]

        # Get probability estimates from the model.
        f_names = model.feature_names
        df_live = X_live[f_names]
        d_data = xgb.DMatrix(df_live)
        prob_matrix = model.predict(d_data)  # shape: (num_points, 5)

        # Viterbi decoding enforces sequential constraints.
        predicted_sequence = viterbi_decode(prob_matrix, transition_matrix)
        predictions_history.append(predicted_sequence)

        # Update live monitor plots.
        update_live_monitor(accumulated_data, predicted_sequence, axes, delay)
        print(
            f"[INFO] Updated live monitor after batch {batch_num}. Total points: {len(accumulated_data)}")

    plt.ioff()
    plt.show()
    return predictions_history


def compute_dynamic_transition_matrix(training_data, num_states=5, smoothing=1e-6):
    """
    Compute the transition matrix from training data by counting transitions between states.

    Parameters:
      training_data : DataFrame
          Your training data containing a 'Fill' column with state labels (0 to num_states-1).
      num_states : int
          The number of unique states.
      smoothing : float
          A small value added to counts to avoid zero probabilities (Laplace smoothing).

    Returns:
      transition_matrix : np.ndarray
          A (num_states x num_states) transition probability matrix.
    """
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


# ---------------------------
# Main Execution
# ---------------------------

if __name__ == "__main__":
    save_path = r"QModel\SavedModels\xgb_forecaster.json"
    test_dir = r"content\test_data"
    # 1. Load and preprocess training data from multiple files:
    data_dir = "content/training_data/full_fill"  # <<< Update this path as needed
    training_data = load_and_preprocess_data(data_dir)
    transition_matrix = compute_dynamic_transition_matrix(training_data)
    if TRAINING:
        # 2. Train the classifier on the balanced training data using XGBoost
        model = train_model(training_data)
        print("[INFO] Model training complete.\n")
        model.save_model(save_path)
    if TESTING:
        model = xgb.Booster()
        model.load_model(save_path)
        # 3. Load a single run from test CSV files.
        test_content = load_content(test_dir)
        for data_file, poi_file in test_content:
            df_test = pd.read_csv(data_file)
            delay = df_test['Relative_time'].max(
            ) / len(df_test["Relative_time"].values)
            loaded_data = load_and_preprocess_single(data_file, poi_file)
            print(
                f"[INFO] Loaded single run data with {len(loaded_data)} data points.\n")

            # 4. Simulate live predictions by streaming the loaded data in batches:
            predictions_history = live_prediction_loop_from_loaded_data(
                model, loaded_data, batch_size=100, delay=delay/2)

            print("\n[INFO] Live prediction simulation complete.")
