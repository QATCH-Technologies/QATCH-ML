from scipy.signal import hilbert
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost as xgb  # Replacing RandomForestClassifier

# NEW IMPORTS for scaling
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Training Data Handling and Feature Computation
# ---------------------------
DATA_TO_LOAD = 20  # adjust as needed
IGNORE_BEFORE = 50
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


def compute_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    # Define parameters for rolling and EWMA calculations
    window = 10
    span = 10

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

    if 'Resonance_Frequency' in df.columns:
        df.drop(columns=['Resonance_Frequency'], inplace=True)

    t_delta = find_time_delta(df)
    if t_delta == -1:
        df['Time_shift'] = 0
    else:
        df.loc[t_delta:, 'Time_shift'] = 1

    return df


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
            if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                matched_poi_file = f.replace(".csv", "_poi.csv")
                loaded_content.append(
                    (os.path.join(data_root, f), os.path.join(data_root, matched_poi_file)))
    return loaded_content

# ---------------------------
# Split Training Data by Run Type
# ---------------------------


def load_and_preprocess_data_split(data_dir: str, required_runs=20):
    """
    Load each CSV (run) separately and split into two groups based on find_time_delta:
      - short_runs: runs where find_time_delta returns an index (i.e. != -1)
      - long_runs: runs where find_time_delta returns -1
    Continue loading data until exactly 'required_runs' are collected for each group.
    """
    short_runs = []
    long_runs = []
    content = load_content(data_dir)
    # Shuffle the list for random ordering.
    random.shuffle(content)

    for file, poi_file in content:
        # Break early if both groups have reached the required number of runs.
        if len(short_runs) >= required_runs and len(long_runs) >= required_runs:
            break

        df = pd.read_csv(file)
        if df.empty or not all(col in df.columns for col in ["Relative_time", "Resonance_Frequency", "Dissipation"]):
            continue  # Skip files missing required columns

        df = df[["Relative_time", "Resonance_Frequency", "Dissipation"]]
        df = compute_additional_features(df)

        Fill_df = pd.read_csv(poi_file, header=None)
        if "Fill" in Fill_df.columns:
            df["Fill"] = Fill_df["Fill"]
        else:
            # If the Fill information is not in a header, initialize with 0 and update using indices.
            df["Fill"] = 0
            change_indices = sorted(Fill_df.iloc[:, 0].values)
            for idx in change_indices:
                df.loc[idx:, "Fill"] += 1

        df["Fill"] = pd.Categorical(df["Fill"]).codes
        unique_Fill = sorted(df["Fill"].unique())
        if len(unique_Fill) != 7:
            print(
                f"[WARNING] File {file} does not have 7 unique Fill values; skipping.")
            continue

        # Reassign regions and map to numeric values.
        df['Fill'] = df['Fill'].apply(reassign_region)
        mapping = {'no_fill': 0, 'init_fill': 1,
                   'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
        df['Fill'] = df['Fill'].map(mapping)

        # Determine if the run is short (time delta detected) or long.
        delta_idx = find_time_delta(df)
        if delta_idx == -1:
            if len(short_runs) < required_runs:
                short_runs.append(df)
        else:
            if len(long_runs) < required_runs:
                long_runs.append(df)

        if len(df) > IGNORE_BEFORE:
            df = df.iloc[IGNORE_BEFORE:]

    if len(short_runs) < required_runs or len(long_runs) < required_runs:
        raise ValueError(
            f"Not enough runs found. Required: {required_runs} short and {required_runs} long, found: {len(short_runs)} short and {len(long_runs)} long.")

    training_data_short = pd.concat(short_runs).sort_values(
        "Relative_time").reset_index(drop=True)
    training_data_long = pd.concat(long_runs).sort_values(
        "Relative_time").reset_index(drop=True)
    return training_data_short, training_data_long

# ---------------------------
# Train the Model (unchanged)
# ---------------------------


def train_model_native(training_data, numerical_features=FEATURES, categorical_features=None, target='Fill'):
    # Separate features and target
    features = numerical_features + \
        (categorical_features if categorical_features else [])
    X = training_data[features].copy()
    y = training_data[target].values

    num_imputer = SimpleImputer(strategy='mean')
    X_num = num_imputer.fit_transform(X[numerical_features])
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    if categorical_features:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_cat = cat_imputer.fit_transform(X[categorical_features])
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_cat = encoder.fit_transform(X_cat)
        X_processed = np.hstack([X_num, X_cat])
    else:
        X_processed = X_num

    dtrain = xgb.DMatrix(X_processed, label=y)
    params = {
        'objective': 'multi:softprob',
        'num_class': 5,
        'eval_metric': 'aucpr',
        'max_depth': 5,
        'learning_rate': 0.1,
        'seed': 42
    }
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=200,
        nfold=5,
        metrics={'aucpr'},
        early_stopping_rounds=10,
        seed=42,
        verbose_eval=True
    )
    optimal_rounds = len(cv_results)
    model = xgb.train(params, dtrain, num_boost_round=optimal_rounds)
    preprocessors = {'num_imputer': num_imputer, 'scaler': scaler}
    if categorical_features:
        preprocessors.update({'cat_imputer': cat_imputer, 'encoder': encoder})
    return model, preprocessors


def predict_native(model, preprocessors, X, numerical_features=FEATURES, categorical_features=None):
    X = X.copy()
    X_num = preprocessors['num_imputer'].transform(X[numerical_features])
    X_num = preprocessors['scaler'].transform(X_num)
    if categorical_features:
        X_cat = preprocessors['cat_imputer'].transform(X[categorical_features])
        X_cat = preprocessors['encoder'].transform(X_cat)
        X_processed = np.hstack([X_num, X_cat])
    else:
        X_processed = X_num
    dmatrix = xgb.DMatrix(X_processed)
    prob_matrix = model.predict(dmatrix)
    return prob_matrix

# ---------------------------
# Live Monitor Plotting Functions
# ---------------------------


def update_live_monitor_two_models(accumulated_data, pred_short, pred_long, delta_idx, axes, delay):
    """
    Update live plots with predictions from both models and an indicator for time delta.
    """
    indices = np.arange(len(accumulated_data))
    axes[0].cla()
    axes[0].plot(indices, accumulated_data["Fill"], label="Actual",
                 color='blue', marker='o', linestyle='--', markersize=3)
    axes[0].plot(indices, pred_short, label="Short Model",
                 color='red', marker='x', linestyle='-', markersize=3)
    axes[0].plot(indices, pred_long, label="Long Model",
                 color='green', marker='^', linestyle='-.', markersize=3)
    if delta_idx != -1:
        axes[0].axvline(x=delta_idx, color='purple',
                        linestyle=':', label="Time Delta Detected")
    axes[0].set_title("Predicted vs Actual Fill")
    axes[0].set_xlabel("Data Point Index")
    axes[0].set_ylabel("Fill Category (numeric)")
    axes[0].legend()
    axes[0].set_ylim(-0.5, 4.5)

    axes[1].cla()
    axes[1].plot(indices, accumulated_data["Dissipation"],
                 label="Dissipation", color='green')
    if delta_idx != -1:
        axes[1].axvline(x=delta_idx, color='purple',
                        linestyle=':', label="Time Delta")
    axes[1].set_title("Dissipation Curve")
    axes[1].set_xlabel("Data Point Index")
    axes[1].set_ylabel("Dissipation")
    axes[1].legend()

    plt.pause(delay)


def simulate_serial_stream_from_loaded(loaded_data, batch_size=100):
    num_rows = len(loaded_data)
    for start_idx in range(0, num_rows, batch_size):
        yield loaded_data.iloc[start_idx:start_idx+batch_size]


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

# ---------------------------
# Live Prediction Loop for Two Models
# ---------------------------


def live_prediction_loop_two_models(model_short, preprocessors_short, model_long, preprocessors_long,
                                    transition_matrix_short, transition_matrix_long,
                                    loaded_data, batch_size=100, delay=0.1):
    """
    Simulate live predictions by streaming data from loaded_data in batches.
    Both the short-run and long-run models are applied and plotted.
    Each uses its corresponding transition matrix for Viterbi decoding.
    """
    accumulated_data = pd.DataFrame(columns=loaded_data.columns)
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    stream_generator = simulate_serial_stream_from_loaded(
        loaded_data, batch_size=batch_size)
    batch_num = 0

    for batch in stream_generator:
        batch_num += 1
        print(
            f"\n[INFO] Streaming batch {batch_num} with {len(batch)} data points...")
        time.sleep(delay)
        accumulated_data = pd.concat(
            [accumulated_data, batch], ignore_index=True)
        accumulated_data = compute_additional_features(accumulated_data)
        if len(accumulated_data) > IGNORE_BEFORE and batch_num == 1:
            accumulated_data = accumulated_data.iloc[IGNORE_BEFORE:]
        X_live = accumulated_data[FEATURES]
        # Get predictions from both models
        prob_matrix_short = predict_native(
            model_short, preprocessors_short, X_live, numerical_features=FEATURES)
        pred_short = viterbi_decode(prob_matrix_short, transition_matrix_short)
        prob_matrix_long = predict_native(
            model_long, preprocessors_long, X_live, numerical_features=FEATURES)
        pred_long = viterbi_decode(prob_matrix_long, transition_matrix_long)
        # Check for a time delta in the accumulated data
        delta_idx = find_time_delta(accumulated_data)
        update_live_monitor_two_models(
            accumulated_data, pred_short, pred_long, delta_idx, axes, delay)
        print(
            f"[INFO] Updated live monitor after batch {batch_num}. Total points: {len(accumulated_data)}")
    plt.ioff()
    plt.show()

# ---------------------------
# The Original find_time_delta Function (unchanged)
# ---------------------------


def find_time_delta(df) -> int:
    time_df = pd.DataFrame()
    time_df["Delta"] = df["Relative_time"].diff()
    threshold = 0.032
    rolling_avg = time_df["Delta"].expanding(min_periods=2).mean()
    time_df["Significant_change"] = (
        time_df["Delta"] - rolling_avg).abs() > threshold
    change_indices = time_df.index[time_df["Significant_change"]].tolist()
    return change_indices[0] if change_indices else -1

# ---------------------------
# Helper Function: load_and_preprocess_single remains unchanged
# ---------------------------


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
    unique_Fill = sorted(df["Fill"].unique())
    # if len(unique_Fill) != 7:
    #     raise ValueError(
    #         f"Expected 7 unique Fill values in file {data_file}, but found {len(unique_Fill)}: {unique_Fill}")
    df["Fill"] = df["Fill"].apply(reassign_region)
    mapping = {'no_fill': 0, 'init_fill': 1,
               'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
    df["Fill"] = df["Fill"].map(mapping)
    print("[INFO] Preprocessed single-file data sample:")
    print(df.head())
    return df


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


# ---------------------------
# Main Execution: Train Two Models and Run Live Predictions
# ---------------------------
if __name__ == "__main__":
    # 1. Load and split training data from multiple files:
    training_data_dir = r"content\training_data\full_fill"  # Update as needed
    training_data_short, training_data_long = load_and_preprocess_data_split(
        training_data_dir)

    # 2. Compute separate dynamic transition matrices.
    transition_matrix_short = compute_dynamic_transition_matrix(
        training_data_short)
    transition_matrix_long = compute_dynamic_transition_matrix(
        training_data_long)

    # 3. Train two separate models.
    model_short, preprocessors_short = train_model_native(training_data_short)
    model_long, preprocessors_long = train_model_native(training_data_long)
    print("[INFO] Model training complete.\n")

    # 4. For each test run, simulate live predictions using both models.
    test_dir = r"content\test_data"
    test_content = load_content(test_dir)
    random.shuffle(test_content)
    for data_file, poi_file in test_content:
        df_test = pd.read_csv(data_file)
        delay = df_test['Relative_time'].max(
        ) / len(df_test["Relative_time"].values)
        loaded_data = load_and_preprocess_single(data_file, poi_file)
        live_prediction_loop_two_models(model_short, preprocessors_short, model_long, preprocessors_long,
                                        transition_matrix_short, transition_matrix_long,
                                        loaded_data, batch_size=100, delay=delay/2)
        print("\n[INFO] Live prediction simulation complete for one test run.")
    print("\n[INFO] All live prediction simulations complete.")
