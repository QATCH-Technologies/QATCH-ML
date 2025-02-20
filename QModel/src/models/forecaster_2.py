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
from xgboost import XGBClassifier  # Replacing RandomForestClassifier

# NEW IMPORTS for scaling
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Training Data Handling
# ---------------------------
# For example, define a transition matrix that gives high probability to remaining in the same state,
# and a small probability to moving to the next state.
transition_matrix = None
DATA_TO_LOAD = 20  # adjust as needed

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
]


def compute_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute live features based on Dissipation.

    Existing features:
      - 'Dissipation': raw sensor reading.
      - 'Relative_time': time stamp.
      - 'Dissipation_rolling_mean': rolling average of Dissipation.
      - 'Dissipation_rolling_median': rolling median of Dissipation.
      - 'Dissipation_ewm': Exponential Weighted Moving Average (EWMA) of Dissipation.

    Additional features:
      - 'Dissipation_diff': First order difference of Dissipation.
      - 'Dissipation_pct_change': Percentage change of Dissipation.
      - 'Dissipation_rolling_std': Rolling standard deviation of Dissipation.
      - 'Dissipation_rate': Rate of change (Dissipation_diff / time difference).
      - 'Dissipation_ratio_to_mean': Ratio of Dissipation to its rolling mean.
      - 'Dissipation_ratio_to_ewm': Ratio of Dissipation to its EWMA.
      - 'Dissipation_envelope': Envelope of the Dissipation signal via the Hilbert transform.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing at least 'Dissipation' and 'Relative_time' columns.

    Returns:
        pd.DataFrame: DataFrame with additional features added.
    """

    # Ensure required columns are present
    required_cols = ['Dissipation', 'Relative_time']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Define parameters for rolling and EWMA calculations
    window = 10  # Adjust the window size if needed
    span = 10    # Adjust the span for the EWMA as required

    # Compute rolling statistics and EWMA for Dissipation
    df['Dissipation_rolling_mean'] = df['Dissipation'].rolling(
        window=window, min_periods=1).mean()
    df['Dissipation_rolling_median'] = df['Dissipation'].rolling(
        window=window, min_periods=1).median()
    df['Dissipation_ewm'] = df['Dissipation'].ewm(
        span=span, adjust=False).mean()
    df['Dissipation_rolling_std'] = df['Dissipation'].rolling(
        window=window, min_periods=1).std()

    # Compute additional features
    df['Dissipation_diff'] = df['Dissipation'].diff()
    df['Dissipation_pct_change'] = df['Dissipation'].pct_change()

    # Compute time difference for rate calculation (assuming Relative_time is numeric)
    df['Relative_time_diff'] = df['Relative_time'].diff().replace(0, np.nan)
    df['Dissipation_rate'] = df['Dissipation_diff'] / df['Relative_time_diff']
    df['Dissipation_ratio_to_mean'] = df['Dissipation'] / \
        df['Dissipation_rolling_mean']
    df['Dissipation_ratio_to_ewm'] = df['Dissipation'] / df['Dissipation_ewm']

    # Compute the envelope using the Hilbert transform.
    # Note: This computes the envelope over the entire signal, so it assumes you have access to all the data.
    df['Dissipation_envelope'] = np.abs(hilbert(df['Dissipation'].values))

    # Optionally, drop any intermediate columns if not needed
    if 'Resonance_Frequency' in df.columns:
        df.drop(columns=['Resonance_Frequency'], inplace=True)

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
    and class imbalances are corrected via oversampling.
    """
    all_data = []
    content = load_content(data_dir)
    num_samples = DATA_TO_LOAD
    sampled_content = random.sample(content, k=min(num_samples, len(content)))

    for file, Fill_path in sampled_content:
        df = pd.read_csv(file)
        if not df.empty and all(col in df.columns for col in ["Relative_time", "Resonance_Frequency", "Dissipation"]):
            df = df[["Relative_time", "Resonance_Frequency", "Dissipation"]]
            df = compute_additional_features(df)

            Fill_df = pd.read_csv(Fill_path, header=None)
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

    # >>> Scaling step removed <<<
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

    # --- Compute correlation matrix ---
    correlation_matrix = combined_data_balanced.corr()
    print("[INFO] Correlation matrix:")
    print(correlation_matrix)

    # --- Plot correlation heatmap ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Features and Fill")
    plt.show()

    return combined_data_balanced


def train_model(training_data, numerical_features=FEATURES, categorical_features=None, target='Fill'):
    """
    Train a classifier on the preprocessed training data using XGBoost with GPU acceleration.
    The pipeline includes imputation and scaling for numerical features and one-hot encoding for categorical features.
    Hyperparameter tuning is performed using GridSearchCV.

    Parameters:
        training_data (pd.DataFrame): The training dataset.
        numerical_features (list): List of numerical feature column names.
        categorical_features (list, optional): List of categorical feature column names.
        target (str): Name of the target column (default is 'Fill').

    Returns:
        Pipeline: The best fitted pipeline model.
    """

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier

    # Define preprocessing for numerical data
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for categorical data if provided
    if categorical_features:
        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, numerical_features),
                ('cat', cat_pipeline, categorical_features)
            ]
        )
    else:
        preprocessor = num_pipeline

    # Configure XGBClassifier to use GPU acceleration
    classifier = XGBClassifier(
        eval_metric='mlogloss',
        device='cuda',     # Enables GPU acceleration
        random_state=42
    )

    # Create the full pipeline with the classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Define hyperparameter grid for tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.1, 0.01]
    }

    # Separate features and target
    features = numerical_features + \
        (categorical_features if categorical_features else [])
    X = training_data[features]

    y = training_data[target]

    # Perform grid search with 5-fold cross-validation (n_jobs=-1 uses all CPU cores for parallel CV)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)

    # Return the best estimator found by grid search
    return grid_search.best_estimator_


# ---------------------------
# Live Monitor Plotting Function
# ---------------------------


def update_live_monitor(accumulated_data, predictions, axes, delay):
    """
    Update the live monitor plots:
      - Left: Predicted vs Actual fill values (numeric).
      - Right: Dissipation curve over the accumulated data.
    """
    # Now predictions and actual values are already numeric.
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

            # Initialize best_score and best_state with the first candidate.
            best_state = allowed_prev[0]
            best_score = dp[t - 1, best_state] + \
                np.log(transition_matrix[best_state, j])

            for i in allowed_prev:
                # Check if transition probability is positive to avoid log(0)
                if transition_matrix[i, j] <= 0:
                    continue
                score = dp[t - 1, i] + np.log(transition_matrix[i, j])
                if score > best_score:
                    best_score = score
                    best_state = i

            dp[t, j] = np.log(prob_matrix[t, j]) + best_score
            backpointer[t, j] = best_state

    # Backtracking to retrieve the best state sequence.
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
    """
    accumulated_data = pd.DataFrame(columns=loaded_data.columns)
    predictions_history = []

    # Setup live monitor figure with two subplots.
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    stream_generator = simulate_serial_stream_from_loaded(
        loaded_data, batch_size=batch_size)
    batch_num = 0
    for batch in stream_generator:
        batch_num += 1
        print(
            f"\n[INFO] Streaming batch {batch_num} with {len(batch)} data points...")
        time.sleep(delay)  # Simulate delay between batches

        # Accumulate data
        accumulated_data = pd.concat(
            [accumulated_data, batch], ignore_index=True)
        accumulated_data = compute_additional_features(accumulated_data)

        X_live = accumulated_data[FEATURES]
        # The pipeline inside the model automatically scales X_live before prediction.
        prob_matrix = model.predict_proba(X_live)  # shape: (num_points, 5)
        # Use Viterbi decoding to enforce that the state sequence only moves forward.
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
    # Extract the state sequence from the 'Fill' column
    states = training_data["Fill"].values

    # Initialize the transition count matrix with zeros.
    # Only allowed transitions will be updated.
    transition_counts = np.zeros((num_states, num_states))

    # Apply smoothing only for the allowed transitions: same state and next state.
    for i in range(num_states):
        transition_counts[i, i] = smoothing  # self-transition
        if i + 1 < num_states:
            # transition to next state
            transition_counts[i, i+1] = smoothing

    # Count transitions between consecutive states.
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        # Only count transitions that are allowed (same state or next state)
        if next_state == current_state or next_state == current_state + 1:
            transition_counts[current_state, next_state] += 1

    # Normalize each row so that the probabilities sum to 1.
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    # To avoid division by zero in case a row has zero sum, we can set such rows to 1.
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_counts / row_sums

    return transition_matrix


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
    if len(unique_Fill) != 7:
        raise ValueError(
            f"Expected 7 unique Fill values in file {data_file}, but found {len(unique_Fill)}: {unique_Fill}")

    # Reassign regions
    df["Fill"] = df["Fill"].apply(reassign_region)

    # Map string labels to numeric values
    mapping = {'no_fill': 0, 'init_fill': 1,
               'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
    df["Fill"] = df["Fill"].map(mapping)

    print("[INFO] Preprocessed single-file data sample:")
    print(df.head())
    return df

# ---------------------------
# Main Execution
# ---------------------------


if __name__ == "__main__":
    # 1. Load and preprocess training data from multiple files:
    save_path = r"QModel\SavedModels\xgb_forecaster.json"
    test_dir = r"content\test_data"
    training_data_dir = r"content\training_data\full_fill"  # Update as needed
    training_data = load_and_preprocess_data(training_data_dir)
    transition_matrix = compute_dynamic_transition_matrix(training_data)
    # 2. Train the classifier on the balanced training data using XGBoost with scaling.
    model = train_model(training_data)
    print("[INFO] Model training complete.\n")

    # 3. Load a single run from test CSV files instead of using mock data.
    data_file = r"content/test_data/00003/MM231106W5_A1_40P_3rd.csv"
    poi_file = r"content/test_data/00003/MM231106W5_A1_40P_3rd_poi.csv"
    df_test = pd.read_csv(data_file)
    delay = df_test['Relative_time'].max(
    ) / len(df_test["Relative_time"].values)

    # 4. Simulate live predictions by streaming the loaded data in batches:
    test_content = load_content(test_dir)
    random.shuffle(test_content)
    for data_file, poi_file in test_content:
        df_test = pd.read_csv(data_file)
        delay = df_test['Relative_time'].max(
        ) / len(df_test["Relative_time"].values)
        loaded_data = load_and_preprocess_single(
            data_file, poi_file)
        live_prediction_loop_from_loaded_data(
            model, loaded_data, batch_size=100, delay=delay/2)
        print("\n[INFO] Live prediction simulation complete.")
    print("\n[INFO] Live prediction simulation complete.")
