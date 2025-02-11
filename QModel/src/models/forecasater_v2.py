from collections import deque
import time
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

TRAINING = False


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


def load_and_preprocess_data(data_dir):
    all_data = []
    content = load_content(data_dir)
    num_samples = 10  # or use all available files
    sampled_content = random.sample(content, k=min(num_samples, len(content)))

    for file, poi_path in sampled_content:
        df = pd.read_csv(file)
        if not df.empty and all(col in df.columns for col in ["Relative_time", "Resonance_Frequency", "Dissipation"]):
            df = df[["Relative_time", "Resonance_Frequency", "Dissipation"]]
            # Load POI data
            poi_df = pd.read_csv(poi_path, header=None)
            if "POI" in poi_df.columns:
                df["POI"] = poi_df["POI"]
            else:
                # If the POI file contains indices where POI changes, create a POI column.
                df["POI"] = 0
                # assuming indices are stored in the first column
                change_indices = poi_df.iloc[:, 0].values
                new_values = list(range(1, len(change_indices)+1))
                last_index = 0
                for idx, new_val in zip(sorted(change_indices), new_values):
                    df.loc[last_index:idx, "POI"] = new_val
                    last_index = idx + 1
                df.loc[last_index:, "POI"] = new_values[-1]
            # Convert POI to categorical integer codes (0, 1, 2, …)
            df["POI"] = pd.Categorical(df["POI"]).codes
            all_data.append(df)

    if all_data:
        combined_data = pd.concat(all_data).sort_values(by='Relative_time')
    else:
        raise ValueError("No valid data found in the specified directory.")

    # Scale only the sensor features.
    scaler_features = MinMaxScaler()
    combined_data[["Relative_time", "Resonance_Frequency", "Dissipation"]] = scaler_features.fit_transform(
        combined_data[["Relative_time", "Resonance_Frequency", "Dissipation"]]
    )

    print("[INFO] Combined data sample:")
    print(combined_data.head())

    return combined_data, scaler_features


def load_and_preprocess_single(data_file: str, poi_file: str):
    """
    Load sensor data from the given CSV file and merge in POI information
    from the associated POI file. Sensor columns should include:
      - Relative_time
      - Resonance_Frequency
      - Dissipation

    The POI file can either have a 'POI' column directly, or it can provide
    indices where POI changes occur.
    """
    # Load the main sensor data
    df = pd.read_csv(data_file)
    required_cols = ["Relative_time", "Resonance_Frequency", "Dissipation"]
    if df.empty or not all(col in df.columns for col in required_cols):
        raise ValueError(
            "Data file is empty or missing required sensor columns.")

    df = df[required_cols]

    # Load the POI data
    poi_df = pd.read_csv(poi_file, header=None)
    if "POI" in poi_df.columns:
        df["POI"] = poi_df["POI"]
    else:
        # Assume the POI file contains change indices (one column)
        df["POI"] = 0
        # Get the indices at which the POI changes
        change_indices = poi_df.iloc[:, 0].values
        # Create new POI values for each change; starting at 1 (or adjust as needed)
        new_values = list(range(1, len(change_indices) + 1))
        last_index = 0
        for idx, new_val in zip(sorted(change_indices), new_values):
            df.loc[last_index:idx, "POI"] = new_val
            last_index = idx + 1
        df.loc[last_index:, "POI"] = new_values[-1]

    # Convert POI values to categorical integer codes (0, 1, 2, …)
    df["POI"] = pd.Categorical(df["POI"]).codes

    # Scale only the sensor features using MinMaxScaler
    scaler = MinMaxScaler()
    df[required_cols] = scaler.fit_transform(df[required_cols])

    print("[INFO] Preprocessed Data Sample:")
    print(df.head())
    return df, scaler
# Example usage:
# data_dir = "path/to/your/data"
# combined_data, scaler = load_and_preprocess_data(data_dir)


def extract_window_features(window_df):
    """
    Given a window (DataFrame) of sensor samples,
    extract a feature vector.

    Here we flatten the three sensor channels. In practice you might
    add summary statistics (mean, std) or frequency-domain features.
    """
    # We only use sensor features; label is not part of features.
    features = window_df[[
        "Relative_time", "Resonance_Frequency", "Dissipation"]].to_numpy().flatten()
    return features


def train_action_classifier(combined_data, window_size=50, model_save_path='action_classifier.pkl'):
    feature_list = []
    label_list = []

    # Create sliding windows. The window label is taken as the POI of the last row.
    for start in range(0, len(combined_data) - window_size):
        window = combined_data.iloc[start:start+window_size]
        features = extract_window_features(window)
        feature_list.append(features)
        label_list.append(window.iloc[-1]["POI"])

    X_train = np.array(feature_list)
    y_train = np.array(label_list)

    print(
        f"[INFO] Training on {X_train.shape[0]} windows with {X_train.shape[1]} features each.")

    # Train a RandomForest classifier.
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Save the trained model.
    joblib.dump(clf, model_save_path)
    print(f"[INFO] Model trained and saved to {model_save_path}")

    return clf


# Example usage:
# clf = train_action_classifier(combined_data)


def run_live_detection(clf: RandomForestClassifier, combined_data: pd.DataFrame,
                       window_size: int = 50, delay: float = 0.01):
    """
    Simulate live detection by streaming the preprocessed data row-by-row.
    A sliding window buffer is maintained; for each full window, features are
    extracted and fed into the classifier to predict the current action (POI).
    A change in prediction is treated as an action transition.

    Additionally, live plotting is implemented to show:
      - Dissipation (sensor value)
      - Actual POI (ground-truth from the data)
      - Predicted POI (from the classifier)
    """
    data_buffer = deque(maxlen=window_size)
    previous_prediction = None
    current_prediction = None

    # Set up live plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    # Plot for Dissipation
    line_diss, = ax1.plot([], [], label="Dissipation", color='blue')
    ax1.set_ylabel("Dissipation")
    ax1.legend()
    # Plot for Actual and Predicted POI
    line_actual, = ax2.plot([], [], label="Actual POI",
                            marker='o', linestyle='-', color='green')
    line_pred, = ax2.plot([], [], label="Predicted POI",
                          marker='x', linestyle='--', color='red')
    ax2.set_ylabel("POI")
    ax2.set_xlabel("Sample Index")
    ax2.legend()

    # Lists to hold data for plotting
    x_data = []
    diss_data = []
    actual_poi_data = []
    predicted_poi_data = []

    print("[INFO] Starting live action detection simulation with live plotting...")
    for idx, row in combined_data.iterrows():
        # Add the new sensor sample to the buffer
        sample = {
            "Relative_time": row["Relative_time"],
            "Resonance_Frequency": row["Resonance_Frequency"],
            "Dissipation": row["Dissipation"]
        }
        data_buffer.append(sample)

        # Update live data for plotting
        x_data.append(idx)
        diss_data.append(row["Dissipation"])
        actual_poi_data.append(row["POI"])
        # Use current_prediction if available; otherwise, show NaN
        predicted_val = current_prediction if current_prediction is not None else np.nan
        predicted_poi_data.append(predicted_val)

        # Once the buffer is full, perform prediction on the current window.
        if len(data_buffer) == window_size:
            window_df = pd.DataFrame(list(data_buffer))
            features = extract_window_features(window_df).reshape(1, -1)
            new_prediction = clf.predict(features)[0]
            current_prediction = new_prediction
            if previous_prediction is None:
                previous_prediction = new_prediction
            if new_prediction != previous_prediction:
                print(
                    f"[Live] At index {idx}: Action transition detected -> New POI: {new_prediction}")
                previous_prediction = new_prediction

        # Update the plot lines with the new data.
        line_diss.set_data(x_data, diss_data)
        line_actual.set_data(x_data, actual_poi_data)
        line_pred.set_data(x_data, predicted_poi_data)

        # Rescale axes to fit the new data.
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        plt.draw()
        plt.pause(delay)

    plt.ioff()
    plt.show()


def main():
    data_dir = "content/training_data/full_fill"  # update with your data directory
    if TRAINING:
        # 1. Load and preprocess data
        combined_data, scaler = load_and_preprocess_data(data_dir)

        # 2. Train the classifier
        clf = train_action_classifier(
            combined_data, window_size=50, model_save_path='QModel/SavedModels/live_classifier.pkl')

    data_file = r"content/test_data/00000/DD240125W1_A5_40P_3rd.csv"
    poi_file = r"content/test_data/00000/DD240125W1_A5_40P_3rd_poi.csv"

    # Load and preprocess the data
    combined_data, scaler = load_and_preprocess_single(data_file, poi_file)

    # Train the classifier using a sliding window approach
    window_size = 50  # adjust based on the temporal resolution you need
    clf = train_action_classifier(
        combined_data, window_size=window_size, model_save_path='QModel/SavedModels/live_classifier.pkl')

    # Run the live detection simulation
    run_live_detection(clf, combined_data, window_size=window_size, delay=0.01)


if __name__ == "__main__":
    main()
