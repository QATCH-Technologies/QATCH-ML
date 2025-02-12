import keras_tuner as kt
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from collections import deque
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Set constants
NUM_CLASSES = 2  # Adjust as needed for your application
DATA_TO_LOAD = 100
OVERWRITE_HYPERPARAMS = True
TRAINING = True

# New constant for slice size (every 100 data points)
SLICE_SIZE = 2

# --- Reassign POI labels into 5 regions ---


def reassign_region(poi):
    if poi == 0:
        return 0
    elif poi in [1, 2, 3]:
        return 0
    elif poi == 4:
        return 0
    elif poi == 5:
        return 0
    elif poi == 6:
        return 1
    else:
        return poi  # fallback if needed


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
    Load sensor features and associated POI information from multiple files.
    After merging and concatenating the data, ensure the POI column has exactly 7
    unique values (0–6). The sensor features are scaled.
    Additionally, correct any class imbalances by oversampling the minority classes.
    Finally, reassign the 7 POI labels into 5 regions.
    """
    import random
    from sklearn.preprocessing import MinMaxScaler

    # Load data from files.
    all_data = []
    content = load_content(data_dir)
    num_samples = DATA_TO_LOAD  # or process all files available
    sampled_content = random.sample(content, k=min(num_samples, len(content)))

    for file, poi_path in sampled_content:
        df = pd.read_csv(file)
        if not df.empty and all(col in df.columns for col in ["Relative_time", "Resonance_Frequency", "Dissipation"]):
            df = df[["Relative_time", "Resonance_Frequency", "Dissipation"]]
            poi_df = pd.read_csv(poi_path, header=None)
            if "POI" in poi_df.columns:
                df["POI"] = poi_df["POI"]
            else:
                # Start with 0 everywhere.
                df["POI"] = 0
                change_indices = sorted(poi_df.iloc[:, 0].values)
                for idx in change_indices:
                    df.loc[idx:, "POI"] += 1
            df["POI"] = pd.Categorical(df["POI"]).codes
            all_data.append(df)

    if all_data:
        combined_data = pd.concat(all_data).sort_values(by='Relative_time')
    else:
        raise ValueError("No valid data found in the specified directory.")

    # --- Ensure exactly 7 unique POI values ---
    unique_poi = sorted(combined_data["POI"].unique())
    if len(unique_poi) != 7:
        raise ValueError(
            f"Expected 7 unique POI values in combined data, but found {len(unique_poi)}: {unique_poi}")

    scaler_features = MinMaxScaler()
    combined_data[["Relative_time", "Resonance_Frequency", "Dissipation"]] = scaler_features.fit_transform(
        combined_data[["Relative_time", "Resonance_Frequency", "Dissipation"]]
    )
    combined_data['POI'] = combined_data['POI'].apply(reassign_region)
    # --- Correct Class Imbalances via Oversampling ---
    class_counts = combined_data['POI'].value_counts()
    max_count = class_counts.max()
    balanced_dfs = []
    for label, group in combined_data.groupby('POI'):
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
    print(combined_data_balanced['POI'].value_counts())

    print("[INFO] Class distribution after reassigning regions:")
    print(combined_data_balanced['POI'].value_counts())

    return combined_data_balanced, scaler_features


def load_and_preprocess_single(data_file: str, poi_file: str):
    """
    Load sensor data from a given CSV file and merge in POI information
    from the associated POI file. Then scale the sensor features and reassign
    the POI values into 5 regions.
    """
    from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv(data_file)
    required_cols = ["Relative_time", "Resonance_Frequency", "Dissipation"]
    if df.empty or not all(col in df.columns for col in required_cols):
        raise ValueError(
            "Data file is empty or missing required sensor columns.")

    df = df[required_cols]
    poi_df = pd.read_csv(poi_file, header=None)
    if "POI" in poi_df.columns:
        df["POI"] = poi_df["POI"]
    else:
        df["POI"] = 0
        change_indices = sorted(poi_df.iloc[:, 0].values)
        for idx in change_indices:
            df.loc[idx:, "POI"] += 1

    df["POI"] = pd.Categorical(df["POI"]).codes
    unique_poi = sorted(df["POI"].unique())
    if len(unique_poi) != 7:
        raise ValueError(
            f"Expected 7 unique POI values in file {data_file}, but found {len(unique_poi)}: {unique_poi}")

    scaler = MinMaxScaler()
    df[required_cols] = scaler.fit_transform(df[required_cols])
    df["POI"] = df["POI"].apply(reassign_region)

    print("[INFO] Preprocessed single-file data sample:")
    print(df.head())
    return df, scaler


def extract_window_features(window_df):
    """
    Given a window (DataFrame) of sensor samples, extract a feature vector.
    Here we simply flatten the three sensor channels.
    """
    features = window_df[[
        "Relative_time", "Resonance_Frequency", "Dissipation"]].to_numpy().flatten()
    return features


def build_model(hp, input_shape, num_classes):
    model = Sequential()
    # --- Convolutional Block ---
    model.add(Conv1D(
        filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size', values=[3, 5]),
        activation='relu',
        input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=hp.Choice('pool_size', values=[2, 3])))
    model.add(Dropout(rate=hp.Float('conv_dropout',
              min_value=0.2, max_value=0.5, step=0.1)))
    # --- LSTM Layer ---
    model.add(LSTM(units=hp.Int('lstm_units', min_value=32,
              max_value=128, step=32), return_sequences=False))
    # --- Dense Layers ---
    model.add(Dense(units=hp.Int('dense_units', min_value=50,
              max_value=150, step=25), activation='relu'))
    model.add(Dropout(rate=hp.Float('dense_dropout',
              min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=hp.Choice(
        'learning_rate', values=[1e-2, 1e-3, 1e-4]))
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_with_tuning(combined_data, window_size=SLICE_SIZE, slice_size=SLICE_SIZE,
                      max_trials=5, executions_per_trial=1, tuner_dir='tuner_dir'):
    """
    Instead of building sliding windows at every single sample, we now take
    non-overlapping slices (of length 'slice_size') from the combined data.
    Each slice (of length 'window_size') becomes one training sample.
    """
    feature_list = []
    label_list = []

    # Use non-overlapping slices every 'slice_size' data points.
    for start in range(0, len(combined_data) - window_size + 1, slice_size):
        window = combined_data.iloc[start:start+window_size]
        # Sensor features as a 2D array: shape (window_size, 3)
        features = window[["Relative_time",
                           "Resonance_Frequency", "Dissipation"]].to_numpy()
        feature_list.append(features)
        # Use the last point in the slice as the label.
        label_list.append(window.iloc[-1]["POI"])

    X_train = np.array(feature_list)
    y_train = np.array(label_list)

    input_shape = X_train.shape[1:]  # (window_size, 3)
    num_classes = NUM_CLASSES

    # Define the tuner.
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, input_shape, num_classes),
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=tuner_dir,
        project_name='cnn_lstm_tuning',
        overwrite=OVERWRITE_HYPERPARAMS
    )

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    tuner.search(X_train, y_train,
                 epochs=50,
                 batch_size=64,
                 validation_split=0.2,
                 callbacks=[early_stopping, reduce_lr],
                 verbose=1)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters:")
    print(best_hp.values)

    best_model.save(
        'QModel/SavedModels/live_classifier_tf_tuned.h5', include_optimizer=False)
    print("[INFO] Best TensorFlow CNN-LSTM model trained and saved to 'QModel/SavedModels/live_classifier_tf_tuned.h5'")
    return best_model


def run_live_detection(model, combined_data: pd.DataFrame, window_size=SLICE_SIZE, slice_size=SLICE_SIZE, delay: float = 0.01):
    """
    Simulate live detection by processing the data in chunks (slices) of 'slice_size'
    data points. For each slice (when at least 'window_size' rows are available),
    we take the last 'window_size' rows and run a prediction.
    The predictions and ground-truth are plotted as the simulation progresses.
    """
    import matplotlib.pyplot as plt

    # Prepare plotting elements.
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
    line_diss, = ax1.plot([], [], label="Dissipation", color='blue')
    ax1.set_ylabel("Dissipation")
    ax1.legend()

    predicted_prob_data = {i: [] for i in range(NUM_CLASSES)}
    predicted_lines = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    for i in range(NUM_CLASSES):
        line, = ax2.plot(
            [], [], label=f"Predicted Class {i}", marker='x', linestyle='--', color=colors[i])
        predicted_lines.append(line)
    ax2.set_ylabel("Predicted Probability")
    ax2.legend(loc="upper left")

    ax2_actual = ax2.twinx()
    line_actual, = ax2_actual.plot(
        [], [], label="Actual POI", marker='o', linestyle='-', color='green')
    ax2_actual.set_ylabel("Actual POI")
    ax2_actual.legend(loc="upper right")

    # Lists for x-axis and data.
    x_data = []
    diss_data = []
    actual_poi_data = []
    # Initialize predicted probability lists.
    for i in range(NUM_CLASSES):
        predicted_prob_data[i] = []

    print("[INFO] Starting live action detection simulation in slices...")

    # Process the data in slices of 'slice_size' rows.
    for slice_start in range(0, len(combined_data), slice_size):
        data_slice = combined_data.iloc[slice_start: slice_start + slice_size]
        # Update plotting lists for the entire slice.
        for idx, row in data_slice.iterrows():
            x_data.append(idx)
            diss_data.append(row["Dissipation"])
            actual_poi_data.append(row["POI"])
        # If there are enough points, take the last 'window_size' rows for prediction.
        if len(data_slice) >= window_size:
            window_df = data_slice.iloc[-window_size:]
            features = window_df[["Relative_time",
                                  "Resonance_Frequency", "Dissipation"]].to_numpy()
            # Shape: (1, window_size, 3)
            features = np.expand_dims(features, axis=0)
            preds = model.predict(features)
            new_prediction = int(np.argmax(preds, axis=1)[0])
            print(
                f"[Live] Data slice ending at index {slice_start + slice_size - 1}: Predicted POI = {new_prediction}")

            # For plotting, extend each predicted probability line by repeating the prediction value
            for i_class in range(NUM_CLASSES):
                predicted_prob_data[i_class].extend(
                    [preds[0][i_class]] * len(data_slice))
        else:
            # Not enough data for prediction; extend with NaN.
            for i_class in range(NUM_CLASSES):
                predicted_prob_data[i_class].extend([np.nan] * len(data_slice))

        # Update plot data.
        line_diss.set_data(x_data, diss_data)
        for i_class in range(NUM_CLASSES):
            predicted_lines[i_class].set_data(
                x_data, predicted_prob_data[i_class])
        line_actual.set_data(x_data, actual_poi_data)
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        ax2_actual.relim()
        ax2_actual.autoscale_view()
        plt.draw()
        plt.pause(delay)

    plt.ioff()
    plt.show()


def main():
    # Path to your training data directory.
    data_dir = "content/training_data/full_fill"  # update with your data directory
    model_save_path = 'QModel/SavedModels/live_classifier_tf_tuned.h5'

    if TRAINING:
        # Load and preprocess training data.
        combined_data, scaler = load_and_preprocess_data(data_dir)
        # Train the classifier using data slices of 100 data points.
        best_model = train_with_tuning(
            combined_data, window_size=SLICE_SIZE, slice_size=SLICE_SIZE)
        keras.models.save_model(best_model, model_save_path)
    else:
        # Load the trained model.
        best_model = keras.models.load_model(model_save_path)
        print(f"[INFO] Loaded TensorFlow model from {model_save_path}")

    # For live detection, load a single test file.
    data_file = r"content/test_data/00003/MM231106W5_A1_40P_3rd.csv"
    poi_file = r"content/test_data/00003/MM231106W5_A1_40P_3rd_poi.csv"
    df = pd.read_csv(data_file)
    max_time = df['Relative_time'].max()
    delay = max_time / len(df)
    combined_data, scaler = load_and_preprocess_single(data_file, poi_file)

    # Run live detection on slices of data.
    run_live_detection(best_model, combined_data,
                       window_size=SLICE_SIZE, slice_size=SLICE_SIZE, delay=delay)


if __name__ == "__main__":
    main()
