import keras_tuner as kt
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from collections import deque
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


NUM_CLASSES = 2
DATA_TO_LOAD = 100
OVERWRITE_HYPERPARAMS = True
TRAINING = False

# --- Reassign POI labels into 5 regions ---


def reassign_region(poi):
    if poi == 0:
        return 0
    elif poi in [1, 2, 3]:
        return 0
    elif poi == 4:
        return 0
    elif poi == 5:
        return 1
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
    Finally, reassign the 7 POI labels into 5 regions according to:
      - Region 0: from start of run to first POI (original POI 0)
      - Region 1: from POI 1 to POI 4 (original POI 1, 2, or 3)
      - Region 2: from POI 4 to POI 5 (original POI 4)
      - Region 3: from POI 5 to POI 6 (original POI 5)
      - Region 4: from POI 6 to end of run (original POI 6)
    """
    import os
    import random
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
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
                # Get the list of change indices (ensure they are sorted)
                change_indices = sorted(poi_df.iloc[:, 0].values)
                # Increment the POI value at each change index and thereafter.
                for idx in change_indices:
                    df.loc[idx:, "POI"] += 1
            # Recode POI as categorical integer codes.
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

    # Scale only the sensor features.
    scaler_features = MinMaxScaler()
    combined_data[["Relative_time", "Resonance_Frequency", "Dissipation"]] = scaler_features.fit_transform(
        combined_data[["Relative_time", "Resonance_Frequency", "Dissipation"]]
    )

    # --- Correct Class Imbalances via Oversampling ---
    class_counts = combined_data['POI'].value_counts()
    max_count = class_counts.max()
    balanced_dfs = []
    for label, group in combined_data.groupby('POI'):
        # Oversample groups with fewer samples than the maximum.
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

    combined_data['POI'] = combined_data['POI'].apply(
        reassign_region)

    print("[INFO] Class distribution after reassigning regions:")
    print(combined_data['POI'].value_counts())

    return combined_data, scaler_features


def load_and_preprocess_single(data_file: str, poi_file: str):
    """
    Load sensor data from the given CSV file and merge in POI information
    from the associated POI file. Sensor columns should include:
      - Relative_time
      - Resonance_Frequency
      - Dissipation

    The POI file can either have a 'POI' column directly or provide change indices.
    After processing, ensure the original POI column has exactly 7 unique values (0–6),
    scale the sensor features, and then reassign the POI values into 5 regions:
      - Region 0: from start of run to first POI (original POI 0)
      - Region 1: from POI 1 to POI 4 (original POI 1, 2, or 3)
      - Region 2: from POI 4 to POI 5 (original POI 4)
      - Region 3: from POI 5 to POI 6 (original POI 5)
      - Region 4: from POI 6 to end of run (original POI 6)
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    # Load the main sensor data.
    df = pd.read_csv(data_file)
    required_cols = ["Relative_time", "Resonance_Frequency", "Dissipation"]
    if df.empty or not all(col in df.columns for col in required_cols):
        raise ValueError(
            "Data file is empty or missing required sensor columns.")

    df = df[required_cols]

    # Load the POI data.
    poi_df = pd.read_csv(poi_file, header=None)
    if "POI" in poi_df.columns:
        df["POI"] = poi_df["POI"]
    else:
        # Start with 0 everywhere.
        df["POI"] = 0
        # Get the list of change indices (ensure they are sorted).
        change_indices = sorted(poi_df.iloc[:, 0].values)
        # For every change index, increment the POI value for that index and all later rows.
        for idx in change_indices:
            df.loc[idx:, "POI"] += 1

    # Recode POI as categorical integer codes.
    df["POI"] = pd.Categorical(df["POI"]).codes

    # --- Ensure exactly 7 unique POI values ---
    unique_poi = sorted(df["POI"].unique())
    if len(unique_poi) != 7:
        raise ValueError(
            f"Expected 7 unique POI values in file {data_file}, but found {len(unique_poi)}: {unique_poi}")

    # Scale only the sensor features.
    scaler = MinMaxScaler()
    df[required_cols] = scaler.fit_transform(df[required_cols])

    df["POI"] = df["POI"].apply(reassign_region)

    print("[INFO] Preprocessed single-file data sample:")
    print(df.head())
    return df, scaler


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


def train_with_tuning(combined_data, window_size=50, max_trials=10, executions_per_trial=2, tuner_dir='tuner_dir'):
    feature_list = []
    label_list = []

    # Create sliding windows from the combined data.
    for start in range(0, len(combined_data) - window_size):
        window = combined_data.iloc[start:start+window_size]
        # Sensor features as a 2D array: shape (window_size, 3)
        features = window[["Relative_time",
                           "Resonance_Frequency", "Dissipation"]].to_numpy()
        feature_list.append(features)
        label_list.append(window.iloc[-1]["POI"])

    # Convert lists to NumPy arrays.
    # Shape: (num_windows, window_size, num_features)
    X_train = np.array(feature_list)
    y_train = np.array(label_list)

    # Define input shape and number of classes.
    input_shape = X_train.shape[1:]  # (window_size, 3)
    num_classes = NUM_CLASSES  # Assuming POI classes are 0 through 6.

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

    # Define callbacks.
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Search for the best hyperparameters.
    tuner.search(X_train, y_train,
                 epochs=50,
                 batch_size=64,
                 validation_split=0.2,
                 callbacks=[early_stopping, reduce_lr],
                 verbose=1)

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]

    # Optionally, print a summary of the best hyperparameters.
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters:")
    print(best_hp.values)

    # Save the best model.
    best_model.save(
        'QModel/SavedModels/live_classifier_tf_tuned.h5', include_optimizer=False)

    print("[INFO] Best TensorFlow CNN-LSTM model trained and saved to 'QModel/SavedModels/live_classifier_tf_tuned.h5'")

    return best_model


def run_live_detection(model, combined_data: pd.DataFrame, window_size: int = 50, delay: float = 0.01):
    """
    Simulate live detection by streaming the preprocessed data row-by-row.
    A sliding window buffer is maintained; for each full window, features are
    extracted and fed into the CNN-LSTM model to obtain the full prediction probability
    vector. All returned probabilities (for classes 0 through 6) are tracked and plotted.
    A change in the argmax prediction is printed as an action transition.

    Two plots are created:
      - The top plot shows a sensor value (Dissipation).
      - The bottom plot shows:
            * Seven lines corresponding to the predicted probability for each class.
            * The actual POI (ground-truth) plotted on a twin y-axis.
    """
    from collections import deque
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Buffer for sliding window.
    data_buffer = deque(maxlen=window_size)
    previous_prediction = None
    current_prediction = None

    # Set up live plotting.
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

    # Top plot: Sensor Dissipation.
    line_diss, = ax1.plot([], [], label="Dissipation", color='blue')
    ax1.set_ylabel("Dissipation")
    ax1.legend()

    # Bottom plot (ax2): predicted probabilities for each class.
    predicted_prob_data = {i: [] for i in range(NUM_CLASSES)}
    predicted_lines = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    for i in range(NUM_CLASSES):
        line, = ax2.plot([], [], label=f"Predicted Class {i}",
                         marker='x', linestyle='--', color=colors[i])
        predicted_lines.append(line)
    ax2.set_ylabel("Predicted Probability")
    ax2.legend(loc="upper left")

    # Create a twin axis on ax2 for the actual POI.
    ax2_actual = ax2.twinx()
    line_actual, = ax2_actual.plot([], [], label="Actual POI",
                                   marker='o', linestyle='-', color='green')
    ax2_actual.set_ylabel("Actual POI")
    ax2_actual.legend(loc="upper right")

    # Lists to hold the x-axis values, dissipation, and actual POI for plotting.
    x_data = []
    diss_data = []
    actual_poi_data = []

    print("[INFO] Starting live action detection simulation with live plotting...")
    for idx, row in combined_data.iterrows():
        # Append the current sensor sample to the sliding window buffer.
        sample = {
            "Relative_time": row["Relative_time"],
            "Resonance_Frequency": row["Resonance_Frequency"],
            "Dissipation": row["Dissipation"]
        }
        data_buffer.append(sample)

        # Update plotting lists.
        x_data.append(idx)
        diss_data.append(row["Dissipation"])
        actual_poi_data.append(row["POI"])

        # When the buffer is full, run a prediction; otherwise, append NaN.
        if len(data_buffer) == window_size:
            window_df = pd.DataFrame(list(data_buffer))
            # For CNN-LSTM, keep the window as a 2D array with shape (window_size, num_features)
            features = window_df[["Relative_time",
                                  "Resonance_Frequency", "Dissipation"]].to_numpy()
            # Expand dims to have shape (1, window_size, 3)
            features = np.expand_dims(features, axis=0)
            preds = model.predict(features)
            # Record all predicted probabilities.
            for i in range(NUM_CLASSES):
                predicted_prob_data[i].append(preds[0][i])
            # Check for change in the maximum probability (argmax).
            new_prediction = int(np.argmax(preds, axis=1)[0])
            current_prediction = new_prediction
            if previous_prediction is None:
                previous_prediction = new_prediction
            if new_prediction != previous_prediction:
                print(
                    f"[Live] At index {idx}: Action transition detected -> New POI: {new_prediction}")
                previous_prediction = new_prediction
        else:
            # Buffer not full yet; append NaN for each class.
            for i in range(NUM_CLASSES):
                predicted_prob_data[i].append(np.nan)

        # Update the sensor dissipation line.
        line_diss.set_data(x_data, diss_data)
        # Update each predicted probability line.
        for i in range(NUM_CLASSES):
            predicted_lines[i].set_data(x_data, predicted_prob_data[i])
        # Update the actual POI line.
        line_actual.set_data(x_data, actual_poi_data)

        # Rescale axes to fit new data.
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

    # Define model save path (update as needed).
    model_save_path = 'QModel/SavedModels/live_classifier_tf_tuned.h5'

    if TRAINING:
        # 1. Load and preprocess data for training.
        combined_data, scaler = load_and_preprocess_data(data_dir)
        # 2. Train the TensorFlow classifier.
        best_model = train_with_tuning(combined_data, window_size=50)
        keras.models.save_model(best_model, model_save_path)

    else:
        # Load the trained TensorFlow model.
        best_model = keras.models.load_model(model_save_path)
        print(f"[INFO] Loaded TensorFlow model from {model_save_path}")

    # Now load a single file for live detection simulation.
    data_file = r"content/test_data/00003/MM231106W5_A1_40P_3rd.csv"
    poi_file = r"content/test_data/00003/MM231106W5_A1_40P_3rd_poi.csv"
    df = pd.read_csv(data_file)
    max_time = df['Relative_time'].max()
    delay = max_time / len(df)
    # Load and preprocess the data for live detection.
    combined_data, scaler = load_and_preprocess_single(data_file, poi_file)

    window_size = 50  # adjust based on the temporal resolution you need
    # x_data = []
    # diss_data = []
    # data_buffer = deque(maxlen=window_size)
    # previous_prediction = None
    # current_prediction = None
    # predicted_prob_data = {i: [] for i in range(NUM_CLASSES)}
    # actual_poi_data = []
    # for idx, row in combined_data.iterrows():
    #     # Append the current sensor sample to the sliding window buffer.
    #     sample = {
    #         "Relative_time": row["Relative_time"],
    #         "Resonance_Frequency": row["Resonance_Frequency"],
    #         "Dissipation": row["Dissipation"]
    #     }
    #     data_buffer.append(sample)

    #     # Update plotting lists.
    #     x_data.append(idx)
    #     diss_data.append(row["Dissipation"])
    #     actual_poi_data.append(row["POI"])
    #     if len(data_buffer) == window_size:
    #         window_df = pd.DataFrame(list(data_buffer))
    #         # For CNN-LSTM, keep the window as a 2D array with shape (window_size, num_features)
    #         features = window_df[["Relative_time",
    #                               "Resonance_Frequency", "Dissipation"]].to_numpy()
    #         # Expand dims to have shape (1, window_size, 3)
    #         features = np.expand_dims(features, axis=0)
    #         preds = best_model.predict(features)
    #         # Record all predicted probabilities.
    #         for i in range(NUM_CLASSES):
    #             predicted_prob_data[i].append(preds[0][i])
    #         # Check for change in the maximum probability (argmax).
    #         new_prediction = int(np.argmax(preds, axis=1)[0])
    #         current_prediction = new_prediction
    #         if previous_prediction is None:
    #             previous_prediction = new_prediction
    #         if new_prediction != previous_prediction:
    #             print(
    #                 f"[Live] At index {idx}: Action transition detected -> New POI: {new_prediction}")
    #             previous_prediction = new_prediction
    # Run the live detection simulation.
    run_live_detection(best_model, combined_data,
                       window_size=window_size, delay=delay)


if __name__ == "__main__":
    main()
