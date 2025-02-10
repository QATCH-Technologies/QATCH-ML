from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, Dense
from keras.optimizers import Adam
from keras.models import Sequential
import joblib
import random
from tqdm import tqdm

SEQUENCE_LEN = 200
TRAINING = True  # Set True to retrain the model

# -----------------------------------------------------------------------------
# Utility: Load list of CSV files along with their corresponding POI file.

# -----------------------------------------------------------------------------
# Custom callback for live training monitoring.


class LiveTrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LiveTrainingMonitor, self).__init__()
        self.epoch_count = []
        self.train_loss = []
        self.val_loss = []
        # Turn on interactive mode.
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Live Training Monitor")
        self.fig.show()
        self.fig.canvas.draw()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_count.append(epoch)
        self.train_loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))

        self.ax.clear()
        self.ax.plot(self.epoch_count, self.train_loss,
                     label="Training Loss", marker='o')
        self.ax.plot(self.epoch_count, self.val_loss,
                     label="Validation Loss", marker='o')
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Live Training Monitor")
        self.ax.legend()
        self.fig.canvas.draw()
        plt.pause(0.01)


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
                    (os.path.join(data_root, f), os.path.join(data_root, matched_poi_file)))
    return loaded_content

# -----------------------------------------------------------------------------
# Data Loading and Preprocessing:
#  - Load sensor features from the main file.
#  - Merge in the POI file (which contains categorical labels).
#  - Factorize POI so they are integer codes.
#  - Scale sensor features.


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
            poi_df = pd.read_csv(poi_path)
            if "POI" in poi_df.columns:
                df["POI"] = poi_df["POI"]
            else:
                # If the POI file contains indices where POI changes, create a POI column.
                df["POI"] = 0
                # assuming indices are stored here
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
        combined_data[["Relative_time", "Resonance_Frequency", "Dissipation"]])

    print("[INFO] Combined data sample:")
    print(combined_data.head())

    return combined_data, scaler_features

# -----------------------------------------------------------------------------
# Create sequences from sensor features along with the next POI label as target.


def create_sequences(features, target, sequence_length):
    sequences = []
    targets = []
    for i in range(len(features) - sequence_length):
        sequences.append(features[i:i + sequence_length])
        targets.append(target[i + sequence_length])
    # shape: (num_samples, sequence_length, num_features)
    sequences = np.array(sequences)
    targets = np.array(targets)      # shape: (num_samples,)
    return sequences, targets

# -----------------------------------------------------------------------------
# Build the CNN-LSTM model for classification.


def build_cnn_lstm_model(sequence_length, n_features, n_classes):
    model = Sequential()
    # TimeDistributed CNN: input shape is (sequence_length, n_features, 1)
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                              input_shape=(sequence_length, n_features, 1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
    model.add(TimeDistributed(Flatten()))
    # LSTM layer
    model.add(LSTM(50, activation='relu'))
    # Final Dense layer with softmax activation for classification.
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',  # targets are integer-coded
                  metrics=['accuracy'])
    model.summary()
    return model

# -----------------------------------------------------------------------------
# Training pipeline.


# Training pipeline.
def train_cnn_lstm_model(data_dir, sequence_length=30):
    data, scaler_features = load_and_preprocess_data(data_dir)
    features = data[["Relative_time",
                     "Resonance_Frequency", "Dissipation"]].values
    target = data["POI"].values
    n_classes = len(np.unique(target))

    # Create sequences for training.
    X, y = create_sequences(features, target, sequence_length)
    # Reshape X to add a channel dimension: (samples, sequence_length, n_features, 1)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

    # Train-validation split.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Build the model.
    model = build_cnn_lstm_model(
        sequence_length, n_features=3, n_classes=n_classes)

    # Set up callbacks: early stopping and our live training monitor.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    live_monitor = LiveTrainingMonitor()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,  # Adjust epochs as needed.
        batch_size=32,
        callbacks=[early_stopping, live_monitor]
    )

    # Save model and scaler.
    model.save("QModel/SavedModels/forecaster_cnn_lstm")
    joblib.dump(scaler_features,
                "QModel/SavedModels/forecaster_scaler_features.pkl")
    # Also save the number of classes.
    with open("QModel/SavedModels/n_classes.txt", "w") as f:
        f.write(str(n_classes))

    return model, scaler_features, n_classes
# -----------------------------------------------------------------------------
# Load saved model and associated scaler and number of classes.


def load_saved_model(model_path, scaler_features_path, n_classes_path):
    model = tf.keras.models.load_model(model_path)
    scaler_features = joblib.load(scaler_features_path)
    with open(n_classes_path, "r") as f:
        n_classes = int(f.read())
    return model, scaler_features, n_classes

# -----------------------------------------------------------------------------
# Prediction pipeline:
# Given partial data (sensor features), predict the next POI class.


def predict_future_cnn_lstm(model, scaler_features, partial_data, sequence_length=30):
    partial_data_scaled = scaler_features.transform(
        partial_data[["Relative_time", "Resonance_Frequency", "Dissipation"]])
    # shape: (sequence_length, 3)
    sequence = partial_data_scaled[-sequence_length:]
    sequence = sequence.reshape((1, sequence_length, 3, 1))

    probabilities = model.predict(sequence)
    predicted_class = np.argmax(probabilities, axis=-1)[0]
    return predicted_class


# -----------------------------------------------------------------------------
# Main block.
if __name__ == "__main__":
    data_dir = 'content/training_data/full_fill'

    if TRAINING:
        model, scaler_features, n_classes = train_cnn_lstm_model(
            data_dir, sequence_length=SEQUENCE_LEN)
    else:
        model, scaler_features, n_classes = load_saved_model(
            r"QModel/SavedModels/forecaster_cnn_lstm",
            r"QModel/SavedModels/forecaster_scaler_features.pkl",
            r"QModel/SavedModels/n_classes.txt"
        )

    # For live monitoring, load a full CSV file with sensor data and its corresponding POI file.
    full_data = pd.read_csv(
        r'C:\Users\paulm\dev\QATCH\QATCH-ML\content\training_data\00913\MM230816W6_DIAT20C_2_3rd.csv')
    poi_path = r'C:\Users\paulm\dev\QATCH\QATCH-ML\content\training_data\00913\MM230816W6_DIAT20C_2_3rd_poi.csv'
    poi_df = pd.read_csv(poi_path)
    if "POI" in poi_df.columns:
        full_data["POI"] = poi_df["POI"]
    else:
        full_data["POI"] = 0
        change_indices = poi_df.iloc[:, 0].values
        new_values = list(range(1, len(change_indices)+1))
        last_index = 0
        for idx, new_val in zip(sorted(change_indices), new_values):
            full_data.loc[last_index:idx, "POI"] = new_val
            last_index = idx + 1
        full_data.loc[last_index:, "POI"] = new_values[-1]
    # Factorize POI in the full data so that the labels match training.
    full_data["POI"] = pd.Categorical(full_data["POI"]).codes

    # Scale the sensor features in full_data.
    full_data_scaled = full_data.copy()
    full_data_scaled[["Relative_time", "Resonance_Frequency", "Dissipation"]] = scaler_features.transform(
        full_data[["Relative_time", "Resonance_Frequency", "Dissipation"]])

    # Initialize live monitoring plot for predictions.
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(full_data))
    ax.set_ylim(-0.5, n_classes - 0.5)  # y-axis shows categorical indices
    line_actual, = ax.plot([], [], label='Actual POI', color='blue')
    line_predicted, = ax.plot(
        [], [], label='Predicted POI', color='orange', linestyle='dashed')
    ax.legend()
    ax.set_title("Live Monitoring: Predicted vs Actual POI")
    ax.set_xlabel("Time (index)")
    ax.set_ylabel("POI Class")

    x_data, y_actual, y_predicted = [], [], []

    def update(frame):
        if frame > SEQUENCE_LEN:
            # Use all rows up to the current frame.
            current_data = full_data_scaled.iloc[:frame]
            predicted_poi = predict_future_cnn_lstm(
                model, scaler_features, current_data, sequence_length=SEQUENCE_LEN)
            actual_poi = full_data.iloc[frame]["POI"]

            x_data.append(frame)
            y_actual.append(actual_poi)
            y_predicted.append(predicted_poi)

            line_actual.set_data(x_data, y_actual)
            line_predicted.set_data(x_data, y_predicted)

            print(
                f"Frame {frame}: Predicted POI = {predicted_poi}, Actual POI = {actual_poi}")
        return line_actual, line_predicted

    ani = FuncAnimation(fig, update, frames=range(
        len(full_data)), blit=True, interval=50)
    plt.show()

    # Optionally, plot the full actual vs. predicted POI after the live demo.
    all_predicted = []
    for i in range(len(full_data)):
        if i > SEQUENCE_LEN:
            predicted = predict_future_cnn_lstm(
                model, scaler_features, full_data_scaled.iloc[:i], sequence_length=SEQUENCE_LEN)
            all_predicted.append(predicted)
        else:
            all_predicted.append(np.nan)

    plt.figure()
    plt.plot(full_data["POI"], color='grey', label="Actual POI")
    plt.plot(all_predicted, color='black', label="Predicted POI")
    plt.title("Forecasted POI vs Actual POI")
    plt.xlabel("Time (index)")
    plt.ylabel("POI Class")
    plt.legend()
    plt.show()
