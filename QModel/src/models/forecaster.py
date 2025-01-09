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
import keras_tuner as kt
import random
from tqdm import tqdm


SEQUENCE_LEN = 200
TRAINING = False


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def load_content(data_dir: str) -> list:
    print(f"[INFO] Loading content from {data_dir}")
    loaded_content = []

    for data_root, _, data_files in tqdm(os.walk(data_dir), desc='Loading files...'):
        for f in data_files:
            if (
                f.endswith(".csv")
                and not f.endswith("_poi.csv")
                and not f.endswith("_lower.csv")
            ):
                matched_poi_file = f.replace(".csv", "_poi.csv")
                loaded_content.append((os.path.join(
                    data_root, f), os.path.join(data_root, matched_poi_file)))
    return loaded_content


def load_and_preprocess_data(data_dir):
    all_data = []

    content = load_content(data_dir)
    num_samples = 2
    sampled_content = random.sample(
        content, k=min(num_samples, len(content)))

    for file, poi in sampled_content:
        df = pd.read_csv(file)
        if not df.empty and 'Relative_time' in df.columns and 'Dissipation' in df.columns:
            df = df[['Relative_time', 'Dissipation']]
            all_data.append(df)

    if all_data:
        combined_data = pd.concat(all_data).sort_values(by='Relative_time')
    else:
        raise ValueError("No valid data found in the specified directory.")

    scaler = MinMaxScaler()
    combined_data['Dissipation'] = scaler.fit_transform(
        combined_data[['Dissipation']])
    print(combined_data)
    return combined_data, scaler


def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    sequences = np.array(sequences)
    targets = np.array(targets)
    sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
    return sequences, targets


def build_cnn_lstm_model(sequence_length, n_features):
    model = Sequential()

    # TimeDistributed CNN Layers
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                              input_shape=(sequence_length, n_features, 1)))
    # No reduction in dimension
    model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
    model.add(TimeDistributed(Flatten()))

    # LSTM Layer
    model.add(LSTM(50, activation='relu'))

    # Output Layer
    model.add(Dense(1))

    # Compile the Model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    model.summary()
    return model

# Training pipeline for the combined CNN-LSTM


def train_cnn_lstm_model(data_dir, sequence_length=30):
    # Load and preprocess data
    data, scaler = load_and_preprocess_data(data_dir)

    # Create sequences
    X, y = create_sequences(data['Dissipation'].values, sequence_length)

    # Reshape X to add a feature dimension for TimeDistributed CNN
    # (samples, sequence_length, features, channels)
    X = X.reshape((X.shape[0], X.shape[1], 1, 1))

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Build the CNN-LSTM model
    model = build_cnn_lstm_model(sequence_length, n_features=1)

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Save the model and scaler
    model.save("QModel/SavedModels/forecaster_cnn_lstm")
    joblib.dump(scaler, "QModel/SavedModels/forecaster_scaler.pkl")

    return model, scaler


def load_saved_model(model_path, scaler_path):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# Prediction pipeline
def predict_future_cnn_lstm(model, scaler, partial_data, sequence_length=30, num_predictions=10):
    # Scale partial data
    partial_data_scaled = scaler.transform(partial_data[['Dissipation']])

    # Create initial sequence
    sequence = partial_data_scaled[-sequence_length:].flatten()
    # Add feature and channel dimensions
    sequence = sequence.reshape((1, sequence_length, 1, 1))

    predictions = []
    for _ in range(num_predictions):
        # Predict next value
        predicted_value = model.predict(sequence)[0][0]
        predictions.append(predicted_value)

        sequence = sequence[:, 1:, :, :]  # Remove the first time step
        sequence = tf.concat([sequence, tf.constant(
            predicted_value, shape=(1, 1, 1, 1))], axis=1)

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1))
    return predictions


if __name__ == "__main__":
    data_dir = 'content/training_data/full_fill'
    if TRAINING:
        model, scaler = train_cnn_lstm_model(
            data_dir, sequence_length=SEQUENCE_LEN)

    full_data = pd.read_csv(
        r'C:\Users\QATCH\dev\QATCH-ML\content\training_data\full_fill\00990\DD240409W1_VOYBUF_I10_3rd.csv')
    model = tf.keras.models.load_model(
        "QModel/SavedModels/forecaster_cnn_lstm")
    scaler = joblib.load("QModel/SavedModels/forecaster_scaler.pkl")

    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(full_data))
    ax.set_ylim(0, 1)
    line_actual, = ax.plot([], [], label='Actual Data', color='blue')
    line_predicted, = ax.plot([], [], label='Predicted',
                              color='orange', linestyle='dashed')
    ax.legend()
    ax.set_title("Live Monitoring: Predictions vs Actual")
    ax.set_xlabel("Time")
    ax.set_ylabel("Dissipation")
    full_data_to_plt = normalize(full_data['Dissipation'].values)
    # Update function for animation
    x_data, y_actual, y_predicted = [], [], []
    predicted_poi = []

    # def update(frame):

    #     if frame > SEQUENCE_LEN:
    #         current_data = full_data.loc[:frame]
    #         future_predictions = predict_future_cnn_lstm(
    #             model, scaler, current_data, SEQUENCE_LEN, num_predictions=1
    #         )
    #         actual_value = full_data_to_plt[frame]
    #         predicted_val = (future_predictions.flatten()[0] -
    #                          np.min(full_data['Dissipation'].values)) / (np.max(full_data['Dissipation'].values) - np.min(full_data['Dissipation'].values))

    #         # Append data for plotting
    #         x_data.append(frame)
    #         y_actual.append(actual_value)
    #         y_predicted.append(predicted_val)
    #         predicted_poi.append(future_predictions.flatten()[0])

    #         # Update plot
    #         line_actual.set_data(x_data, y_actual)
    #         line_predicted.set_data(x_data, y_predicted)

    #         # Print to console
    #         print(
    #             f'Predicted {predicted_val}, was {actual_value}')

    #     return line_actual, line_predicted
    # ani = FuncAnimation(fig, update, frames=range(
    #     len(full_data)), blit=True, interval=50)
    # plt.show()

    for i in range(len(full_data)):
        if i > 200:
            future_predictions = predict_future_cnn_lstm(
                model, scaler, full_data.loc[:i], sequence_length=SEQUENCE_LEN, num_predictions=1)
            print(
                f'Predicted {future_predictions.flatten()}, was {full_data.loc[i]["Dissipation"]}')
            predicted_poi.append(future_predictions.flatten()[0])

    plt.figure()
    plt.plot(full_data_to_plt, color='grey')
    plt.plot(normalize(predicted_poi), color='black')
    plt.show()
