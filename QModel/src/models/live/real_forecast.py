import os
import random
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# 1. Data loading


def load_content(data_dir: str, num_datasets: int = np.inf) -> List[Tuple[str, str]]:
    """
    Load dataset paths and corresponding POI files
    """
    if not os.path.exists(data_dir):
        logging.error("Data directory does not exist: %s", data_dir)
        return []
    loaded = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".csv") or f.endswith(("_poi.csv", "_lower.csv")):
                continue
            data_path = os.path.join(root, f)
            poi_path = data_path.replace(".csv", "_poi.csv")
            if os.path.exists(poi_path):
                loaded.append((data_path, poi_path))
    random.shuffle(loaded)
    return loaded if num_datasets == np.inf else loaded[:int(num_datasets)]

# 2. Preprocessing (only Dissipation)


def preprocess_files(
    file_pairs: List[Tuple[str, str]],
    window_size: int,
    forecast_horizon: int,
    scaler: MinMaxScaler = None
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Convert CSVs into sliding-window sequences for training.
    Only uses the 'Dissipation' signal.
    Returns X, y, and fitted scaler.
    """
    sequences, targets, all_data = [], [], []
    for data_path, _ in file_pairs:
        df = pd.read_csv(data_path)
        arr = df[['Dissipation']].values.astype(float)
        all_data.append(arr)
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(np.vstack(all_data))
    for arr in all_data:
        scaled = scaler.transform(arr)
        n = len(scaled)
        for i in range(window_size, n - forecast_horizon):
            sequences.append(scaled[i-window_size:i])
            targets.append(scaled[i:i+forecast_horizon])
    return np.array(sequences), np.array(targets), scaler

# 3. Model creation (single feature, full history input)


def create_forecast_model(
    forecast_horizon: int,
    latent_dim: int = 64
) -> Model:
    """
    Build a Seq2Seq LSTM forecaster on a single time series,
    accepting variable-length history at inference time.
    """
    enc_inputs = Input(shape=(None, 1), name='encoder_input')
    enc_out = LSTM(latent_dim, name='encoder_lstm')(enc_inputs)
    dec_input = RepeatVector(forecast_horizon, name='repeat_vector')(enc_out)
    dec_out = LSTM(latent_dim, return_sequences=True,
                   name='decoder_lstm')(dec_input)
    outputs = TimeDistributed(Dense(1), name='decoder_output')(dec_out)
    model = Model(enc_inputs, outputs, name='seq2seq_full_history')
    model.compile(optimizer='adam', loss='mse')
    return model

# 4. Training & Evaluation


def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 1,
    batch_size: int = 32
):
    """
    Train the model with early stopping and plot loss curves.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2
    )
    # Plot training & validation loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training History')
    plt.show()
    return history

# 5. Quick sanity check for sequence learning


def plot_prediction_sample(
    model: Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler: MinMaxScaler
):
    """
    Plot model forecast vs true for a random validation window.
    """
    idx = np.random.randint(0, len(X_val))
    x_sample = X_val[idx:idx+1]
    y_true = y_val[idx]
    y_pred = model.predict(x_sample)[0]
    # inverse scale
    y_true_i = scaler.inverse_transform(y_true)
    y_pred_i = scaler.inverse_transform(y_pred)
    plt.figure(figsize=(6, 4))
    plt.plot(y_true_i.flatten(), label='true')
    plt.plot(y_pred_i.flatten(), label='pred')
    plt.xlabel('Step ahead')
    plt.ylabel('Dissipation')
    plt.legend()
    plt.title('Validation Sample Forecast')
    plt.show()

# 6. Streaming plot with manual delays and sliding-window input Streaming plot with manual delays and sliding-window input


def simulate_and_plot(
    model: Model,
    test_file: str,
    window_size: int,
    forecast_horizon: int,
    scaler: MinMaxScaler,
    delay: float = 0.1
):
    """
    Stream full history for plotting but use only the last `window_size` points
    as model input so forecasts update with new data.
    """
    df = pd.read_csv(test_file)
    diss = df['Dissipation'].values.astype(float)
    rel_time = df['Relative_time'].values.astype(float)

    history_vals: List[float] = []
    history_times: List[float] = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line_hist, = ax.plot([], [], label='Diss History')
    line_pred, = ax.plot([], [], '--', label='Diss Forecast')

    ax.set_xlabel('Relative Time (s)')
    ax.set_ylabel('Dissipation')
    ax.set_title('History & Forecast')
    ax.legend()

    for i in range(len(diss)):
        history_vals.append(diss[i])
        history_times.append(rel_time[i])

        if len(history_vals) >= window_size:
            # take last window_size points for model input
            window_vals = history_vals[-window_size:]
            window_arr = np.array(window_vals).reshape(1, window_size, 1)
            # scale
            flat = window_arr.reshape(-1, 1)
            flat_s = scaler.transform(flat)
            inp_s = flat_s.reshape(1, window_size, 1)
            # forecast
            pred_s = model.predict(inp_s, verbose=0)[0]
            pred = scaler.inverse_transform(pred_s).flatten()

            # future times
            if i + forecast_horizon < len(df):
                ft = rel_time[i+1:i+1+forecast_horizon]
            else:
                dt = rel_time[1] - rel_time[0]
                ft = rel_time[i] + dt * np.arange(1, forecast_horizon+1)

            # update plot lines
            line_hist.set_data(history_times, history_vals)
            line_pred.set_data(ft, pred)

            # dynamic axes
            x_min = history_times[0]
            x_max = max(history_times[-1], ft[-1])
            ax.set_xlim(x_min, x_max + (rel_time[1]-rel_time[0]))
            all_y = np.concatenate([history_vals, pred])
            y_min, y_max = all_y.min(), all_y.max()
            pad = 0.05 * (y_max - y_min)
            ax.set_ylim(y_min - pad, y_max + pad)

        # draw and pause
        fig.canvas.draw()
        plt.pause(delay)

    plt.ioff()
    plt.show()


# 6. Example usage
if __name__ == "__main__":
    DATA_DIR = "content/live/train"
    WINDOW_SIZE = 50
    FORECAST_HORIZON = 10

    file_pairs = load_content(DATA_DIR, num_datasets=10)
    train_pairs = file_pairs[:-1]
    test_path = file_pairs[-1][0]

    X, y, scaler = preprocess_files(train_pairs, WINDOW_SIZE, FORECAST_HORIZON)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = create_forecast_model(FORECAST_HORIZON)
    train_model(model, X_train, y_train, X_val, y_val)
    simulate_and_plot(model, test_path, WINDOW_SIZE,
                      FORECAST_HORIZON, scaler, delay=0.1)
