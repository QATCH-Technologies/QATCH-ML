from sklearn.preprocessing import RobustScaler
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks, regularizers
import matplotlib.pyplot as plt
from typing import Tuple

# ——— Robust preprocessing (no scaling here) ———


def load_all_runs(data_dir: str, max_runs: int = np.inf):
    pairs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".csv") and not f.endswith(("_poi.csv", "_lower.csv")):
                p_csv = os.path.join(root, f.replace(".csv", "_poi.csv"))
                if os.path.exists(p_csv):
                    pairs.append((os.path.join(root, f), p_csv))
    random.shuffle(pairs)
    return pairs if max_runs == np.inf else pairs[:int(max_runs)]


def preprocess_df(
    df: pd.DataFrame,
    clip_percentile: Tuple[float, float] = (1, 99),
    smooth_window: int = 5
) -> pd.DataFrame:
    """
    1) Sort & zero‑base time.
    2) Linear interp any gaps.
    3) IQR‑clip outliers.
    4) Median smooth.
    5) First‑order diffs.
    6) Rolling std and interaction features.
    """
    df = df.sort_values('Relative_time').reset_index(drop=True)
    df['Relative_time'] -= df['Relative_time'].iloc[0]
    df = df.interpolate(method='linear', limit_direction='both')

    # IQR clipping
    for c in ('Dissipation', 'Resonance_Frequency'):
        lo, hi = np.percentile(df[c], clip_percentile)
        df[c] = df[c].clip(lo, hi)

    # median smoothing
    for c in ('Dissipation', 'Resonance_Frequency'):
        df[c] = df[c].rolling(smooth_window, center=True,
                              min_periods=1).median()

    # first‑order diffs
    df['Dissipation_diff'] = df['Dissipation'].diff().fillna(0)
    df['Resonance_Frequency_diff'] = df['Resonance_Frequency'].diff().fillna(0)

    # rolling std
    df['Dissipation_std'] = df['Dissipation'].rolling(
        smooth_window, center=True, min_periods=1).std().fillna(0)
    df['Resonance_Frequency_std'] = df['Resonance_Frequency'].rolling(
        smooth_window, center=True, min_periods=1).std().fillna(0)

    # interaction term
    df['Diff_mul'] = df['Dissipation_diff'] * df['Resonance_Frequency_diff']

    return df


# ——— Data loading & global scaling ———
DS_FACTOR = 5
WINDOW_SIZE = 64


def parse_file_reg(
    data_path: str,
    poi_path: str,
    ds_factor: int = DS_FACTOR,
    clip_percentile: Tuple[float, float] = (1, 99),
    smooth_window: int = 5
) -> tuple[np.ndarray, int]:
    df = pd.read_csv(data_path)[
        ['Relative_time', 'Dissipation', 'Resonance_Frequency']]
    df = preprocess_df(df, clip_percentile, smooth_window)

    poi_raw = int(pd.read_csv(poi_path, header=None).values.squeeze()[-1])
    X = df[[
        'Relative_time',
        'Dissipation',
        'Resonance_Frequency',
        'Dissipation_diff',
        'Resonance_Frequency_diff',
        'Dissipation_std',
        'Resonance_Frequency_std',
        'Diff_mul'
    ]].values.astype(np.float32)

    X_ds = X[::ds_factor]
    poi_ds = poi_raw // ds_factor
    return X_ds, poi_ds


# Load raw runs
paths = load_all_runs("content/live/train", max_runs=200)
raw_runs = [parse_file_reg(d, p) for d, p in paths]

# Split train/val
split = int(0.8 * len(raw_runs))
train_raw, val_raw = raw_runs[:split], raw_runs[split:]

# Fit a global scaler on training data
all_train = np.vstack([X for X, _ in train_raw])
scaler = RobustScaler().fit(all_train)

# Apply global scaling
train_runs = [(scaler.transform(X), y) for X, y in train_raw]
val_runs = [(scaler.transform(X), y) for X, y in val_raw]

# ——— Dataset pipeline with time‑encoding ———


def make_regression_dataset(
    runs: list[tuple[np.ndarray, int]],
    batch_size: int = 64,
    window_size: int = WINDOW_SIZE
) -> tf.data.Dataset:
    n_features = runs[0][0].shape[1]

    def gen():
        for X_ds, poi_ds in runs:
            T = X_ds.shape[0]
            for t in range(window_size, T):
                win = X_ds[t-window_size:t].copy()
                # time encoding: seconds before now
                win[:, 0] -= win[-1, 0]
                steps = max(poi_ds - t, 0)
                yield win, np.log1p(steps)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(window_size, n_features), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
    )
    return ds.cache() \
             .shuffle(10_000) \
             .batch(batch_size) \
             .prefetch(tf.data.AUTOTUNE)


# Build datasets
ds_train = make_regression_dataset(train_runs, batch_size=128)
ds_val = make_regression_dataset(val_runs,   batch_size=128)

# ——— TCN‑inspired model ———


def tcn_block(x, filters, kernel_size=3, dilation_rate=1):
    prev = x
    x = layers.Conv1D(
        filters, kernel_size,
        padding='causal',
        dilation_rate=dilation_rate,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.Conv1D(
        filters, kernel_size,
        padding='causal',
        dilation_rate=dilation_rate,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    if prev.shape[-1] != filters:
        prev = layers.Dense(filters)(prev)
    x = layers.Add()([x, prev])
    return layers.Activation('relu')(x)


def build_regressor(window_size=WINDOW_SIZE, n_features=None):
    inp = layers.Input((window_size, n_features), name="window")
    x = inp
    # stack of dilated convs
    for rate in (1, 2, 4):
        x = tcn_block(x, filters=64, dilation_rate=rate)
        x = layers.SpatialDropout1D(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.LayerNormalization()(x)
    out = layers.Dense(1, activation='linear', name='log_steps')(x)

    optim = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model = models.Model(inp, out)
    model.compile(
        optimizer=optim,
        loss=losses.Huber(),
        metrics=['mae']
    )
    return model


# Instantiate & train
n_features = train_runs[0][0].shape[1]
reg_model = build_regressor(window_size=WINDOW_SIZE, n_features=n_features)
reg_model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=50,
    callbacks=[
        callbacks.EarlyStopping(
            monitor='val_mae', patience=5, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        ),
        callbacks.ModelCheckpoint(
            'best_reg.h5', monitor='val_loss', save_best_only=True
        )
    ]
)

# ——— Inference / live sim adjustment ———


def simulate_regression_live(
    data_path: str,
    model: tf.keras.Model,
    window_size: int = WINDOW_SIZE,
    ds_factor: int = DS_FACTOR
):
    # parse raw and scaled data separately
    X_ds_raw, poi_ds = parse_file_reg(
        data_path, data_path.replace('.csv', '_poi.csv'), ds_factor)
    X_ds = scaler.transform(X_ds_raw)

    # true stop time from raw
    if 0 <= poi_ds < len(X_ds_raw):
        stop_time = float(X_ds_raw[poi_ds, 0])
    else:
        stop_time = None

    buf, times, diss, preds, pred_times = [], [], [], [], []
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    for t_ds, (raw_row, scaled_row) in enumerate(zip(X_ds_raw, X_ds)):
        # current timestamp from raw data
        current_time = float(raw_row[0])
        times.append(current_time)
        # processed dissipation from scaled
        diss.append(float(scaled_row[1]))
        buf.append(scaled_row)
        if len(buf) > window_size:
            buf.pop(0)

        # predicted steps and predicted stop time
        if len(buf) == window_size:
            win = np.array(buf)[None, ...]
            log_pred = model.predict(win, verbose=0)[0, 0]
            pred_steps = max(np.expm1(log_pred), 0.0)
            # index and time for predicted stop
            pred_idx = min(int(round(t_ds + pred_steps)), len(X_ds_raw) - 1)
            pred_time = float(X_ds_raw[pred_idx, 0])
        else:
            pred_steps = float(max(poi_ds - t_ds, 0))
            pred_time = None

        preds.append(pred_steps)
        pred_times.append(pred_time)

        # ——— plot dissipation + true & predicted stops ———
        ax1.clear()
        ax1.set_title('Dissipation (processed)')
        ax1.plot(times, diss, label='Dissipation')
        if stop_time is not None:
            ax1.axvline(stop_time, color='r',
                        linestyle='--', label='True stop')
        if pred_time is not None:
            ax1.axvline(pred_time, color='b', linestyle='--',
                        label='Predicted stop')
        ax1.legend(loc='upper right')

        # ——— plot predicted steps ———
        ax2.clear()
        ax2.set_title('Predicted steps until stop')
        ax2.plot(times, preds, label='predicted steps')
        ax2.legend(loc='upper right')

        plt.pause(0.001)

    plt.ioff()
    plt.show()


# demo on a held‑out run
simulate_regression_live(
    'content/static/valid/02503/MM240625Y4_IGG200_2_3rd.csv', reg_model
)
