from sklearn.preprocessing import RobustScaler
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import matplotlib.pyplot as plt
from typing import Tuple
import keras_tuner as kt

# ——— Hyperparameters ———
EPOCHS = 10
DS_FACTOR = 5
WINDOW_SIZE = 64
NEAR_THRESHOLD = 10.0  # seconds to classify as 'near stop'

# ——— Robust preprocessing (no scaling here) ———


def load_all_runs(data_dir: str, max_runs: int = np.inf):
    pairs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.csv') and not f.endswith(('_poi.csv', '_lower.csv')):
                p_csv = os.path.join(root, f.replace('.csv', '_poi.csv'))
                if os.path.exists(p_csv):
                    pairs.append((os.path.join(root, f), p_csv))
    random.shuffle(pairs)
    return pairs if max_runs == np.inf else pairs[:int(max_runs)]


def preprocess_df(
    df: pd.DataFrame,
    clip_percentile: Tuple[float, float] = (1, 99),
    smooth_window: int = 5
) -> pd.DataFrame:
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

    # diffs and rolling std
    df['Dissipation_diff'] = df['Dissipation'].diff().fillna(0)
    df['Resonance_Frequency_diff'] = df['Resonance_Frequency'].diff().fillna(0)
    df['Dissipation_std'] = df['Dissipation'].rolling(
        smooth_window, center=True, min_periods=1).std().fillna(0)
    df['Resonance_Frequency_std'] = df['Resonance_Frequency'].rolling(
        smooth_window, center=True, min_periods=1).std().fillna(0)

    # interaction
    df['Diff_mul'] = df['Dissipation_diff'] * df['Resonance_Frequency_diff']
    return df

# ——— Inception‐style TCN block ———


def inception_tcn_block(x, filters, dilation_rate=1):
    convs = []
    for k in (3, 5, 7):
        convs.append(layers.Conv1D(
            filters, k,
            padding='causal',
            dilation_rate=dilation_rate,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4)
        )(x))
    x_cat = layers.Concatenate()(convs)
    x_proj = layers.Conv1D(filters, 1, padding='same',
                           activation='relu')(x_cat)

    # residual
    res = x
    if res.shape[-1] != filters:
        res = layers.Conv1D(filters, 1, padding='same')(res)
    out = layers.Add()([x_proj, res])
    return layers.Activation('relu')(out)

# ——— Data loading & parse ———


def parse_file_reg(
    data_path: str,
    poi_path: str,
    ds_factor: int = DS_FACTOR,
    clip_percentile: Tuple[float, float] = (1, 99),
    smooth_window: int = 5
) -> tuple[np.ndarray, float]:
    df = pd.read_csv(data_path)[
        ['Relative_time', 'Dissipation', 'Resonance_Frequency']]
    df = preprocess_df(df, clip_percentile, smooth_window)

    pois = pd.read_csv(poi_path, header=None).values.squeeze()
    if len(pois) < 6:
        raise ValueError(f"Expected ≥6 POIs, got {len(pois)}")
    poi_raw = int(pois[5])

    X = df[[
        'Relative_time', 'Dissipation', 'Resonance_Frequency',
        'Dissipation_diff', 'Resonance_Frequency_diff',
        'Dissipation_std', 'Resonance_Frequency_std', 'Diff_mul'
    ]].values.astype(np.float32)
    X_ds = X[::ds_factor]

    idx_ds = poi_raw // ds_factor
    if idx_ds < 0 or idx_ds >= len(X_ds):
        raise IndexError(f"Downsampled POI index {idx_ds} out of range")
    stop_time = float(X_ds[idx_ds, 0])

    return X_ds, stop_time


# ——— Prepare runs & scaling ———
paths = load_all_runs("content/live/train", max_runs=300)
raw_runs = [parse_file_reg(d, p) for d, p in paths]
split = int(0.8 * len(raw_runs))
train_raw = raw_runs[:split]
val_raw = raw_runs[split:]
all_train = np.vstack([X for X, _ in train_raw])
scaler = RobustScaler().fit(all_train)
train_runs = [(scaler.transform(X), stop) for X, stop in train_raw]
val_runs = [(scaler.transform(X), stop) for X, stop in val_raw]

# ——— Dataset w/ two-targets ———


def make_regression_dataset(
    runs: list[tuple[np.ndarray, float]],
    batch_size: int = 64,
    window_size: int = WINDOW_SIZE
) -> tf.data.Dataset:
    n_features = runs[0][0].shape[1]

    def gen():
        for X_ds, stop_time in runs:
            T = X_ds.shape[0]
            for t in range(window_size, T):
                win = X_ds[t-window_size:t].copy()
                current_time = win[-1, 0]
                win[:, 0] -= current_time
                time_to_stop = max(stop_time - current_time, 0.0)
                near_stop = float(time_to_stop <= NEAR_THRESHOLD)
                yield win, {'near_stop': near_stop, 'time_delta': time_to_stop}

    return (tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((window_size, n_features), tf.float32),
            {
                'near_stop':  tf.TensorSpec((), tf.float32),
                'time_delta': tf.TensorSpec((), tf.float32)
            }
        )
    )
        .cache().shuffle(10_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))


# build datasets
n_features = train_runs[0][0].shape[1]
ds_train = make_regression_dataset(train_runs, batch_size=128)
ds_val = make_regression_dataset(val_runs,   batch_size=128)

# ——— Two-head TCN model ———


def build_two_head_model(window_size=WINDOW_SIZE, n_features=None):
    inp = layers.Input((window_size, n_features), name="window")
    x = inp
    for rate in (1, 2, 4):
        x = inception_tcn_block(x, filters=64, dilation_rate=rate)
        x = layers.SpatialDropout1D(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    shared = layers.Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l2(1e-4))(x)
    shared = layers.LayerNormalization()(shared)

    c = layers.Dense(32, activation='relu')(shared)
    c = layers.Dense(1, activation='sigmoid', name='near_stop')(c)
    r = layers.Dense(32, activation='relu')(shared)
    r = layers.Dense(1, activation='linear', name='time_delta')(r)

    model = models.Model(inp, {'near_stop': c, 'time_delta': r})
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss={
            'near_stop':  'binary_crossentropy',
            'time_delta': 'mae'
        },
        metrics={
            'near_stop':  'accuracy',
            'time_delta': 'mae'
        },
        loss_weights={'near_stop': 1.0, 'time_delta': 1.0}
    )
    return model


# Instantiate & train
two_head_model = build_two_head_model(WINDOW_SIZE, n_features)
two_head_model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    callbacks=[
        callbacks.EarlyStopping('val_time_delta_mae',
                                patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(
            'val_time_delta_loss', factor=0.5, patience=3)
    ]
)

# ——— Live simulation ———


def simulate_regression_live(
    data_path: str,
    model: tf.keras.Model,
    window_size: int = WINDOW_SIZE,
    ds_factor: int = DS_FACTOR
):
    X_ds_raw, stop_time = parse_file_reg(
        data_path, data_path.replace('.csv', '_poi.csv'), ds_factor)
    X_ds = scaler.transform(X_ds_raw)

    buf, times, diss = [], [], []
    preds, pred_ts = [], []
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    for raw_row, scaled_row in zip(X_ds_raw, X_ds):
        current_time = float(raw_row[0])
        times.append(current_time)
        diss.append(float(scaled_row[1]))

        buf.append(scaled_row)
        if len(buf) > window_size:
            buf.pop(0)

        if len(buf) == window_size:
            win = np.array(buf)[None, ...]
            out_cls, out_reg = model.predict(win, verbose=0)
            pred_offset = float(out_reg[0, 0])
            pred_time = current_time + max(pred_offset, 0.0)

            preds.append(pred_offset)
            pred_ts.append(current_time)

        # Plotting
        ax1.clear()
        ax1.plot(times, diss, label='Dissipation')
        ax1.axvline(stop_time, color='r', ls='--', label='True stop')
        if preds:
            ax1.axvline(pred_time, color='b', ls='--', label='Predicted stop')
        ax1.legend(loc='upper right')

        ax2.clear()
        if preds:
            ax2.plot(pred_ts, preds, label='time to stop')
        ax2.set_title('Predicted seconds until stop')
        ax2.legend(loc='upper right')

        plt.pause(0.001)

    plt.ioff()
    plt.show()


# Demo
simulate_regression_live(
    'content/dropbox_dump/02480/MM240625Y4_PBS_1_3rd.csv', two_head_model
)
