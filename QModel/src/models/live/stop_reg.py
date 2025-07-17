from sklearn.preprocessing import RobustScaler
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks, regularizers
import matplotlib.pyplot as plt
from typing import Tuple


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
    5) First‑ and second‑order diffs.
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
    # second‑order diffs
    df['Dissipation_diff2'] = df['Dissipation_diff'].diff().fillna(0)
    df['RF_diff2'] = df['Resonance_Frequency_diff'].diff().fillna(0)

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
    # include both 1st and 2nd order diffs
    X = df[[
        'Relative_time',
        'Dissipation',
        'Resonance_Frequency',
        'Dissipation_diff',
        'Resonance_Frequency_diff',
        'Dissipation_diff2',
        'RF_diff2',
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
all_train_feat = np.vstack([X[:, 1:] for X, _ in train_raw])
scaler = RobustScaler().fit(all_train_feat)

# Helper to re‑inject the unscaled time column


def scale_with_time(X: np.ndarray) -> np.ndarray:
    scaled_feats = scaler.transform(X[:, 1:])
    return np.concatenate([X[:, :1], scaled_feats], axis=1)


# Apply to each run
train_runs = [(scale_with_time(X), y) for X, y in train_raw]
val_runs = [(scale_with_time(X), y) for X, y in val_raw]

# ——— Weighted generator to oversample near‑stop windows with extras ———


def gen_weighted(runs, window_size=WINDOW_SIZE):
    import numpy as _np
    for X_ds, poi_ds in runs:
        T = X_ds.shape[0]
        stop_time = X_ds[poi_ds, 0]
        windows, targets = [], []
        for t in range(window_size, T):
            win = X_ds[t-window_size:t].copy()
            current_time = win[-1, 0]
            time_to_stop = max(stop_time - current_time, 0.0)
            windows.append(win)
            targets.append(time_to_stop)
        targets = _np.array(targets)
        # weight more heavily the small gaps
        weights = 1.0 / (_np.log1p(targets) + 1.0)
        weights /= weights.sum()
        # sample one epoch worth
        for idx in _np.random.choice(len(windows), size=len(windows), p=weights):
            win = windows[idx]
            target = targets[idx]
            # window‑level extras
            area = _np.trapz(win[:, 1], x=win[:, 0])
            slope_change = win[:, 1].ptp()
            extras = _np.stack([
                _np.full(window_size, area),
                _np.full(window_size, slope_change)
            ], axis=1)
            win_ext = _np.concatenate([win, extras], axis=1)
            yield win_ext, _np.log1p(target)

# ——— Dataset pipeline with weighted sampling & extras ———


def make_regression_dataset(
    runs: list[tuple[np.ndarray, int]],
    batch_size: int = 64,
    window_size: int = WINDOW_SIZE
) -> tf.data.Dataset:
    base_n = runs[0][0].shape[1]
    n_features = base_n + 2  # + area and slope_change
    ds = tf.data.Dataset.from_generator(
        lambda: gen_weighted(runs, window_size),
        output_signature=(
            tf.TensorSpec((window_size, n_features), tf.float32),
            tf.TensorSpec((),               tf.float32),
        )
    )
    return (
        ds
        .cache()
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


# Build datasets
DS_TRAIN = make_regression_dataset(train_runs, batch_size=128)
DS_VAL = make_regression_dataset(val_runs,   batch_size=128)

# ——— TCN‑inspired model ———


def tcn_block(x, filters, kernel_size=5, dilation_rate=1):
    prev = x
    x = layers.Conv1D(filters, kernel_size,
                      padding='causal',
                      dilation_rate=dilation_rate,
                      activation='relu',
                      kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Conv1D(filters, kernel_size,
                      padding='causal',
                      dilation_rate=dilation_rate,
                      activation='relu',
                      kernel_regularizer=regularizers.l2(1e-5))(x)
    if prev.shape[-1] != filters:
        prev = layers.Dense(filters)(prev)
    x = layers.Add()([x, prev])
    return layers.Activation('relu')(x)


def build_regressor(window_size=WINDOW_SIZE, n_features=None):
    inp = layers.Input((window_size, n_features), name="window")
    x = inp
    for rate in (1, 2, 4, 8, 16):
        x = tcn_block(x, filters=128, dilation_rate=rate)
        x = layers.SpatialDropout1D(0.3)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64,
                     activation='relu',
                     kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.LayerNormalization()(x)
    out = layers.Dense(1,
                       activation='linear',
                       name='log_time')(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss=losses.Huber(),
        metrics=['mae']
    )
    return model


# Train
n_features = train_runs[0][0].shape[1] + 2
model = build_regressor(window_size=WINDOW_SIZE, n_features=n_features)
model.fit(
    DS_TRAIN,
    validation_data=DS_VAL,
    epochs=50,
    callbacks=[
        callbacks.EarlyStopping(
            monitor='val_mae', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        callbacks.ModelCheckpoint(
            'best_reg.h5', monitor='val_loss', save_best_only=True)
    ]
)


def simulate_regression_live(
    data_path: str,
    model: tf.keras.Model,
    window_size: int = WINDOW_SIZE,
    ds_factor: int = DS_FACTOR
):
    # parse & downsample raw
    X_ds_raw, poi_ds = parse_file_reg(
        data_path, data_path.replace('.csv', '_poi.csv'), ds_factor
    )
    # separate time & features
    times_raw = X_ds_raw[:, :1]
    feats_raw = X_ds_raw[:, 1:]
    # scale features
    feats_scaled = scaler.transform(feats_raw)
    # recombine
    X_ds = np.concatenate([times_raw, feats_scaled], axis=1)

    stop_time = float(X_ds_raw[poi_ds, 0]) if 0 <= poi_ds < len(
        X_ds_raw) else None

    buf, times, diss, preds = [], [], [], []
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    for i in range(len(X_ds_raw)):
        scaled_row = X_ds[i]
        current_time = float(X_ds_raw[i, 0])
        times.append(current_time)
        diss.append(float(scaled_row[1]))

        buf.append(scaled_row)
        if len(buf) > window_size:
            buf.pop(0)

        if len(buf) == window_size:
            win = np.array(buf, dtype=np.float32)
            # time‑encode
            win[:, 0] -= win[-1, 0]
            # compute extras
            area = np.trapz(win[:, 1], x=win[:, 0])
            slope_change = win[:, 1].ptp()
            extras = np.stack([
                np.full(window_size, area),
                np.full(window_size, slope_change)
            ], axis=1)
            win_ext = np.concatenate([win, extras], axis=1)[None, ...]

            log_pred = model.predict(win_ext, verbose=0)[0, 0]
            delta_t = max(np.expm1(log_pred), 0.0)
            pred_time = current_time + delta_t
        else:
            delta_t, pred_time = None, None

        preds.append(delta_t if delta_t is not None else 0.0)

        ax1.clear()
        ax1.plot(times, diss, label='Dissipation')
        if stop_time is not None:
            ax1.axvline(stop_time, color='r',
                        linestyle='--', label='True stop')
        if pred_time is not None:
            ax1.axvline(pred_time, color='b', linestyle='--',
                        label='Predicted stop')
        ax1.legend(loc='upper right')

        ax2.clear()
        ax2.plot(times, preds, label='Seconds to stop')
        ax2.legend(loc='upper right')
        plt.pause(0.001)

    plt.ioff()
    plt.show()


# content/static/valid/02503/MM240625Y4_IGG200_2_3rd.csv
# demo on a held‑out run
simulate_regression_live(
    'content/dropbox_dump/02480/MM240625Y4_PBS_1_3rd.csv', model
)
