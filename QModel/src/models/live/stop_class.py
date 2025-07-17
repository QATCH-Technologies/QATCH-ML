import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, losses, callbacks
import matplotlib.pyplot as plt
from typing import Tuple
import tensorflow_addons as tfa
# ——— Robust preprocessing ———


def preprocess_df(
    df: pd.DataFrame,
    clip_percentile: Tuple[float, float] = (1, 99),
    smooth_window: int = 5
) -> pd.DataFrame:
    """  
    1) Sorts by Relative_time and zero‑bases it.  
    2) Linear‐interpolates any gaps.  
    3) Clips outliers at given percentiles.  
    4) Smooths via rolling mean.  
    5) Scales Dissipation & Resonance_Frequency to [0,1].
    """
    # 1) sort & zero‑base time
    df = df.sort_values('Relative_time').reset_index(drop=True)
    df['Relative_time'] -= df['Relative_time'].iloc[0]

    # 2) fill small gaps
    df = df.interpolate(method='linear', limit_direction='both')

    # 3) clip outliers for numeric signals
    lower = df[['Dissipation', 'Resonance_Frequency']
               ].quantile(clip_percentile[0]/100)
    upper = df[['Dissipation', 'Resonance_Frequency']
               ].quantile(clip_percentile[1]/100)
    df['Dissipation'] = df['Dissipation'].clip(
        lower['Dissipation'], upper['Dissipation'])
    df['Resonance_Frequency'] = df['Resonance_Frequency'].clip(
        lower['Resonance_Frequency'], upper['Resonance_Frequency']
    )

    # 4) smooth signals
    df['Dissipation'] = (
        df['Dissipation']
        .rolling(window=smooth_window, center=True, min_periods=1)
        .mean()
    )
    df['Resonance_Frequency'] = (
        df['Resonance_Frequency']
        .rolling(window=smooth_window, center=True, min_periods=1)
        .mean()
    )

    df['Dissipation_diff'] = df['Dissipation'].diff().fillna(0)
    df['Resonance_Frequency_diff'] = df['Resonance_Frequency'].diff().fillna(0)

    # 5) scale to [0,1]
    scaler = MinMaxScaler()
    df[['Dissipation', 'Resonance_Frequency']] = scaler.fit_transform(
        df[['Dissipation', 'Resonance_Frequency']]
    )

    return df


def downsample(X: np.ndarray, factor: int = 5) -> np.ndarray:
    """Keep every `factor`‑th row of X along axis=0."""
    return X[::factor]


def load_paths(data_dir: str, max_datasets: int = np.inf):
    """Walk data_dir and return list of (data_csv, poi_csv)."""
    pairs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".csv") and not f.endswith(("_poi.csv", "_lower.csv")):
                data_path = os.path.join(root, f)
                poi_path = data_path.replace(".csv", "_poi.csv")
                if os.path.exists(poi_path):
                    pairs.append((data_path, poi_path))
    random.shuffle(pairs)
    return pairs if max_datasets == np.inf else pairs[:int(max_datasets)]


def parse_file(
    data_path: str,
    poi_path: str,
    ds_factor: int = 5,
    clip_percentile: Tuple[float, float] = (1, 99),
    smooth_window: int = 5
) -> Tuple[np.ndarray, int]:
    """
    Reads CSV + POI, applies robust preprocessing, downsamples, adjusts POI.
    Returns:
      X_ds:  (T_ds x 3) float32 array
      poi_ds: int index into the downsampled array
    """
    # raw load
    df = pd.read_csv(data_path)[
        ['Relative_time', 'Dissipation', 'Resonance_Frequency']]
    df = preprocess_df(df, clip_percentile=clip_percentile,
                       smooth_window=smooth_window)

    # extract features & POI
    X = df[['Relative_time', 'Dissipation', 'Resonance_Frequency']
           ].values.astype(np.float32)
    poi_raw = int(pd.read_csv(poi_path, header=None).values.squeeze()[-1])

    # downsample
    X_ds = downsample(X, factor=ds_factor)
    poi_ds = poi_raw // ds_factor
    return X_ds, poi_ds

# ——— windowing & model unchanged ———


WINDOW_SIZE = 256
STEP = 1


def make_windowed_dataset(runs, batch_size=64, neg_ratio=5):
    X_list, y_list = [], []

    for X_np, poi in runs:
        # convert and frame
        X = tf.convert_to_tensor(X_np)                   # (T, feat)
        T = tf.shape(X)[0]
        frames = tf.signal.frame(
            X,
            frame_length=WINDOW_SIZE,
            frame_step=STEP,
            axis=0
        )  # (num_frames, WINDOW_SIZE, feat)

        # generate matching labels
        num_frames = tf.shape(frames)[0]
        ends = tf.range(
            WINDOW_SIZE,
            WINDOW_SIZE + num_frames * STEP,
            STEP
        )                                         # (num_frames,)
        labels = tf.cast(ends == poi, tf.int32)  # (num_frames,)

        # split positives & negatives
        pos = tf.boolean_mask(frames, labels == 1)
        neg = tf.boolean_mask(frames, labels == 0)
        if tf.size(pos) == 0:
            continue
        # sample negatives
        neg = tf.random.shuffle(neg)[: neg_ratio * tf.shape(pos)[0]]

        X_list.append(tf.concat([pos, neg], axis=0))
        y_list.append(
            tf.concat(
                [tf.ones(tf.shape(pos)[0]), tf.zeros(tf.shape(neg)[0])],
                axis=0
            )
        )

    # build the dataset
    X_all = tf.concat(X_list, axis=0)
    y_all = tf.concat(y_list, axis=0)
    ds = tf.data.Dataset.from_tensor_slices((X_all, y_all))
    return ds.shuffle(10_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(window_size=WINDOW_SIZE):
    inp = layers.Input((window_size, 3), name="window")  # now 5 features
    x = layers.Conv1D(32, 3, padding="same", activation=None)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Bidirectional(layers.LSTM(
        64, return_sequences=False, dropout=0.3))(x)
    x = layers.Dense(64, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_stop")(x)
    model = models.Model(inp, out)
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
        gamma=2.0,       # focus parameter
        alpha=0.25,      # balance parameter
        from_logits=False
    )

    model.compile(
        optimizer="adam",
        loss=loss_fn,
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    return model

# ——— load, split, train ———


paths = load_paths("content/live/train", max_datasets=400)
runs = [
    parse_file(d, p, ds_factor=5, clip_percentile=(1, 99), smooth_window=5)
    for d, p in paths
]

split = int(0.8*len(runs))
train, val = runs[:split], runs[split:]

ds_train = make_windowed_dataset(train, batch_size=128)
ds_val = make_windowed_dataset(val,   batch_size=128)
cb = [
    callbacks.EarlyStopping(monitor="val_auc", mode="max",
                            patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=3),
    callbacks.ModelCheckpoint(
        "best_stop_model.h5", monitor="val_auc", mode="max", save_best_only=True)
]
model = build_model()
model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=30,
    class_weight={0: 1.0, 1: 500.0},
    callbacks=cb
)

# ——— live simulation unchanged ———


def simulate_live(
    data_path: str,
    model: tf.keras.Model,
    window_size: int = WINDOW_SIZE,
    ds_factor: int = 5
):
    # parse_file now returns (T, F) X_ds and poi_ds index
    X_ds, poi_ds = parse_file(
        data_path, data_path.replace(".csv", "_poi.csv"), ds_factor
    )

    # precompute the true stop time
    stop_time = X_ds[poi_ds, 0]

    buf, times, diss, preds = [], [], [], []
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    for row in X_ds:
        t, d = row[0], row[1]
        times.append(t)
        diss.append(d)
        buf.append(row)
        if len(buf) > window_size:
            buf.pop(0)

        if len(buf) == window_size:
            p = model.predict(np.array(buf)[None, ...], verbose=0)[0, 0]
        else:
            p = 0.0
        preds.append(p)

        ax1.clear()
        ax1.plot(times, diss, label="Dissipation (proc)")
        # use the static stop_time, not indexing into times
        ax1.axvline(stop_time, color="r", ls="--", label="True stop")
        ax1.legend(loc="upper right")

        ax2.clear()
        ax2.plot(times, preds, label="p_stop")
        ax2.axhline(0.5, color="k", ls=":", label="Threshold")
        ax2.legend(loc="upper right")

        plt.pause(0.001)

    plt.ioff()
    plt.show()


# simulate on a held‑out run
simulate_live(
    r"content/static/valid/02487/MM240625Y3_IGG50_2_3rd.csv", model)
