import math
from typing import Sequence, Optional
from pathlib import Path
import pickle
import argparse
import platform
import sys
from sklearn.preprocessing import RobustScaler, StandardScaler
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks, regularizers
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.model_selection import KFold
from keras import mixed_precision
import tensorflow_addons as tfa
TRAINING = True
# ——— SIGNIFICANTLY IMPROVED HYPERPARAMETERS ———
EPOCHS = 2  # Increase training epochs
DS_FACTOR = 5  # Reduce downsampling to preserve important details
WINDOW_SIZE = 512  # Larger window for better temporal context
NEAR_THRESHOLD = 1.0  # More reasonable threshold
BATCH_SIZE = 64  # Larger batches for stable gradients
LEARNING_RATE = 5e-4  # Better learning rate
DROPOUT_RATE = 0.2  # Reduced dropout for better learning
L2_REG = 1e-6  # Lighter regularization
PATIENCE = 5  # More patience for convergence

print("Py:", sys.version)
print("TF:", tf.__version__)
print("Built w/ CUDA:", tf.test.is_built_with_cuda())
print("Physical GPUs:", tf.config.list_physical_devices('GPU'))

# Enable mixed precision for better performance
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")
except:
    print("Mixed precision not available, using float32")


def load_all_runs(data_dir: str,
                  max_runs: int = np.inf,
                  bin_sec: float = 15.0,  # Larger bins for better balance
                  per_bin: int = None,
                  mode: str = "undersample",
                  assign_strategy: str = "median"  # More stable assignment
                  ):
    """Improved file selector with better time balancing"""
    rng = random.Random(42)

    candidates = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.csv') and not f.endswith(('_poi.csv', '_lower.csv')):
                p_csv = os.path.join(root, f.replace('.csv', '_poi.csv'))
                if os.path.exists(p_csv):
                    candidates.append((os.path.join(root, f), p_csv))

    if max_runs != np.inf and len(candidates) <= max_runs:
        rng.shuffle(candidates)
        return candidates[:int(max_runs)]

    # Bin runs by duration for balanced training
    bins = {}
    for d_csv, p_csv in candidates:
        try:
            df = pd.read_csv(d_csv, usecols=['Relative_time'])
            rel = df['Relative_time'].to_numpy()
            rel = rel - rel[0]

            if assign_strategy == "median":
                b = int(np.median(rel) // bin_sec)
                bins.setdefault(b, []).append((d_csv, p_csv))
            else:
                b0 = int(rel.min() // bin_sec)
                b1 = int(rel.max() // bin_sec)
                for b in range(b0, b1 + 1):
                    bins.setdefault(b, []).append((d_csv, p_csv))
        except Exception as e:
            print(f"Skipping {d_csv}: {e}")
            continue

    if not bins:
        rng.shuffle(candidates)
        return candidates if max_runs == np.inf else candidates[:int(max_runs)]

    sizes = [len(v) for v in bins.values() if v]
    if not sizes:
        rng.shuffle(candidates)
        return candidates if max_runs == np.inf else candidates[:int(max_runs)]

    if per_bin is None:
        per_bin = min(int(np.median(sizes) * 1.5),
                      max(sizes))  # Better bin sizing

    balanced = []
    for b, lst in bins.items():
        if not lst:
            continue
        if mode == "undersample":
            take = min(per_bin, len(lst))
            balanced.extend(rng.sample(lst, take))
        elif mode == "oversample":
            take = max(per_bin, len(lst))
            if len(lst) < take:
                balanced.extend(lst + rng.choices(lst, k=take - len(lst)))
            else:
                balanced.extend(rng.sample(lst, take))

    rng.shuffle(balanced)
    seen, uniq = set(), []
    for pair in balanced:
        if pair not in seen:
            seen.add(pair)
            uniq.append(pair)
            if len(uniq) >= (int(max_runs) if max_runs != np.inf else 10**12):
                break

    return uniq


def _rolling_slope_fast(y: pd.Series, window: int) -> pd.Series:
    """
    Least-squares slope of y vs. x (x = 0..window-1) for each rolling window.
    Matches np.polyfit(..., 1)[0] exactly for equally spaced x.
    """
    n = window
    if n <= 1:
        return pd.Series(np.zeros(len(y)), index=y.index)

    # constants for x
    x = np.arange(n, dtype=np.float64)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = n * sum_x2 - sum_x ** 2  # constant

    yv = y.to_numpy(dtype=np.float64)
    idx = np.arange(len(yv), dtype=np.float64)

    # prefix sums
    csum_y = np.concatenate(([0.0], np.cumsum(yv)))
    csum_xy = np.concatenate(([0.0], np.cumsum(yv * idx)))

    out = np.full(len(yv), np.nan, dtype=np.float64)

    # compute for full windows only (to match original behavior -> NaNs then fillna(0))
    for end in range(n, len(yv) + 1):
        start = end - n
        sum_yw = csum_y[end] - csum_y[start]
        sum_xyw = csum_xy[end] - csum_xy[start]

        # shift x each window: effective x are 0..n-1 so we must compensate the absolute idx
        # sum_xyw currently uses absolute idx; convert to relative:
        # sum( (i-start) * y_i ) = sum_xyw - start * sum_yw
        rel_sum_xyw = sum_xyw - start * sum_yw

        num = n * rel_sum_xyw - sum_x * sum_yw
        out[end - 1] = num / denom

    return pd.Series(out, index=y.index).fillna(0.0)


def plot_run_features(
    df: pd.DataFrame,
    time_col: str = "Relative_time",
    include: Optional[Sequence[str]] = None,
    exclude: Sequence[str] = (),
    cols_per_row: int = 4,
    sharex: bool = True,
    standardize: bool = False,
    figsize_per_plot: tuple = (3.2, 2.0),
    tight: bool = True,
):
    """
    Plot all (or selected) numeric features vs time.

    Parameters
    ----------
    df : DataFrame
        Output of enhanced_preprocess_df (must contain time_col).
    time_col : str
        Column to use on the x-axis.
    include : list[str] | None
        If provided, only these columns will be plotted (besides time_col).
    exclude : list[str]
        Columns to skip (patterns allowed via substring match).
    cols_per_row : int
        Number of subplots per row.
    sharex : bool
        If True, subplots share the x-axis.
    standardize : bool
        If True, z-score each series (per column) to comparable scales.
    figsize_per_plot : (w, h)
        Size of each subplot cell.
    tight : bool
        If True, call tight_layout at the end.

    Returns
    -------
    fig, axes : matplotlib Figure and ndarray of Axes
    """
    # -------- pick feature columns --------
    num_cols = df.select_dtypes("number").columns.tolist()
    if include is not None:
        feat_cols = [c for c in include if c in num_cols and c != time_col]
    else:
        feat_cols = [c for c in num_cols if c != time_col]

    if exclude:
        feat_cols = [c for c in feat_cols if not any(
            ex in c for ex in exclude)]

    n = len(feat_cols)
    if n == 0:
        raise ValueError("No features to plot after filtering.")

    # -------- standardize if requested --------
    plot_df = df[[time_col] + feat_cols].copy()
    if standardize:
        for c in feat_cols:
            s = plot_df[c]
            std = s.std(ddof=0)
            plot_df[c] = (s - s.mean()) / (std if std != 0 else 1.0)

    # -------- layout --------
    rows = math.ceil(n / cols_per_row)
    fig_w = cols_per_row * figsize_per_plot[0]
    fig_h = rows * figsize_per_plot[1]
    fig, axes = plt.subplots(
        rows, cols_per_row, figsize=(fig_w, fig_h), sharex=sharex)
    axes = axes.ravel() if n > 1 else [axes]

    x = plot_df[time_col].to_numpy()

    for i, col in enumerate(feat_cols):
        ax = axes[i]
        ax.plot(x, plot_df[col].to_numpy(), linewidth=1)
        ax.set_title(col, fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        if i % cols_per_row == 0:
            ax.set_ylabel(col if standardize else "")
        if i >= (rows - 1) * cols_per_row:
            ax.set_xlabel(time_col, fontsize=8)

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if tight:
        fig.tight_layout(pad=1.1)

    return fig, axes


def enhanced_preprocess_df(
    df: pd.DataFrame,
    clip_percentile: Tuple[float, float] = (0.5, 99.5),
    smooth_window: int = 3
) -> pd.DataFrame:
    """Enhanced preprocessing with better feature engineering (faster version, same outputs)."""

    df = df.sort_values('Relative_time').reset_index(drop=True)
    df['Relative_time'] -= df['Relative_time'].iloc[0]

    # Interpolation (kept cubic to preserve returns)
    df = df.interpolate(method='cubic', limit_direction='both')

    # Percentile clip (vectorized loop)
    for c in ('Dissipation', 'Resonance_Frequency'):
        lo, hi = np.percentile(df[c].to_numpy(), clip_percentile)
        df[c] = df[c].clip(lo, hi)

    # Minimal smoothing
    for c in ('Dissipation', 'Resonance_Frequency'):
        df[c] = df[c].rolling(smooth_window, center=True, min_periods=1).mean()

    # First & second diffs
    df['Dissipation_diff'] = df['Dissipation'].diff().fillna(0)
    df['Resonance_Frequency_diff'] = df['Resonance_Frequency'].diff().fillna(0)

    df['Dissipation_diff2'] = df['Dissipation_diff'].diff().fillna(0)
    df['Resonance_Frequency_diff2'] = df['Resonance_Frequency_diff'].diff().fillna(0)

    # Rolling std & slopes (use fast slope)
    for window in (5, 10, 20):
        diss_roll = df['Dissipation'].rolling(window, min_periods=1)
        freq_roll = df['Resonance_Frequency'].rolling(window, min_periods=1)

        df[f'Dissipation_std_{window}'] = diss_roll.std().fillna(0)
        df[f'Resonance_Frequency_std_{window}'] = freq_roll.std().fillna(0)

        # slopes
        df[f'Dissipation_slope_{window}'] = _rolling_slope_fast(
            df['Dissipation'], window)
        df[f'Resonance_Frequency_slope_{window}'] = _rolling_slope_fast(
            df['Resonance_Frequency'], window)

    # Cross features
    df['Ratio'] = df['Dissipation'] / (df['Resonance_Frequency'] + 1e-8)
    df['Ratio_diff'] = df['Ratio'].diff().fillna(0)

    # Moving averages & deviations
    for window in (10, 20):
        ma_d = df['Dissipation'].rolling(window, min_periods=1).mean()
        ma_f = df['Resonance_Frequency'].rolling(window, min_periods=1).mean()

        df[f'Dissipation_ma_{window}'] = ma_d
        df[f'Resonance_Frequency_ma_{window}'] = ma_f
        df[f'Dissipation_dev_{window}'] = df['Dissipation'] - ma_d
        df[f'Resonance_Frequency_dev_{window}'] = df['Resonance_Frequency'] - ma_f

    return df


def parse_file_enhanced(
    data_path: str,
    poi_path: str,
    ds_factor: int = DS_FACTOR
) -> tuple[np.ndarray, float]:
    """Enhanced parsing with better feature extraction"""
    df = pd.read_csv(data_path)[
        ['Relative_time', 'Dissipation', 'Resonance_Frequency']]
    # df = enhanced_preprocess_df(df)
    df = enhanced_preprocess_df(df.copy())
    # fig, axes = plot_run_features(
    #     df,
    #     exclude=("std_", "ma_", "dev_"),  # example filter
    #     standardize=False,
    #     cols_per_row=4
    # )
    # plt.show()
    pois = pd.read_csv(poi_path, header=None).values.squeeze()
    if len(pois) < 6:
        raise ValueError(f"Expected ≥6 POIs, got {len(pois)}")
    poi_raw = int(pois[5])

    # Use all engineered features
    feature_cols = [col for col in df.columns if col != 'index']
    X = df[feature_cols].values.astype(np.float32)
    X_ds = X[::ds_factor]

    idx_ds = poi_raw // ds_factor
    if idx_ds < 0 or idx_ds >= len(X_ds):
        raise IndexError(f"Downsampled POI index {idx_ds} out of range")
    stop_time = float(X_ds[idx_ds, 0])

    return X_ds, stop_time


def create_advanced_dataset(runs, batch_size=64, window_size=WINDOW_SIZE, is_training=True):
    """Advanced dataset with better target engineering"""
    n_features = runs[0][0].shape[1]

    def gen():
        for X_ds, stop_time in runs:
            times_abs = X_ds[:, 0].astype(np.float32)
            T = len(times_abs)

            for t in range(window_size, T):
                # ----- window -----
                win = X_ds[t-window_size:t].copy()
                cur_t = times_abs[t-1]

                # Relative time encoding
                rel_times = times_abs[t-window_size:t] - cur_t
                win[:, 0] = rel_times

                # ----- dt (length = window_size) -----
                seg_times = times_abs[t-window_size:t]
                # First delta repeats the first real delta (edge padding)
                d = np.diff(seg_times).astype(np.float32)
                if d.size == 0:  # extremely small edge-case
                    d = np.array([0.0], dtype=np.float32)
                dt = np.pad(d, (1, 0), mode="edge")  # now len == window_size
                # Safety:
                if dt.shape[0] != window_size:
                    dt = dt[:window_size]

                # ----- targets -----
                time_to_stop = max(stop_time - cur_t, 0.0)
                near_stop = float(time_to_stop <= NEAR_THRESHOLD)

                log_time = np.log1p(time_to_stop)
                sqrt_time = np.sqrt(time_to_stop)
                progress = min(cur_t / (stop_time + 1e-8), 1.0)

                offsets_back = -rel_times  # == cur_t - seg_times
                alive_seq = (offsets_back < time_to_stop).astype(np.float32)

                # Final sanity (catch early)
                assert win.shape == (window_size, n_features), win.shape
                assert dt.shape == (window_size,), dt.shape
                assert alive_seq.shape == (window_size,), alive_seq.shape

                yield (
                    {"window": win.astype(np.float32), "dt": dt},
                    {
                        "time_left": np.float32(time_to_stop),
                        "log_time": np.float32(log_time),
                        "sqrt_time": np.float32(sqrt_time),
                        "near_stop": np.float32(near_stop),
                        "progress": np.float32(progress),
                        "alive_seq": alive_seq.astype(np.float32),
                    },
                )

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                "window": tf.TensorSpec((window_size, n_features), tf.float32),
                "dt": tf.TensorSpec((window_size,), tf.float32),
            },
            {
                "time_left": tf.TensorSpec((), tf.float32),
                "log_time": tf.TensorSpec((), tf.float32),
                "sqrt_time": tf.TensorSpec((), tf.float32),
                "near_stop": tf.TensorSpec((), tf.float32),
                "progress": tf.TensorSpec((), tf.float32),
                "alive_seq": tf.TensorSpec((window_size,), tf.float32),
            },
        ),
    )

    if is_training:
        ds = ds.shuffle(20_000, reshuffle_each_iteration=True)

    # drop_remainder helps if batch size doesn't divide evenly
    return ds.batch(batch_size, drop_remainder=is_training).prefetch(tf.data.AUTOTUNE)


def build_advanced_model(window_size=WINDOW_SIZE, n_features=None, dropout=DROPOUT_RATE, l2=L2_REG):
    """Advanced architecture with attention and residual connections"""
    inp = layers.Input((window_size, n_features), name="window")
    dt = layers.Input((window_size,), name="dt")

    # Input normalization
    x = layers.LayerNormalization()(inp)

    # Multi-scale temporal feature extraction
    features = []

    # Branch 1: Fine-grained temporal patterns
    x1 = x
    for i, (filters, kernel) in enumerate([(64, 3), (96, 5), (128, 7)]):
        x1 = layers.Conv1D(
            filters, kernel, padding='same',
            kernel_regularizer=regularizers.l2(l2),
            name=f"fine_conv_{i}"
        )(x1)
        x1 = layers.LayerNormalization()(x1)
        x1 = layers.Activation('swish')(x1)
        x1 = layers.SpatialDropout1D(dropout * 0.5)(x1)
    features.append(x1)

    # Branch 2: Medium-scale patterns with dilation
    x2 = x
    for i, (filters, dilation) in enumerate([(64, 2), (96, 4), (128, 8)]):
        x2 = layers.Conv1D(
            filters, 3, padding='same', dilation_rate=dilation,
            kernel_regularizer=regularizers.l2(l2),
            name=f"medium_conv_{i}"
        )(x2)
        x2 = layers.LayerNormalization()(x2)
        x2 = layers.Activation('swish')(x2)
        x2 = layers.SpatialDropout1D(dropout * 0.5)(x2)
    features.append(x2)

    # Branch 3: Long-range dependencies with large kernels
    x3 = x
    for i, (filters, kernel) in enumerate([(64, 15), (96, 31), (128, 63)]):
        x3 = layers.Conv1D(
            filters, kernel, padding='same',
            kernel_regularizer=regularizers.l2(l2),
            name=f"long_conv_{i}"
        )(x3)
        x3 = layers.LayerNormalization()(x3)
        x3 = layers.Activation('swish')(x3)
        x3 = layers.SpatialDropout1D(dropout * 0.5)(x3)
    features.append(x3)

    # Combine multi-scale features
    x = layers.Concatenate()(features)
    x = layers.Conv1D(256, 1, padding='same', name="feature_fusion")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('swish')(x)

    # Self-attention for long-range dependencies
    attention = layers.MultiHeadAttention(
        num_heads=8, key_dim=32, dropout=dropout, name="self_attention"
    )(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)

    # Final temporal processing
    x = layers.Conv1D(128, 3, padding='same', name="final_conv")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.SpatialDropout1D(dropout)(x)

    # Global feature extraction for scalar predictions
    g_avg = layers.GlobalAveragePooling1D()(x)
    g_max = layers.GlobalMaxPooling1D()(x)
    g = layers.Concatenate()([g_avg, g_max])
    g = layers.Dense(256, activation="swish", name="global_dense1")(g)
    g = layers.Dropout(dropout)(g)
    g = layers.Dense(128, activation="swish", name="global_dense2")(g)
    g = layers.Dropout(dropout)(g)

    # Multiple prediction heads with different loss functions

    # Primary time prediction (seconds)
    time_left = layers.Dense(64, activation="swish")(g)
    time_left = layers.Dense(1, activation="softplus",
                             name="time_left")(time_left)

    # Log-time prediction for better numerical stability
    log_time = layers.Dense(64, activation="swish")(g)
    log_time = layers.Dense(1, activation="linear", name="log_time")(log_time)

    # Square-root time for medium-range predictions
    sqrt_time = layers.Dense(64, activation="swish")(g)
    sqrt_time = layers.Dense(1, activation="softplus",
                             name="sqrt_time")(sqrt_time)

    # Binary classifier for near-stop detection
    near_stop = layers.Dense(32, activation="swish")(g)
    near_stop = layers.Dense(1, activation="sigmoid",
                             name="near_stop")(near_stop)

    # Progress indicator
    progress = layers.Dense(32, activation="swish")(g)
    progress = layers.Dense(1, activation="sigmoid", name="progress")(progress)

    # Per-timestep survival prediction
    survival_logits = layers.Conv1D(
        1, 1, padding="same", name="survival_logits")(x)
    alive_seq = layers.Activation("sigmoid", name="alive_seq")(survival_logits)
    alive_seq = layers.Reshape((window_size,))(alive_seq)

    model = models.Model(
        inputs={"window": inp, "dt": dt},
        outputs={
            "time_left": time_left,
            "log_time": log_time,
            "sqrt_time": sqrt_time,
            "near_stop": near_stop,
            "progress": progress,
            "alive_seq": alive_seq
        }
    )

    # Advanced optimizer with learning rate scheduling
    opt = opt = tfa.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=1e-5)

    model.compile(
        optimizer=opt,
        loss={
            # Robust to outliers
            "time_left": tf.keras.losses.Huber(delta=2.0),
            "log_time": tf.keras.losses.MeanSquaredError(),
            "sqrt_time": tf.keras.losses.MeanAbsoluteError(),
            "near_stop": tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            "progress": tf.keras.losses.MeanSquaredError(),
            "alive_seq": tf.keras.losses.BinaryCrossentropy()
        },
        loss_weights={
            "time_left": 1.0,      # Primary target
            "log_time": 0.5,       # Numerical stability
            "sqrt_time": 0.3,      # Medium-range accuracy
            "near_stop": 0.2,      # Classification aid
            "progress": 0.1,       # Auxiliary
            "alive_seq": 0.1       # Sequence learning
        },
        metrics={
            "time_left": [
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse")
            ],
            "log_time": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "near_stop": [tf.keras.metrics.BinaryAccuracy(name="acc")],
            "progress": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "alive_seq": [tf.keras.metrics.BinaryAccuracy(name="acc")]
        },
        jit_compile=False,
    )

    return model


def train_advanced_model():
    """Training with comprehensive improvements"""
    print("Loading training data...")
    paths = load_all_runs("content/live/train",
                          max_runs=100)  # Reasonable limit
    print(f"Found {len(paths)} runs")

    raw_runs = []
    for i, (d, p) in enumerate(paths):
        print(i, d, p)
        try:
            run_data = parse_file_enhanced(d, p)
            raw_runs.append(run_data)
            if (i + 1) % 100 == 0:
                print(f"Loaded {i + 1}/{len(paths)} runs")
        except Exception as e:
            print(f"Skipping run {i}: {e}")
            continue

    print(f"Successfully loaded {len(raw_runs)} runs")

    if len(raw_runs) < 10:
        raise ValueError("Too few valid runs for training")

    # Better train/validation split
    np.random.shuffle(raw_runs)
    split = int(0.8 * len(raw_runs))
    train_raw = raw_runs[:split]
    val_raw = raw_runs[split:]

    # Robust scaling
    print("Fitting scaler...")
    all_train = np.vstack([X[:, 1:]
                          for X, _ in train_raw])   # drop time column
    scaler = RobustScaler().fit(all_train)

    train_runs = [
        (np.concatenate([X[:, :1], scaler.transform(X[:, 1:])], axis=1), stop)
        for X, stop in train_raw
    ]
    val_runs = [
        (np.concatenate([X[:, :1], scaler.transform(X[:, 1:])], axis=1), stop)
        for X, stop in val_raw
    ]
    n_features = train_runs[0][0].shape[1]
    print(f"Using {n_features} features")
    print(f"Train: {len(train_runs)}, Val: {len(val_runs)}")

    # Create datasets
    ds_train = create_advanced_dataset(
        train_runs, batch_size=BATCH_SIZE, is_training=True)
    ds_val = create_advanced_dataset(
        val_runs, batch_size=BATCH_SIZE, is_training=False)

    # Build model
    model = build_advanced_model(WINDOW_SIZE, n_features)
    print(f"Model parameters: {model.count_params():,}")

    # Advanced callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_time_left_mae",
            patience=PATIENCE,
            restore_best_weights=True,
            min_delta=0.01,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_time_left_mae",
            factor=0.3,
            patience=PATIENCE//3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_advanced_model.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        callbacks.CSVLogger('advanced_training_log.csv'),
        # Cosine annealing for learning rate
        callbacks.LearningRateScheduler(
            lambda epoch: LEARNING_RATE * 0.5 *
            (1 + np.cos(np.pi * epoch / EPOCHS)),
            verbose=0
        )
    ]

    print("Starting training...")
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )

    # Final evaluation
    print("\nEvaluating model...")
    train_eval = model.evaluate(ds_train, verbose=0, return_dict=True)
    val_eval = model.evaluate(ds_val, verbose=0, return_dict=True)

    print("\n" + "="*50)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Train Time MAE: {train_eval['time_left_mae']:.3f}s")
    print(f"Val Time MAE:   {val_eval['time_left_mae']:.3f}s")
    print(f"Train Time RMSE: {train_eval['time_left_rmse']:.3f}s")
    print(f"Val Time RMSE:   {val_eval['time_left_rmse']:.3f}s")
    print(f"Val Near-Stop Acc: {val_eval['near_stop_acc']:.3f}")
    print(
        f"Generalization Gap: {val_eval['time_left_mae']/train_eval['time_left_mae']:.2f}x")

    if val_eval['time_left_mae'] > 10.0:
        print("\n⚠️  WARNING: High validation error suggests poor model performance")
        print("   Consider: more data, different architecture, or feature engineering")
    elif val_eval['time_left_mae'] < 2.0:
        print("\n✅ GOOD: Model achieving reasonable accuracy")

    return model, scaler, history


# ... keep all your existing imports/code above ...


def simulate_advanced_live(
    data_path: str,
    model: tf.keras.Model,
    scaler,
    window_size: int = WINDOW_SIZE,
    ds_factor: int = DS_FACTOR,
    pause: float = 0.001,
    update_every: int = 10
):
    """Enhanced live simulation with better visualization"""
    print(f"Running simulation on: {data_path}")

    # Load & process raw
    X_ds_raw, stop_time = parse_file_enhanced(
        data_path,
        data_path.replace('.csv', '_poi.csv'),
        ds_factor
    )

    # Scale features (keep col0 = time unscaled)
    feats_scaled = scaler.transform(X_ds_raw[:, 1:])
    X_ds = np.concatenate([X_ds_raw[:, :1], feats_scaled],
                          axis=1).astype(np.float32)

    raw_times = X_ds_raw[:, 0].astype(np.float32)
    diss_vals = X_ds_raw[:, 1].astype(np.float32)

    # Buffers
    buf_scaled, buf_times = [], []
    times = []
    preds_sec, pred_times, errors = [], [], []
    near_probs, progress_vals = [], []

    # Live plotting
    plt.ion()
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1])

    ax_main = fig.add_subplot(gs[0, :])
    ax_pred = fig.add_subplot(gs[1, 0])
    ax_error = fig.add_subplot(gs[1, 1])
    ax_dist = fig.add_subplot(gs[1, 2])
    ax_prob = fig.add_subplot(gs[2, 0])
    ax_progress = fig.add_subplot(gs[2, 1])
    ax_metrics = fig.add_subplot(gs[2, 2])

    print("Starting live simulation...")
    for i in range(len(X_ds)):
        cur_t = raw_times[i]
        times.append(cur_t)

        buf_scaled.append(X_ds[i])
        buf_times.append(cur_t)
        if len(buf_scaled) > window_size:
            buf_scaled.pop(0)
            buf_times.pop(0)

        if len(buf_scaled) == window_size:
            win = np.array(buf_scaled, dtype=np.float32)
            rel_times = np.array(buf_times, dtype=np.float32) - buf_times[-1]
            win[:, 0] = rel_times  # keep time col as relative

            t_arr = np.array(buf_times, dtype=np.float32)
            dt = np.diff(t_arr, prepend=t_arr[0])

            outs = model.predict(
                {"window": win[None, ...], "dt": dt[None, ...]}, verbose=0)
            # Handle dict/list returns

            def get(o, k):  # safe extractor
                if isinstance(o, dict):
                    return o[k]
                # assume identical order as model.outputs
                name2idx = {t.name.split(
                    '/')[0]: j for j, t in enumerate(model.outputs)}
                return o[name2idx[k]]

            pred_sec = max(float(get(outs, "time_left")[0, 0]), 0.0)
            near_prob = float(get(outs, "near_stop")[0, 0])
            progress_val = float(get(outs, "progress")[0, 0])

            pred_stop_time = cur_t + pred_sec
            error = abs(pred_stop_time - stop_time)

            preds_sec.append(pred_sec)
            pred_times.append(cur_t)
            near_probs.append(near_prob)
            progress_vals.append(progress_val)
            errors.append(error)

        # Update plots
        if i % update_every == 0 and preds_sec:
            # Main signal
            ax_main.clear()
            ax_main.plot(times, diss_vals[:len(
                times)], 'b-', alpha=0.7, label='Dissipation', linewidth=1)
            ax_main.axvline(stop_time, color='red', ls='--',
                            lw=3, label='True Stop', alpha=0.8)
            ax_main.axvline(pred_times[-1] + preds_sec[-1], color='green', ls='--', lw=3,
                            label=f'Pred Stop ({preds_sec[-1]:.1f}s)', alpha=0.8)
            ax_main.set_xlabel('Time (s)')
            ax_main.set_ylabel('Dissipation')
            ax_main.set_title(
                'Live Stop Position Prediction', fontweight='bold')
            ax_main.legend(fontsize=9)
            ax_main.grid(True, alpha=0.3)

            # Time-to-stop evolution
            ax_pred.clear()
            ax_pred.plot(pred_times, preds_sec, lw=2)
            ax_pred.set_xlabel('Current Time (s)')
            ax_pred.set_ylabel('Predicted Time Left (s)')
            ax_pred.set_title('Prediction Trajectory')
            ax_pred.grid(True, alpha=0.3)

            # Absolute error over time
            ax_error.clear()
            ax_error.plot(pred_times, errors, lw=2)
            ax_error.set_xlabel('Current Time (s)')
            ax_error.set_ylabel('|Pred Stop - True Stop| (s)')
            ax_error.set_title('Absolute Error')
            ax_error.grid(True, alpha=0.3)

            # Distribution of predictions
            ax_dist.clear()
            ax_dist.hist(preds_sec, bins=30)
            ax_dist.set_xlabel('Predicted Time Left (s)')
            ax_dist.set_ylabel('Count')
            ax_dist.set_title('Prediction Distribution')

            # Near-stop probability
            ax_prob.clear()
            ax_prob.plot(pred_times, near_probs, lw=2)
            ax_prob.set_ylim(0, 1)
            ax_prob.set_xlabel('Current Time (s)')
            ax_prob.set_ylabel('P(Near Stop)')
            ax_prob.set_title('Near-Stop Probability')
            ax_prob.grid(True, alpha=0.3)

            # Progress value (0→1)
            ax_progress.clear()
            ax_progress.plot(pred_times, progress_vals, lw=2)
            ax_progress.set_ylim(0, 1)
            ax_progress.set_xlabel('Current Time (s)')
            ax_progress.set_ylabel('Progress')
            ax_progress.set_title('Estimated Progress')
            ax_progress.grid(True, alpha=0.3)

            # Metrics box
            ax_metrics.clear()
            ax_metrics.axis('off')
            txt = (
                f"Current t: {cur_t:.1f}s\n"
                f"Pred time left: {preds_sec[-1]:.2f}s\n"
                f"Pred stop: {pred_times[-1] + preds_sec[-1]:.2f}s\n"
                f"True stop: {stop_time:.2f}s\n"
                f"Abs error: {errors[-1]:.2f}s\n"
                f"P(near stop): {near_probs[-1]:.3f}\n"
                f"Progress: {progress_vals[-1]:.3f}"
            )
            ax_metrics.text(0.05, 0.95, txt, va='top', ha='left', fontsize=11,
                            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))

            fig.tight_layout()
            plt.pause(pause)

    print("Simulation finished. Close the plot window to end.")
    plt.ioff()
    plt.show(block=True)

    return {
        "times": np.array(times),
        "pred_times": np.array(pred_times),
        "preds_sec": np.array(preds_sec),
        "errors": np.array(errors),
        "near_probs": np.array(near_probs),
        "progress": np.array(progress_vals),
        "true_stop_time": stop_time
    }


def save_scaler(scaler, path: str | Path):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(path: str | Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train and/or simulate advanced stop-time model")
    parser.add_argument("--train_dir", type=str, default=os.path.join("content", "live", "train"),
                        help="Directory with training CSVs (+ _poi.csv files)")
    parser.add_argument("--max_runs", type=int, default=2000,
                        help="Max runs to load for training")
    parser.add_argument("--model_out", type=str, default="best_advanced_model.keras",
                        help="Where to save the trained model")
    parser.add_argument("--scaler_out", type=str, default="scaler.pkl",
                        help="Where to save the fitted scaler")
    parser.add_argument("--simulate", type=str, default=os.path.join("content", "static", "valid", "02487", "MM240625Y3_IGG50_2_3rd.csv"),
                        help="Path to a single CSV to simulate on")
    parser.add_argument("--model_in", type=str, default="best_advanced_model.keras",
                        help="Path to a saved model to load for simulation")
    parser.add_argument("--scaler_in", type=str, default="scaler.pkl",
                        help="Path to a saved scaler to load for simulation")
    parser.add_argument("--no_gui", action="store_true",
                        help="Run simulation without interactive plotting")
    parser.add_argument("--pause", type=float, default=0.001,
                        help="Pause interval for plt.pause during live sim")
    parser.add_argument("--update_every", type=int, default=10,
                        help="Plot update interval (steps)")
    args = parser.parse_args()
    if TRAINING:
        # Training
        global EPOCHS  # allow override if desired
        model, scaler, history = train_advanced_model()
        model.save(args.model_out)
        save_scaler(scaler, args.scaler_out)
        print(f"\nSaved model to {args.model_out}")
        print(f"Saved scaler to {args.scaler_out}")

    # Simulation
    if args.model_in is None or args.scaler_in is None:
        print("ERROR: --simulate requires --model_in and --scaler_in")
        sys.exit(1)

    model = tf.keras.models.load_model(args.model_in, compile=False)
    scaler = load_scaler(args.scaler_in)
    print("Model & scaler loaded.")

    if args.no_gui:
        # Disable interactive mode to avoid hanging
        plt.ioff()

    _ = simulate_advanced_live(
        data_path=args.simulate,
        model=model,
        scaler=scaler,
        window_size=WINDOW_SIZE,
        ds_factor=DS_FACTOR,
        pause=args.pause,
        update_every=args.update_every
    )


if __name__ == "__main__":
    main()
