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
import keras_tuner as kt
from sklearn.model_selection import KFold

# ——— Improved Hyperparameters ———
EPOCHS = 1  # More epochs with better regularization
DS_FACTOR = 5
WINDOW_SIZE = 64  # Reduced for better generalization
NEAR_THRESHOLD = 2.0  # Increased threshold for more balanced classification
BATCH_SIZE = 32  # Smaller batches for better gradient estimates
LEARNING_RATE = 1e-4  # More conservative
DROPOUT_RATE = 0.3  # Balanced regularization
L2_REG = 1e-5  # Lighter regularization

print("Py:", sys.version)
print("TF:", tf.__version__)
print("Built w/ CUDA:", tf.test.is_built_with_cuda())
print("Physical GPUs:", tf.config.list_physical_devices('GPU'))


class HazardToTime(tf.keras.layers.Layer):
    def call(self, inputs):
        # inputs: (B, W) hazards; dt: (B, W)
        hazards, dt = inputs
        hazards = tf.clip_by_value(hazards, 1e-6, 1-1e-6)
        # survival S_{k-1}
        log_surv = tf.math.cumsum(
            tf.math.log1p(-hazards), axis=1, exclusive=True)
        surv = tf.exp(log_surv)  # (B, W)
        exp_time = tf.reduce_sum(surv * dt, axis=1, keepdims=True)
        return exp_time  # (B,1)


# Set mixed precision
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")
except:
    print("Mixed precision not available, using float32")


def load_all_runs(data_dir: str,
                  max_runs: int = np.inf,
                  bin_sec: float = 10.0,
                  per_bin: int | None = None,
                  mode: str = "undersample",      # or "oversample"
                  assign_strategy: str = "span"   # "span" | "median"
                  ):
    """
    Time-balanced file selector. Returns [(data_csv, poi_csv), ...].

    assign_strategy:
        - "span": put a run in every 10s bin it touches
        - "median": put a run only in the bin of its median Relative_time
    """
    rng = random.Random(42)

    # 1) collect all candidate pairs
    candidates = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.csv') and not f.endswith(('_poi.csv', '_lower.csv')):
                p_csv = os.path.join(root, f.replace('.csv', '_poi.csv'))
                if os.path.exists(p_csv):
                    candidates.append((os.path.join(root, f), p_csv))

    # trivial case
    if max_runs != np.inf and len(candidates) <= max_runs:
        rng.shuffle(candidates)
        return candidates[:int(max_runs)]

    # 2) bin runs
    bins: dict[int, list[tuple[str, str]]] = {}
    for d_csv, p_csv in candidates:
        try:
            df = pd.read_csv(d_csv, usecols=['Relative_time'])
            rel = df['Relative_time'].to_numpy()
            rel = rel - rel[0]  # normalize start
            if assign_strategy == "median":
                b = int(np.median(rel) // bin_sec)
                bins.setdefault(b, []).append((d_csv, p_csv))
            else:  # "span"
                b0 = int(rel.min() // bin_sec)
                b1 = int(rel.max() // bin_sec)
                for b in range(b0, b1 + 1):
                    bins.setdefault(b, []).append((d_csv, p_csv))
        except Exception as e:
            print(f"Skipping {d_csv}: {e}")
            continue

    # if no bins built, just fallback
    if not bins:
        rng.shuffle(candidates)
        return candidates if max_runs == np.inf else candidates[:int(max_runs)]

    # 3) choose target per_bin if not provided
    sizes = [len(v) for v in bins.values() if v]
    if not sizes:  # all empty somehow
        rng.shuffle(candidates)
        return candidates if max_runs == np.inf else candidates[:int(max_runs)]

    if per_bin is None:
        per_bin = int(np.median(sizes))  # tweak to taste: min/max/quantile

    # 4) rebalance
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
        else:
            raise ValueError("mode must be 'undersample' or 'oversample'")

    # 5) dedupe & cap max_runs
    rng.shuffle(balanced)
    seen, uniq = set(), []
    for pair in balanced:
        if pair not in seen:
            seen.add(pair)
            uniq.append(pair)
            if len(uniq) >= (int(max_runs) if max_runs != np.inf else 10**12):
                break

    return uniq


def conservative_preprocess_df(
    df: pd.DataFrame,
    clip_percentile: Tuple[float, float] = (1, 99),
    smooth_window: int = 5
) -> pd.DataFrame:
    """More conservative preprocessing to prevent overfitting"""
    df = df.sort_values('Relative_time').reset_index(drop=True)
    df['Relative_time'] -= df['Relative_time'].iloc[0]
    df = df.interpolate(method='linear', limit_direction='both')

    # Gentler clipping
    for c in ('Dissipation', 'Resonance_Frequency'):
        lo, hi = np.percentile(df[c], clip_percentile)
        df[c] = df[c].clip(lo, hi)

    # Light smoothing
    for c in ('Dissipation', 'Resonance_Frequency'):
        df[c] = df[c].rolling(smooth_window, center=True, min_periods=1).mean()

    # Essential features only to prevent overfitting
    df['Dissipation_diff'] = df['Dissipation'].diff().fillna(0)
    df['Resonance_Frequency_diff'] = df['Resonance_Frequency'].diff().fillna(0)

    # Only short-term statistics
    window = 10
    df[f'Dissipation_std'] = df['Dissipation'].rolling(
        window, min_periods=1).std().fillna(0)
    df[f'Resonance_Frequency_std'] = df['Resonance_Frequency'].rolling(
        window, min_periods=1).std().fillna(0)

    # Simple derived features
    df['Ratio'] = df['Dissipation'] / (df['Resonance_Frequency'] + 1e-8)

    return df


def simple_tcn_block(x, filters, kernel_size=3, dilation_rate=1, name_prefix=""):
    """Simplified TCN block to prevent overfitting"""
    # Single convolution
    conv = layers.Conv1D(
        filters, kernel_size,
        padding='causal',
        dilation_rate=dilation_rate,
        kernel_regularizer=regularizers.l2(L2_REG),
        name=f"{name_prefix}_conv"
    )(x)

    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)
    conv = layers.SpatialDropout1D(DROPOUT_RATE)(conv)

    # Residual connection
    if x.shape[-1] != filters:
        res = layers.Conv1D(filters, 1, padding='same',
                            name=f"{name_prefix}_res")(x)
    else:
        res = x

    out = layers.Add()([conv, res])
    return out


def parse_file_conservative(
    data_path: str,
    poi_path: str,
    ds_factor: int = DS_FACTOR
) -> tuple[np.ndarray, float]:
    """Conservative parsing with minimal feature engineering"""
    df = pd.read_csv(data_path)[
        ['Relative_time', 'Dissipation', 'Resonance_Frequency']]
    df = conservative_preprocess_df(df)

    pois = pd.read_csv(poi_path, header=None).values.squeeze()
    if len(pois) < 6:
        raise ValueError(f"Expected ≥6 POIs, got {len(pois)}")
    poi_raw = int(pois[5])

    # Select only essential features
    feature_cols = [col for col in df.columns if col != 'index']
    X = df[feature_cols].values.astype(np.float32)
    X_ds = X[::ds_factor]

    idx_ds = poi_raw // ds_factor
    if idx_ds < 0 or idx_ds >= len(X_ds):
        raise IndexError(f"Downsampled POI index {idx_ds} out of range")
    stop_time = float(X_ds[idx_ds, 0])

    return X_ds, stop_time


def balanced_dataset(runs, batch_size=32, window_size=WINDOW_SIZE, is_training=True):
    n_features = runs[0][0].shape[1]

    def gen():
        for X_ds, stop_time in runs:
            T = X_ds.shape[0]
            for t in range(window_size, T):
                win = X_ds[t-window_size:t].copy()
                current_time = win[-1, 0]
                # time column -> seconds from now (negative inside window)
                win[:, 0] -= current_time

                time_to_stop = max(stop_time - current_time, 0.0)
                near_stop = float(time_to_stop <= NEAR_THRESHOLD)

                yield (
                    {"window": win,
                     "dt": np.full((window_size,), DS_FACTOR, np.float32)},
                    {
                        "near_stop": near_stop,
                        "log_time": np.log1p(time_to_stop),
                        "time_left": time_to_stop,
                    }
                )

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                "window": tf.TensorSpec((window_size, n_features), tf.float32),
                "dt": tf.TensorSpec((window_size,), tf.float32)
            },
            {
                "near_stop": tf.TensorSpec((), tf.float32),
                "log_time": tf.TensorSpec((), tf.float32),
                "time_left": tf.TensorSpec((), tf.float32),
            }
        )
    )
    if is_training:
        ds = ds.shuffle(10_000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_time_to_event_model(window_size=WINDOW_SIZE, n_features=None):
    inp = layers.Input((window_size, n_features), name="window")
    dt = layers.Input((window_size,), name="dt")

    x = layers.Conv1D(64, 3, padding='causal', activation='relu',
                      kernel_regularizer=regularizers.l2(L2_REG))(inp)
    x = layers.BatchNormalization()(x)

    for i, (rate, f) in enumerate(zip([1, 2, 4, 8], [64, 96, 96, 64])):
        x = simple_tcn_block(x, f, dilation_rate=rate, name_prefix=f"tcn_{i}")

    # Per-timestep hazard head
    h = layers.Conv1D(1, 1, activation='sigmoid', name='hazard_raw')(x)
    h = layers.Reshape((window_size,), name='hazard')(h)

    # Expected time left from hazards
    time_left = HazardToTime(name='hazard_to_time')([h, dt])

    # Scalar heads
    log_time = layers.Lambda(lambda z: tf.math.log1p(z),
                             name='log_time')(time_left)

    near_stop = layers.Dense(1, activation='sigmoid', name='near_stop')(
        layers.Dense(32, activation='relu')(layers.GlobalAveragePooling1D()(x))
    )

    model = models.Model(inputs={"window": inp, "dt": dt},
                         outputs={"near_stop": near_stop,
                                  "log_time": log_time,
                                  "time_left": time_left})

    # Custom monotonic penalty
    def monotonic_penalty(y_true, y_pred):
        # y_pred shape (B,1) here -> penalty 0. (We’ll add it on hazard_to_time output.)
        return 0.0

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=0.5)

    model.compile(
        optimizer=opt,
        loss={
            "near_stop": "binary_crossentropy",
            "log_time": "mae",
            "time_left": "huber"
        },
        loss_weights={
            "near_stop": 0.2,
            "log_time": 1.0,
            "time_left": 0.2
        },
        metrics={
            "near_stop": ["accuracy"],
            "log_time": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "time_left": [tf.keras.metrics.MeanAbsoluteError(name="mae")]
        }
    )
    return model


def train_improved_model():
    """Training with better validation and regularization"""
    # Load data with limits to prevent overfitting
    # Reduced dataset size
    paths = load_all_runs("content/live/train", max_runs=np.inf)
    print(f"Loading {len(paths)} runs")

    raw_runs = []
    for i, (d, p) in enumerate(paths):
        try:
            run_data = parse_file_conservative(d, p)
            raw_runs.append(run_data)
        except Exception as e:
            print(f"Skipping run {i}: {e}")
            continue

    print(f"Successfully loaded {len(raw_runs)} runs")

    # Better train/val split with shuffling
    np.random.shuffle(raw_runs)
    split = int(0.85 * len(raw_runs))  # Larger training set
    train_raw = raw_runs[:split]
    val_raw = raw_runs[split:]

    # Conservative scaling
    all_train = np.vstack([X for X, _ in train_raw])
    scaler = StandardScaler().fit(all_train)  # StandardScaler often works better

    train_runs = [(scaler.transform(X), stop) for X, stop in train_raw]
    val_runs = [(scaler.transform(X), stop) for X, stop in val_raw]

    n_features = train_runs[0][0].shape[1]
    print(f"Using {n_features} features")
    print(f"Train: {len(train_runs)}, Val: {len(val_runs)}")

    # Create datasets
    ds_train = balanced_dataset(
        train_runs, batch_size=BATCH_SIZE, is_training=True)
    ds_val = balanced_dataset(
        val_runs, batch_size=BATCH_SIZE, is_training=False)

    # Build model
    model = build_time_to_event_model(WINDOW_SIZE, n_features)
    print(f"Model parameters: {model.count_params():,}")

    # Improved callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.01
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_normalized_time_mae',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'improved_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        callbacks.CSVLogger('improved_training_log.csv')
    ]

    # Train with validation monitoring
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )

    # Evaluate final performance
    train_eval = model.evaluate(ds_train, verbose=0)
    val_eval = model.evaluate(ds_val, verbose=0)

    print("\nFinal Performance:")
    print(f"Train Loss: {train_eval[0]:.4f}")
    print(f"Val Loss: {val_eval[0]:.4f}")
    print(f"Overfitting Ratio: {val_eval[0]/train_eval[0]:.2f}")

    return model, scaler, history


def simulate_improved_live(
    data_path: str,
    model: tf.keras.Model,
    scaler,
    window_size: int = WINDOW_SIZE,
    ds_factor: int = DS_FACTOR
):
    """Improved live simulation with better visualization"""
    X_ds_raw, stop_time = parse_file_conservative(
        data_path, data_path.replace('.csv', '_poi.csv'), ds_factor)
    X_ds = scaler.transform(X_ds_raw)

    buf, times, dissipation_values = [], [], []
    predictions, prediction_times = [], []
    errors = []

    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    for raw_row, scaled_row in zip(X_ds_raw, X_ds):
        current_time = float(raw_row[0])
        times.append(current_time)
        # Use raw dissipation for plotting
        dissipation_values.append(float(raw_row[1]))

        buf.append(scaled_row)
        if len(buf) > window_size:
            buf.pop(0)

        if len(buf) == window_size:
            win = np.array(buf, dtype=np.float32)[None, ...]
            raw_out = model.predict(win, verbose=0)

            # Extract predictions
            if isinstance(raw_out, dict):
                near_prob = float(raw_out['near_stop'][0, 0])
                norm_time = float(raw_out['normalized_time'][0, 0])
                raw_time = float(raw_out['time_delta'][0, 0])
            else:
                # Handle list output
                name_to_idx = {n: i for i, n in enumerate(model.output_names)}
                near_prob = float(raw_out[name_to_idx['near_stop']][0, 0])
                norm_time = float(
                    raw_out[name_to_idx['normalized_time']][0, 0])
                raw_time = float(raw_out[name_to_idx['time_delta']][0, 0])

            # Use the normalized prediction scaled back
            pred_time_to_stop = max(norm_time * 100.0, 0.0)  # Scale back
            predicted_stop_time = current_time + pred_time_to_stop

            predictions.append(pred_time_to_stop)
            prediction_times.append(current_time)

            # Calculate current error
            current_error = abs(predicted_stop_time - stop_time)
            errors.append(current_error)

        # Update plots
        ax1.clear()
        ax1.plot(times, dissipation_values, 'b-',
                 alpha=0.7, label='Dissipation')
        ax1.axvline(stop_time, color='red', linestyle='--',
                    linewidth=2, label='True Stop')
        if len(predictions) > 0:
            pred_stop = prediction_times[-1] + predictions[-1]
            ax1.axvline(pred_stop, color='green', linestyle='--',
                        linewidth=2, label='Predicted Stop')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Dissipation')
        ax1.set_title('Signal vs Predictions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.clear()
        if len(predictions) > 1:
            ax2.plot(prediction_times, predictions, 'g-',
                     linewidth=2, label='Time to Stop')
            ax2.set_xlabel('Current Time (s)')
            ax2.set_ylabel('Predicted Time to Stop (s)')
            ax2.set_title('Time-to-Stop Predictions')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        ax3.clear()
        if len(errors) > 1:
            ax3.plot(prediction_times, errors, 'r-', linewidth=2, alpha=0.7)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Prediction Error (s)')
            ax3.set_title('Prediction Error Over Time')
            ax3.grid(True, alpha=0.3)

            # Show running average
            if len(errors) > 10:
                running_avg = np.convolve(errors, np.ones(10)/10, mode='valid')
                ax3.plot(prediction_times[9:], running_avg, 'k--', linewidth=2,
                         label=f'10-pt avg: {running_avg[-1]:.1f}s')
                ax3.legend()

        ax4.clear()
        if len(errors) > 0:
            ax4.hist(errors, bins=20, alpha=0.7, color='orange')
            ax4.set_xlabel('Prediction Error (s)')
            ax4.set_ylabel('Frequency')
            ax4.set_title(
                f'Error Distribution\nMean: {np.mean(errors):.1f}s, Std: {np.std(errors):.1f}s')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.01)

    plt.ioff()

    # Final statistics
    if errors:
        print(f"\nFinal Performance:")
        print(f"Mean Error: {np.mean(errors):.2f}s")
        print(f"Std Error: {np.std(errors):.2f}s")
        print(f"Median Error: {np.median(errors):.2f}s")
        print(f"Max Error: {np.max(errors):.2f}s")

    plt.show()


if __name__ == "__main__":
    print("Training improved model...")
    model, scaler, history = train_improved_model()

    print("\nRunning simulation...")
    simulate_improved_live(
        'content/static/valid/02480/MM240625Y3_IGG10_2_3rd.csv',
        model, scaler
    )
