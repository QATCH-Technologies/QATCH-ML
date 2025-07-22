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


class LearnableScale(layers.Layer):
    def __init__(self, init=1.0, **kwargs):
        super().__init__(**kwargs)
        self.init = init

    def build(self, input_shape):
        self.tau = self.add_weight(
            name="tau",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.init),
            dtype=self.compute_dtype,   # honors mixed precision policy
            trainable=True,
        )

    def call(self, x):
        return x * self.tau


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


def balanced_dataset(runs,
                     batch_size=32,
                     window_size=WINDOW_SIZE,
                     is_training=True):
    """
    runs: list of (X_ds, stop_time) where col0 is absolute seconds.
    We:
      • never expose absolute time: col0 is shifted so last row == 0.
      • compute per-step dt from the window itself (no future info).
      • build alive_seq from true remaining seconds, not DS_FACTOR guesses.
    """
    n_features = runs[0][0].shape[1]

    def gen():
        for X_ds, stop_time in runs:
            times_abs = X_ds[:, 0].astype(np.float32)
            T = len(times_abs)

            for t in range(window_size, T):
                # Slice window [t-window_size, t)
                win = X_ds[t-window_size:t].copy()

                # Current (last) absolute time in this window
                cur_t = times_abs[t-1]

                # --- Feature time column: make it strictly relative to cur_t
                # so model can't key on absolute progress in the run.
                rel_times = times_abs[t-window_size:t] - \
                    cur_t  # <=0, last is 0
                win[:, 0] = rel_times

                # --- dt per step (needed by hazard integrator); use diffs inside window
                # length window_size, align with hazards (same length)
                dt = np.diff(times_abs[t-window_size:t+1]).astype(np.float32)
                # last value repeat to keep shape W (or pad head); simplest:
                dt = np.concatenate([dt, [dt[-1]]])  # (W,)

                # Targets
                time_to_stop = max(stop_time - cur_t, 0.0)
                near_stop = float(time_to_stop <= NEAR_THRESHOLD)

                # Alive sequence: for each position k in window (0=oldest),
                # are we still "alive" at that step relative to cur_t?
                # Offsets from each row to cur_t are -rel_times (positive back in time).
                # If time_to_stop > offset, we're alive.
                offsets_back = -rel_times  # >=0
                alive_seq = (offsets_back < time_to_stop).astype(np.float32)

                yield (
                    {"window": win.astype(np.float32), "dt": dt},
                    {
                        "near_stop": near_stop,
                        "log_time": np.log1p(time_to_stop),
                        "time_left": time_to_stop,
                        "alive_seq": alive_seq
                    }
                )

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                "window": tf.TensorSpec((window_size, n_features), tf.float32),
                "dt": tf.TensorSpec((window_size,), tf.float32),
            },
            {
                "near_stop": tf.TensorSpec((), tf.float32),
                "log_time": tf.TensorSpec((), tf.float32),
                "time_left": tf.TensorSpec((), tf.float32),
                "alive_seq": tf.TensorSpec((window_size,), tf.float32),
            }
        )
    )
    if is_training:
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_stopnet(window_size=WINDOW_SIZE, n_features=None,
                  dropout=DROPOUT_RATE, l2=L2_REG):
    inp = layers.Input((window_size, n_features), name="window")
    dt = layers.Input((window_size,), name="dt")

    # --- Feature stem (depthwise-separable causal convs) ---
    x = inp
    for i, dil in enumerate([1, 2, 4, 8, 16]):
        res = x
        x = layers.SeparableConv1D(
            filters=96 if i < 3 else 128,
            kernel_size=5,
            dilation_rate=dil,
            padding="causal",
            depthwise_regularizer=regularizers.l2(l2),
            pointwise_regularizer=regularizers.l2(l2),
            name=f"dws_conv_{i}"
        )(x)
        x = layers.BatchNormalization()(x)
        # Gated linear unit
        a, b = tf.split(x, 2, axis=-1)
        x = tf.nn.swish(a) * tf.sigmoid(b)
        x = layers.SpatialDropout1D(dropout)(x)
        # match channels for residual
        if res.shape[-1] != x.shape[-1]:
            res = layers.Conv1D(x.shape[-1], 1, padding="same")(res)
        x = layers.Add()([x, res])

    # Global summary for scalar heads
    g = layers.GlobalAveragePooling1D()(x)
    g = layers.Dense(128, activation="relu")(g)
    g = layers.Dropout(dropout)(g)

    # --------- Main scalar head: time_left (seconds) ----------
    # Positive output via softplus; learnable scale τ for stability
    tau = tf.Variable(1.0, dtype=tf.float32, name="time_scale")
    time_left_raw = layers.Dense(
        1, activation="softplus", name="time_left_raw")(g)
    time_left = LearnableScale(name="time_left")(time_left_raw)

    # log_time (just log1p of above to keep metrics readable)
    log_time = layers.Lambda(lambda z: tf.math.log1p(z),
                             name="log_time")(time_left)

    # --------- Auxiliary near_stop classifier ----------
    near_stop = layers.Dense(1, activation="sigmoid", name="near_stop")(g)

    # --------- Hazard / alive sequence head ----------
    # Per-timestep logits -> sigmoids -> cumulative hazard monotonic
    h_logits = layers.Conv1D(1, 1, padding="same", name="hazard_logits")(x)
    hazards = layers.Activation("sigmoid", name="hazard_prob")(h_logits)
    hazards = layers.Reshape((window_size,), name="hazards")(hazards)

    # Survival from hazards
    hazards_clip = layers.Lambda(
        lambda z: tf.clip_by_value(z, 1e-6, 1-1e-6))(hazards)
    log_surv = layers.Lambda(lambda z: tf.math.cumsum(
        tf.math.log1p(-z), axis=1, exclusive=True))(hazards_clip)
    surv = layers.Lambda(lambda z: tf.exp(z), name="survival")(log_surv)

    # Alive sequence label expects 1 while alive; our survival ≈ P(alive)
    alive_pred = layers.Lambda(lambda z: z, name="alive_seq")(surv)

    # Expected time from hazards as check (not backprop main loss)
    exp_time = layers.Lambda(lambda z: tf.reduce_sum(z[0]*z[1], axis=1, keepdims=True),
                             name="hazard_to_time")([surv, dt])

    model = models.Model(
        inputs={"window": inp, "dt": dt},
        outputs={
            "near_stop": near_stop,
            "log_time": log_time,
            "time_left": time_left,
            "alive_seq": alive_pred,
            "hazard_to_time": exp_time
        }
    )

    opt = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=opt,
        loss={
            "near_stop": tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02),
            "log_time": tf.keras.losses.MeanAbsoluteError(),
            "time_left": tf.keras.losses.Huber(delta=5.0),
            "alive_seq": tf.keras.losses.BinaryCrossentropy(),
            "hazard_to_time": tf.keras.losses.MeanAbsoluteError()
        },
        loss_weights={
            "near_stop": 0.1,
            "log_time": 1.0,       # main target
            "time_left": 0.1,      # raw-seconds reg (stabilizer)
            "alive_seq": 0.05,     # weak aux
            "hazard_to_time": 0.0  # monitor only
        },
        metrics={
            "near_stop": ["accuracy"],
            "log_time": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "time_left": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "alive_seq": [tf.keras.metrics.BinaryAccuracy(name="acc")],
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
    model = build_stopnet(WINDOW_SIZE, n_features)
    print(f"Model parameters: {model.count_params():,}")

    # Improved callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor="val_log_time_mae", patience=6,
                                restore_best_weights=True, min_delta=0.01),
        callbacks.ReduceLROnPlateau(monitor="val_log_time_mae", factor=0.5,
                                    patience=3, min_lr=1e-6),
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
    """
    Live simulation that mirrors your current pipeline:
      • log_time head = log1p(seconds)
      • time_left head = raw seconds
      • No external TIME_SCALE
      • Window time column re-zeroed, dt from inside the window
    """
    # ---- Load & scale like training ----
    X_ds_raw, stop_time = parse_file_conservative(
        data_path, data_path.replace('.csv', '_poi.csv'), ds_factor
    )

    # scale only feature cols (exclude absolute time)
    feats_scaled = scaler.transform(X_ds_raw[:, 1:])
    X_ds = np.concatenate([X_ds_raw[:, :1], feats_scaled],
                          axis=1).astype(np.float32)

    raw_times = X_ds_raw[:, 0].astype(np.float32)
    diss_vals = X_ds_raw[:, 1].astype(np.float32)

    buf_scaled, buf_times = [], []
    times = []
    preds_sec, pred_times, errors = [], [], []
    near_probs = []

    # ---- Plot setup ----
    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

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
            # re-zero time col
            rel_times = np.array(buf_times, dtype=np.float32) - buf_times[-1]
            win[:, 0] = rel_times

            # dt from within window (length = window_size)
            dt = np.diff(
                np.array(buf_times + [buf_times[-1]], dtype=np.float32))
            dt = np.concatenate([dt, [dt[-1]]]).astype(np.float32)

            outs = model.predict(
                {"window": win[None, ...], "dt": dt[None, ...]}, verbose=0)

            # ---- Extract predictions ----
            if isinstance(outs, dict):
                # Prefer raw seconds if present
                if "time_left" in outs:
                    pred_sec = float(outs["time_left"][0, 0])
                elif "log_time" in outs:
                    pred_sec = float(np.expm1(outs["log_time"][0, 0]))
                else:
                    raise KeyError("Model outputs lack time_left/log_time")
                near_prob = float(outs.get("near_stop", [[np.nan]])[0, 0])
            else:
                idx = {n: j for j, n in enumerate(model.output_names)}
                if "time_left" in idx:
                    pred_sec = float(outs[idx["time_left"]][0, 0])
                elif "log_time" in idx:
                    pred_sec = float(np.expm1(outs[idx["log_time"]][0, 0]))
                else:
                    raise KeyError("Model outputs lack time_left/log_time")
                near_prob = float(outs[idx.get("near_stop", 0)][0, 0])

            pred_sec = max(pred_sec, 0.0)
            pred_stop_time = cur_t + pred_sec

            preds_sec.append(pred_sec)
            pred_times.append(cur_t)
            near_probs.append(near_prob)
            errors.append(abs(pred_stop_time - stop_time))

        # ---- Plots ----
        ax1.clear()
        ax1.plot(times, diss_vals[:len(times)], 'b-',
                 alpha=0.7, label='Dissipation')
        ax1.axvline(stop_time, color='red', ls='--', lw=2, label='True Stop')
        if preds_sec:
            ax1.axvline(pred_times[-1] + preds_sec[-1],
                        color='green', ls='--', lw=2, label='Pred Stop')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Dissipation')
        ax1.set_title('Signal vs Predictions')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2.clear()
        if len(preds_sec) > 1:
            ax2.plot(pred_times, preds_sec, lw=2, label='Pred TTS (s)')
            ax2.set_xlabel('Current Time (s)')
            ax2.set_ylabel('Seconds to Stop')
            ax2.set_title('Time-to-Stop Predictions')
            ax2.grid(True, alpha=0.3)
            if near_probs and not np.isnan(near_probs[-1]):
                ax2b = ax2.twinx()
                ax2b.plot(pred_times, near_probs, ls='--',
                          alpha=0.4, label='Near Prob')
                ax2b.set_ylabel('Near-Stop Prob')
                ax2b.set_ylim(0, 1)

        ax3.clear()
        if len(errors) > 1:
            ax3.plot(pred_times, errors, 'r-', lw=2, alpha=0.7)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Abs Error (s)')
            ax3.set_title('Prediction Error Over Time')
            ax3.grid(True, alpha=0.3)
            if len(errors) > 10:
                ra = np.convolve(errors, np.ones(10)/10, mode='valid')
                ax3.plot(pred_times[9:], ra, 'k--', lw=2,
                         label=f'10-pt avg: {ra[-1]:.1f}s')
                ax3.legend()

        ax4.clear()
        if errors:
            ax4.hist(errors, bins=20, alpha=0.7)
            ax4.set_xlabel('Abs Error (s)')
            ax4.set_ylabel('Freq')
            ax4.set_title(
                f'Error Dist  μ={np.mean(errors):.1f}s  σ={np.std(errors):.1f}s  med={np.median(errors):.1f}s')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.01)

    plt.ioff()
    if errors:
        print("\nFinal Performance:")
        print(f"Mean Error:   {np.mean(errors):.2f}s")
        print(f"Std Error:    {np.std(errors):.2f}s")
        print(f"Median Error: {np.median(errors):.2f}s")
        print(f"Max Error:    {np.max(errors):.2f}s")
    plt.show()


if __name__ == "__main__":
    print("Training improved model...")
    model, scaler, history = train_improved_model()

    print("\nRunning simulation...")
    simulate_improved_live(
        'content/static/valid/02480/MM240625Y3_IGG10_2_3rd.csv',
        model, scaler
    )
