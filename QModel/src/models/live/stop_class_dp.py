from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.preprocessing import RobustScaler, StandardScaler
import tensorflow as tf
from tqdm import tqdm
import keras_tuner as kt

# ─── parameters & mappings ─────────────────────────────────────────────────────

WIN_SIZE = 32
DS_FACTOR = 5
TARGET_POIS = {6}
LABEL_MAP = {0: 0, 6: 1}

# ─── existing preprocessing funcs (unchanged) ───────────────────────────────────


def downsample_and_rebaseline(
    df: pd.DataFrame,
    factor: int = 5,
    window_size: int = WIN_SIZE,
    diss_col: str = "Dissipation",
    rf_col: str = "Resonance_Frequency"
) -> pd.DataFrame:
    df_ds = df.iloc[::factor].reset_index(drop=True)
    base_d = df_ds[diss_col].iloc[:window_size].mean()
    base_rf = df_ds[rf_col].iloc[:window_size].mean()
    df_ds[diss_col] = df_ds[diss_col] - base_d
    df_ds[rf_col] = -(df_ds[rf_col] - base_rf)
    return df_ds


def make_windows(df: pd.DataFrame, window_size: int = 50) -> List[pd.DataFrame]:
    windows = []
    for start in range(0, len(df) - window_size + 1, window_size):
        win = df.iloc[start:start + window_size].reset_index(drop=True)
        windows.append(win)
    return windows


def label_windows(
    num_windows: int,
    poi_indices: np.ndarray,
    factor: int,
    window_size: int,
    target_pois: set
) -> List[int]:
    poi_ds = poi_indices // factor
    labels = []
    for w in range(num_windows):
        start = w * window_size
        end = start + window_size
        hits = [
            i for i, ds in enumerate(poi_ds, start=1)
            if start <= ds < end and i in target_pois
        ]
        labels.append(hits[0] if hits else 0)
    return labels

# ─── feature extractor (unchanged) ──────────────────────────────────────────────


class FeatureExtractor:
    def __init__(self, window_size: int = 100, scaler_cls=StandardScaler):
        self.window_size = window_size
        self.scalers = {
            "Relative_time": scaler_cls(),
            "Dissipation": scaler_cls(),
            "Resonance_Frequency": scaler_cls(),
        }

    def fit_scalers(self, csv_paths: List[Path], downsample_factor: int = 5):
        all_t, all_d, all_rf = [], [], []
        for path in csv_paths:
            df = pd.read_csv(path)
            df_ds = downsample_and_rebaseline(
                df, factor=downsample_factor, window_size=self.window_size
            )
            all_t.append(df_ds[["Relative_time"]])
            all_d.append(df_ds[["Dissipation"]])
            all_rf.append(df_ds[["Resonance_Frequency"]])
        big_t = pd.concat(all_t, axis=0)
        big_d = pd.concat(all_d, axis=0)
        big_rf = pd.concat(all_rf, axis=0)
        self.scalers["Relative_time"].fit(big_t)
        self.scalers["Dissipation"].fit(big_d)
        self.scalers["Resonance_Frequency"].fit(big_rf)

    def transform_window(self, win: pd.DataFrame) -> np.ndarray:
        df = win.copy()
        for col in ["Relative_time", "Dissipation", "Resonance_Frequency"]:
            df[col] = self.scalers[col].transform(df[[col]]).ravel()

        df["dissipation_change"] = df["Dissipation"].diff().fillna(0)
        df["rf_change"] = df["Resonance_Frequency"].diff().fillna(0)
        df["diss_x_rf"] = df["Dissipation"] * df["Resonance_Frequency"]
        df["change_prod"] = df["dissipation_change"] * df["rf_change"]

        cols = [
            "Relative_time", "Dissipation", "Resonance_Frequency",
            "dissipation_change", "rf_change", "diss_x_rf", "change_prod"
        ]
        return df[cols].to_numpy()

# ─── dataset builder (unchanged) ────────────────────────────────────────────────


def build_dataset(
    csv_paths: List[Path],
    window_size: int = WIN_SIZE,
    factor: int = DS_FACTOR
) -> Tuple[np.ndarray, np.ndarray]:
    fe = FeatureExtractor(window_size=window_size)
    fe.fit_scalers(csv_paths, downsample_factor=factor)

    X_list, y_list = [], []
    for csv_path in tqdm(csv_paths):
        poi_path = csv_path.with_name(f"{csv_path.stem}_poi.csv")
        if not poi_path.exists():
            print(f"POI file not found, skipping run: {poi_path}")
            continue

        try:
            poi_indices = np.loadtxt(poi_path, dtype=int, ndmin=1)
        except Exception as e:
            print(f"Could not load POI file {poi_path} ({e}), skipping")
            continue

        df = pd.read_csv(csv_path)
        df_ds = downsample_and_rebaseline(
            df, factor=factor, window_size=window_size)
        windows = make_windows(df_ds, window_size=window_size)
        labels = label_windows(len(windows), poi_indices,
                               factor, window_size, TARGET_POIS)

        for win, lbl in zip(windows, labels):
            X_list.append(fe.transform_window(win))
            y_list.append(LABEL_MAP[lbl])

    if not X_list:
        raise ValueError(
            "No valid runs found (all POI files missing or unreadable)")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X, y

# ─── hyperparameter tuning model builder ──────────────────────────────────────


def build_tuning_model(hp):
    """Build model with hyperparameters to tune"""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(WIN_SIZE, 7)))

    # Tune number of Conv1D layers
    num_conv_layers = hp.Int(
        'num_conv_layers', min_value=2, max_value=4, default=3)

    for i in range(num_conv_layers):
        # Tune filters for each layer
        filters = hp.Int(f'conv_{i}_filters', min_value=16,
                         max_value=128, step=16, default=32)
        kernel_size = hp.Choice(
            f'conv_{i}_kernel', values=[3, 5, 7], default=3)

        model.add(tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same'
        ))

        # Tune pooling
        if hp.Boolean(f'pool_{i}', default=True):
            pool_size = hp.Choice(f'pool_{i}_size', values=[2, 3], default=2)
            model.add(tf.keras.layers.MaxPooling1D(pool_size))

        # Optional batch normalization
        if hp.Boolean(f'batch_norm_{i}', default=False):
            model.add(tf.keras.layers.BatchNormalization())

    # Global pooling choice
    pooling_type = hp.Choice('global_pooling', values=[
                             'avg', 'max'], default='avg')
    if pooling_type == 'avg':
        model.add(tf.keras.layers.GlobalAveragePooling1D())
    else:
        model.add(tf.keras.layers.GlobalMaxPooling1D())

    # Tune dense layers
    num_dense_layers = hp.Int(
        'num_dense_layers', min_value=1, max_value=3, default=1)

    for i in range(num_dense_layers):
        units = hp.Int(f'dense_{i}_units', min_value=32,
                       max_value=256, step=32, default=64)
        model.add(tf.keras.layers.Dense(units, activation='relu'))

        # Tune dropout
        dropout_rate = hp.Float(
            f'dropout_{i}', min_value=0.1, max_value=0.7, step=0.1, default=0.5)
        model.add(tf.keras.layers.Dropout(dropout_rate))

    # Output layer
    num_classes = len(LABEL_MAP)
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    # Tune optimizer and learning rate
    optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'rmsprop', 'sgd'], default='adam')
    learning_rate = hp.Float(
        'learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)

    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.9)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ─── custom tuner class for early stopping ─────────────────────────────────────


class EarlyStoppingTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['callbacks'] = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            )
        ]
        return super(EarlyStoppingTuner, self).run_trial(trial, *args, **kwargs)

# ─── main execution ─────────────────────────────────────────────────────────────


# Data preparation (same as before)
all_csvs = [
    p for p in Path("content/live/train").rglob("*.csv")
    if not p.name.endswith(("_poi.csv", "_lower.csv"))
]
train_paths, val_paths = train_test_split(
    all_csvs, test_size=0.2, random_state=42)

print("Building training dataset...")
X_train, y_train = build_dataset(train_paths)

# Balance dataset (same as before)
bg_mask = (y_train == 0)
X_bg, y_bg = X_train[bg_mask], y_train[bg_mask]
X_non0, y_non0 = X_train[~bg_mask], y_train[~bg_mask]

bg_count = len(y_bg)
X_upsampled, y_upsampled = [], []
for cls in np.unique(y_non0):
    Xc = X_non0[y_non0 == cls]
    yc = y_non0[y_non0 == cls]
    Xr, yr = resample(Xc, yc, replace=True,
                      n_samples=bg_count, random_state=42)
    X_upsampled.append(Xr)
    y_upsampled.append(yr)

X_pois = np.concatenate(X_upsampled, axis=0)
y_pois = np.concatenate(y_upsampled, axis=0)
X_train_bal = np.concatenate([X_bg, X_pois], axis=0)
y_train_bal = np.concatenate([y_bg, y_pois], axis=0)

perm = np.random.permutation(len(X_train_bal))
X_train, y_train = X_train_bal[perm], y_train_bal[perm]

unique, counts = np.unique(y_train, return_counts=True)
print("Post-balance class counts:")
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls}: {cnt}")

print("Building validation dataset...")
X_val, y_val = build_dataset(val_paths)

# ─── hyperparameter tuning ─────────────────────────────────────────────────────

print("\nStarting hyperparameter tuning...")

# Initialize tuner
tuner = EarlyStoppingTuner(
    build_tuning_model,
    objective=kt.Objective('val_accuracy', direction='max'),
    max_trials=20,  # Adjust based on your computational budget
    directory='hyperparameter_tuning',
    project_name='poi_detection_tuning'
)

# Display search space
tuner.search_space_summary()

# Search for best hyperparameters
tuner.search(
    X_train, y_train,
    epochs=15,
    validation_data=(X_val, y_val),
    batch_size=64,
    verbose=1
)

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n" + "="*50)
print("BEST HYPERPARAMETERS FOUND:")
print("="*50)

# Display best hyperparameters
for param_name in best_hp.values:
    print(f"{param_name}: {best_hp.get(param_name)}")

print(
    f"\nBest validation accuracy: {tuner.oracle.get_best_trials(1)[0].score}")

# ─── train final model with best hyperparameters ──────────────────────────────

print("\nTraining final model with best hyperparameters...")

# Create tf.data datasets
batch_size = 64
train_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .shuffle(len(X_train))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# Train the best model
history = best_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,  # More epochs for final training
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ],
    verbose=1
)

# Display final model summary
print("\n" + "="*50)
print("FINAL OPTIMIZED MODEL SUMMARY:")
print("="*50)
best_model.summary()

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the best model
best_model.save('best_poi_detection_model.h5')
print("Best model saved as 'best_poi_detection_model.h5'")

# ─── live simulation (using the optimized model) ──────────────────────────────

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
num_classes = len(LABEL_MAP)

live_dir = Path("content/live/valid")
val_paths = sorted(
    p for p in live_dir.rglob("*.csv")
    if not p.name.endswith("_poi.csv")
)

fe_live = FeatureExtractor(window_size=WIN_SIZE)
fe_live.fit_scalers(train_paths, downsample_factor=DS_FACTOR)

for sample_path in val_paths:
    print(f"\n=== Live sim on: {sample_path} ===")

    poi_path = str(sample_path).replace(".csv", "_poi.csv")
    try:
        orig_pois = np.loadtxt(poi_path, delimiter=",", dtype=int, ndmin=1)
    except:
        print(f"Could not load POI file {poi_path}, skipping...")
        continue

    df_sample = pd.read_csv(sample_path)
    df_ds = downsample_and_rebaseline(
        df_sample, factor=DS_FACTOR, window_size=WIN_SIZE)
    windows = make_windows(df_ds, window_size=WIN_SIZE)

    plt.ion()
    fig, (ax_d, ax_rf, ax_p) = plt.subplots(3, 1, figsize=(8, 10))

    buffer_t, buffer_d, buffer_rf = [], [], []

    line_d, = ax_d.plot([], [], linestyle='-', label="Dissipation")
    line_rf, = ax_rf.plot([], [], linestyle='-', label="Resonance Freq")

    for ax in (ax_d, ax_rf):
        ax.set_ylabel("Scaled Value")
        ax.legend()
        ax.grid(True)
    ax_rf.set_xlabel("Relative Time (scaled)")

    bar_containers = ax_p.bar(range(num_classes), [0] * num_classes)
    ax_p.set_ylim(0, 1)
    ax_p.set_xticks(range(num_classes))
    ax_p.set_xticklabels([INV_LABEL_MAP[i] for i in range(num_classes)])
    ax_p.set_ylabel("Probability")
    ax_p.set_title("Class Probabilities (Optimized Model)")

    poi_txt = ax_d.text(
        0.05, 0.9, "", transform=ax_d.transAxes,
        fontsize=14, color='red',
        bbox=dict(facecolor='white', alpha=0.7),
        visible=False
    )

    scatter_preds = ax_d.scatter(
        [], [], marker='X', s=100, color='red', label="Predicted POI")
    ax_d.legend()

    for i, win in enumerate(windows, start=1):
        feats = fe_live.transform_window(win)
        x, y_d, y_rf = feats[:, 0], feats[:, 1], feats[:, 2]

        # Use the optimized model for prediction
        preds = best_model.predict(feats[np.newaxis, ...], verbose=0)[0]
        cls = np.argmax(preds)
        lbl = INV_LABEL_MAP[cls]

        buffer_t.extend(x)
        buffer_d.extend(y_d)
        buffer_rf.extend(y_rf)
        line_d.set_data(buffer_t, buffer_d)
        line_rf.set_data(buffer_t, buffer_rf)

        for ax in (ax_d, ax_rf):
            ax.relim()
            ax.autoscale_view()

        if lbl != 0:
            pred_time, pred_val = x[-1], y_d[-1]
            old = scatter_preds.get_offsets()
            new = np.vstack([old, [pred_time, pred_val]]) if old.size else np.array(
                [[pred_time, pred_val]])
            scatter_preds.set_offsets(new)

        for bar, p in zip(bar_containers, preds):
            bar.set_height(p)

        fig.suptitle(
            f"Optimized Model - Run {sample_path.name} — Window {i}/{len(windows)} — Predicted POI: {lbl}")

        if lbl != 0:
            poi_txt.set_text(f"POI {lbl}")
            poi_txt.set_visible(True)
        else:
            poi_txt.set_visible(False)

        fig.canvas.draw_idle()
        plt.pause(0.1)

    plt.ioff()
    plt.show()
    plt.close(fig)

print("\nHyperparameter tuning and model optimization complete!")
