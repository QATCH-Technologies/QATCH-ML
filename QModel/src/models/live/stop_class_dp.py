from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.preprocessing import RobustScaler, StandardScaler
import tensorflow as tf
from tqdm import tqdm
import keras_tuner as kt

# ─── parameters & mappings ─────────────────────────────────────────────────────

WIN_SIZE = 32
DS_FACTOR = 5

# ADAPTABLE CONFIGURATION - Change these to target different POIs
TARGET_POIS = {1, 4, 5, 6}  # Set of POI classes to detect
# Maps original POI IDs to model classes
LABEL_MAP = {0: 0, 1: 1, 4: 2, 5: 3, 6: 4}

# Derived values
NUM_CLASSES = len(LABEL_MAP)
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
BACKGROUND_CLASS = 0  # Always use 0 for background
POI_CLASSES = set(range(1, NUM_CLASSES))  # All non-background classes

print(f"Configuration:")
print(f"  Target POIs: {TARGET_POIS}")
print(f"  Label mapping: {LABEL_MAP}")
print(f"  Number of classes: {NUM_CLASSES}")
print(f"  POI classes (model): {POI_CLASSES}")

# ─── existing preprocessing funcs ───────────────────────────────────────────────


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
    target_pois: set,
    label_map: Dict[int, int]
) -> List[int]:
    """Enhanced to handle multiple POI types with priority order"""
    poi_ds = poi_indices // factor
    labels = []

    for w in range(num_windows):
        start = w * window_size
        end = start + window_size

        # Find all POIs in this window
        hits = []
        for i, ds in enumerate(poi_ds, start=1):
            if start <= ds < end and i in target_pois:
                hits.append(i)

        if hits:
            # If multiple POIs in one window, prioritize by order in TARGET_POIS
            # Sort by the order they appear in TARGET_POIS set (convert to sorted list)
            target_poi_list = sorted(list(target_pois))
            for poi_type in target_poi_list:
                if poi_type in hits:
                    labels.append(label_map[poi_type])
                    break
        else:
            labels.append(label_map[0])  # Background

    return labels

# ─── enhanced feature extractor ─────────────────────────────────────────────────


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

        # Enhanced features
        df["dissipation_change"] = df["Dissipation"].diff().fillna(0)
        df["rf_change"] = df["Resonance_Frequency"].diff().fillna(0)
        df["diss_x_rf"] = df["Dissipation"] * df["Resonance_Frequency"]
        df["change_prod"] = df["dissipation_change"] * df["rf_change"]

        # Add temporal position feature
        df["temporal_position"] = np.linspace(0, 1, len(df))

        # Add cumulative features
        df["cumulative_diss_change"] = df["dissipation_change"].cumsum()
        df["cumulative_rf_change"] = df["rf_change"].cumsum()

        cols = [
            "Relative_time", "Dissipation", "Resonance_Frequency",
            "dissipation_change", "rf_change", "diss_x_rf", "change_prod",
            "temporal_position", "cumulative_diss_change", "cumulative_rf_change"
        ]
        return df[cols].to_numpy()

# ─── multi-class aware loss function ────────────────────────────────────────────


def temporal_weighted_multiclass_loss(num_classes):
    """
    Creates a custom loss function for multi-class POI detection
    that penalizes early false positives more than late ones
    """
    def loss_fn(y_true, y_pred):
        # Standard categorical crossentropy
        base_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred)

        # Create temporal penalty: higher penalty for early predictions
        batch_size = tf.shape(y_true)[0]
        temporal_penalty = tf.linspace(
            2.0, 1.0, batch_size)  # Higher penalty early

        # Apply penalty when predicting any POI class but true label is background
        false_positive_mask = tf.cast(tf.equal(y_true, 0), tf.float32)

        # Sum probabilities of all POI classes (non-background)
        poi_probs = tf.reduce_sum(
            y_pred[:, 1:], axis=1) if num_classes > 1 else y_pred[:, 0]
        penalty = temporal_penalty * poi_probs * false_positive_mask

        return base_loss + penalty

    return loss_fn

# ─── dataset builder with multi-class support ──────────────────────────────────


def build_dataset_with_temporal_info(
    csv_paths: List[Path],
    window_size: int = WIN_SIZE,
    factor: int = DS_FACTOR,
    target_pois: set = TARGET_POIS,
    label_map: Dict[int, int] = LABEL_MAP
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fe = FeatureExtractor(window_size=window_size)
    fe.fit_scalers(csv_paths, downsample_factor=factor)

    X_list, y_list, temporal_pos_list = [], [], []

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
                               factor, window_size, target_pois, label_map)

        for i, (win, lbl) in enumerate(zip(windows, labels)):
            X_list.append(fe.transform_window(win))
            y_list.append(lbl)
            # Store temporal position (0 = early, 1 = late)
            temporal_pos_list.append(i / max(len(windows) - 1, 1))

    if not X_list:
        raise ValueError(
            "No valid runs found (all POI files missing or unreadable)")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    temporal_positions = np.array(temporal_pos_list, dtype=np.float32)
    return X, y, temporal_positions

# ─── adaptive model builder ────────────────────────────────────────────────────


def build_tuning_model(hp):
    """Enhanced model with adaptive output layer"""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(WIN_SIZE, 10)))

    # Tune number of Conv1D layers
    num_conv_layers = hp.Int(
        'num_conv_layers', min_value=2, max_value=4, default=3)

    for i in range(num_conv_layers):
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

        if hp.Boolean(f'pool_{i}', default=True):
            pool_size = hp.Choice(f'pool_{i}_size', values=[2, 3], default=2)
            model.add(tf.keras.layers.MaxPooling1D(pool_size))

        if hp.Boolean(f'batch_norm_{i}', default=False):
            model.add(tf.keras.layers.BatchNormalization())

    # Global pooling choice
    pooling_type = hp.Choice('global_pooling', values=[
                             'avg', 'max'], default='avg')
    if pooling_type == 'avg':
        model.add(tf.keras.layers.GlobalAveragePooling1D())
    else:
        model.add(tf.keras.layers.GlobalMaxPooling1D())

    # Dense layers
    num_dense_layers = hp.Int(
        'num_dense_layers', min_value=1, max_value=3, default=2)

    for i in range(num_dense_layers):
        units = hp.Int(f'dense_{i}_units', min_value=32,
                       max_value=256, step=32, default=64)
        model.add(tf.keras.layers.Dense(units, activation='relu'))

        dropout_rate = hp.Float(
            f'dropout_{i}', min_value=0.1, max_value=0.7, step=0.1, default=0.5)
        model.add(tf.keras.layers.Dropout(dropout_rate))

    # Adaptive output layer based on NUM_CLASSES
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    # Optimizer selection
    optimizer_choice = hp.Choice(
        'optimizer', values=['adam', 'rmsprop'], default='adam')
    learning_rate = hp.Float(
        'learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)

    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ─── multi-class enhanced predictor ─────────────────────────────────────────────


class MultiClassEnhancedPredictor:
    def __init__(self, model, confidence_threshold=0.7, num_classes=NUM_CLASSES):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        self.prediction_history = []

    def predict_with_confidence(self, features):
        """Predict with confidence thresholding for multi-class"""
        probs = self.model.predict(features[np.newaxis, ...], verbose=0)[0]
        confidence = np.max(probs)
        predicted_class = np.argmax(probs)

        # Store prediction history
        self.prediction_history.append((predicted_class, confidence, probs))

        # Keep only recent history
        if len(self.prediction_history) > 5:
            self.prediction_history.pop(0)

        return predicted_class, confidence, probs

    def force_poi_if_missing(self, window_position, total_windows):
        """Force POI detection if we're late and haven't found any POI"""
        if window_position > 0.8 * total_windows:  # In last 20% of windows
            # Check if any POI class has been predicted
            poi_predictions = [
                p[0] for p in self.prediction_history if p[0] != BACKGROUND_CLASS]

            if not poi_predictions:  # No POI detected yet
                # Find the most confident POI prediction from recent history
                if self.prediction_history:
                    recent_probs = [p[2] for p in self.prediction_history[-3:]]

                    # Get max probability for any POI class across recent windows
                    max_poi_prob = 0
                    best_poi_class = None

                    for probs in recent_probs:
                        for poi_class in POI_CLASSES:
                            if probs[poi_class] > max_poi_prob:
                                max_poi_prob = probs[poi_class]
                                best_poi_class = poi_class

                    # Lower threshold for late detection
                    if max_poi_prob > 0.25:
                        return best_poi_class, max_poi_prob

        return None, None

# ─── adaptive balancing function ────────────────────────────────────────────────


def balance_multiclass_dataset(X, y, temporal_positions, target_ratio='auto'):
    """
    Balance dataset for multiple POI classes
    """
    print("Balancing multi-class dataset...")

    # Separate background and POI samples
    bg_mask = (y == BACKGROUND_CLASS)
    X_bg, y_bg, temp_bg = X[bg_mask], y[bg_mask], temporal_positions[bg_mask]

    # Get all POI samples
    poi_mask = ~bg_mask
    X_poi, y_poi, temp_poi = X[poi_mask], y[poi_mask], temporal_positions[poi_mask]

    bg_count = len(y_bg)

    # Balance each POI class
    X_balanced_list = [X_bg]
    y_balanced_list = [y_bg]

    print(f"Background samples: {bg_count}")

    for poi_class in POI_CLASSES:
        poi_class_mask = (y_poi == poi_class)
        if not np.any(poi_class_mask):
            print(f"Warning: No samples found for POI class {poi_class}")
            continue

        X_class = X_poi[poi_class_mask]
        y_class = y_poi[poi_class_mask]
        temp_class = temp_poi[poi_class_mask]

        print(f"POI class {poi_class} original samples: {len(X_class)}")

        # Bias towards later examples for this class
        late_mask = temp_class > 0.5

        if np.any(late_mask):
            late_X, late_y = X_class[late_mask], y_class[late_mask]
            early_X, early_y = X_class[~late_mask], y_class[~late_mask]

            # Target samples per class (equal to background or proportional)
            target_samples = bg_count // len(POI_CLASSES)  # Equal distribution

            # 70% late, 30% early
            n_late = int(0.7 * target_samples)
            n_early = target_samples - n_late

            if len(late_X) > 0 and len(early_X) > 0:
                Xr_late, yr_late = resample(late_X, late_y, replace=True,
                                            n_samples=min(n_late, target_samples), random_state=42)
                Xr_early, yr_early = resample(early_X, early_y, replace=True,
                                              n_samples=min(n_early, target_samples), random_state=42)
                Xr = np.concatenate([Xr_late, Xr_early])
                yr = np.concatenate([yr_late, yr_early])
            elif len(late_X) > 0:
                Xr, yr = resample(late_X, late_y, replace=True,
                                  n_samples=target_samples, random_state=42)
            else:
                Xr, yr = resample(early_X, early_y, replace=True,
                                  n_samples=target_samples, random_state=42)
        else:
            # No temporal split possible, just resample
            target_samples = bg_count // len(POI_CLASSES)
            Xr, yr = resample(X_class, y_class, replace=True,
                              n_samples=target_samples, random_state=42)

        X_balanced_list.append(Xr)
        y_balanced_list.append(yr)
        print(f"POI class {poi_class} balanced samples: {len(Xr)}")

    # Combine all classes
    X_balanced = np.concatenate(X_balanced_list, axis=0)
    y_balanced = np.concatenate(y_balanced_list, axis=0)

    # Shuffle
    perm = np.random.permutation(len(X_balanced))
    X_balanced, y_balanced = X_balanced[perm], y_balanced[perm]

    # Print final distribution
    unique, counts = np.unique(y_balanced, return_counts=True)
    print("Final balanced class distribution:")
    for cls, cnt in zip(unique, counts):
        original_poi = INV_LABEL_MAP[cls]
        print(f"  Class {cls} (POI {original_poi}): {cnt}")

    return X_balanced, y_balanced

# ─── custom tuner class ─────────────────────────────────────────────────────────


class EarlyStoppingTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['callbacks'] = [
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
        ]
        return super(EarlyStoppingTuner, self).run_trial(trial, *args, **kwargs)

# ─── main execution ─────────────────────────────────────────────────────────────


# Data preparation
all_csvs = [
    p for p in Path("content/live/train").rglob("*.csv")
    if not p.name.endswith(("_poi.csv", "_lower.csv"))
]
train_paths, val_paths = train_test_split(
    all_csvs, test_size=0.2, random_state=42)

print("Building training dataset with temporal information...")
X_train, y_train, temporal_pos_train = build_dataset_with_temporal_info(
    train_paths, target_pois=TARGET_POIS, label_map=LABEL_MAP)

# Multi-class balancing
X_train, y_train = balance_multiclass_dataset(
    X_train, y_train, temporal_pos_train)

print("Building validation dataset...")
X_val, y_val, _ = build_dataset_with_temporal_info(
    val_paths, target_pois=TARGET_POIS, label_map=LABEL_MAP)

# ─── hyperparameter tuning ─────────────────────────────────────────────────────

print(f"\nStarting hyperparameter tuning for {NUM_CLASSES}-class problem...")

tuner = EarlyStoppingTuner(
    build_tuning_model,
    objective=kt.Objective('val_accuracy', direction='max'),
    max_trials=15,
    directory=f'multiclass_tuning_{NUM_CLASSES}classes',
    project_name=f'multiclass_poi_detection_{NUM_CLASSES}classes'
)

tuner.search_space_summary()

tuner.search(
    X_train, y_train,
    epochs=20,
    validation_data=(X_val, y_val),
    batch_size=32,
    verbose=1
)

best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n" + "="*50)
print("BEST HYPERPARAMETERS FOUND:")
print("="*50)

for param_name in best_hp.values:
    print(f"{param_name}: {best_hp.get(param_name)}")

print(
    f"\nBest validation accuracy: {tuner.oracle.get_best_trials(1)[0].score}")

# ─── train final model with custom loss ───────────────────────────────────────

print(f"\nTraining final {NUM_CLASSES}-class model...")

# Recompile with custom loss
custom_loss = temporal_weighted_multiclass_loss(NUM_CLASSES)
best_model.compile(
    optimizer=best_model.optimizer,
    loss=custom_loss,
    metrics=['accuracy']
)

batch_size = 32
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

history = best_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7
        )
    ],
    verbose=1
)

print("\n" + "="*50)
print(f"FINAL {NUM_CLASSES}-CLASS MODEL SUMMARY:")
print("="*50)
best_model.summary()

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'{NUM_CLASSES}-Class Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'{NUM_CLASSES}-Class Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model_filename = f'multiclass_poi_model_{NUM_CLASSES}classes.h5'
best_model.save(model_filename)
print(f"Multi-class model saved as '{model_filename}'")

# ─── enhanced live simulation ──────────────────────────────────────────────────

live_dir = Path("content/live/valid")
val_paths = sorted(
    p for p in live_dir.rglob("*.csv")
    if not p.name.endswith("_poi.csv")
)

fe_live = FeatureExtractor(window_size=WIN_SIZE)
fe_live.fit_scalers(train_paths, downsample_factor=DS_FACTOR)

# Create color map for different POI classes
colors = plt.cm.Set1(np.linspace(0, 1, NUM_CLASSES))
class_colors = {i: colors[i] for i in range(NUM_CLASSES)}

for sample_path in val_paths:
    print(f"\n=== Multi-Class Live Simulation: {sample_path} ===")

    poi_path = str(sample_path).replace(".csv", "_poi.csv")
    try:
        orig_pois = np.loadtxt(poi_path, delimiter=",", dtype=int, ndmin=1)
        print(f"Original POIs in file: {orig_pois}")
    except:
        print(f"Could not load POI file {poi_path}, skipping...")
        continue

    df_sample = pd.read_csv(sample_path)
    df_ds = downsample_and_rebaseline(
        df_sample, factor=DS_FACTOR, window_size=WIN_SIZE)
    windows = make_windows(df_ds, window_size=WIN_SIZE)

    # Initialize multi-class predictor
    predictor = MultiClassEnhancedPredictor(
        best_model, confidence_threshold=0.6)
    detected_pois = []

    plt.ion()
    fig, (ax_d, ax_rf, ax_p) = plt.subplots(3, 1, figsize=(12, 14))

    buffer_t, buffer_d, buffer_rf = [], [], []

    line_d, = ax_d.plot([], [], linestyle='-', label="Dissipation")
    line_rf, = ax_rf.plot([], [], linestyle='-', label="Resonance Freq")

    for ax in (ax_d, ax_rf):
        ax.set_ylabel("Scaled Value")
        ax.legend()
        ax.grid(True)
    ax_rf.set_xlabel("Relative Time (scaled)")

    # Multi-class probability bars
    bar_containers = ax_p.bar(range(NUM_CLASSES), [0] * NUM_CLASSES,
                              color=[class_colors[i] for i in range(NUM_CLASSES)])
    ax_p.set_ylim(0, 1)
    ax_p.set_xticks(range(NUM_CLASSES))
    class_labels = [
        f"BG" if i == 0 else f"POI {INV_LABEL_MAP[i]}" for i in range(NUM_CLASSES)]
    ax_p.set_xticklabels(class_labels, rotation=45)
    ax_p.set_ylabel("Probability")
    ax_p.set_title(f"Multi-Class Probabilities ({NUM_CLASSES} classes)")

    poi_txt = ax_d.text(
        0.05, 0.9, "", transform=ax_d.transAxes,
        fontsize=14, color='red',
        bbox=dict(facecolor='white', alpha=0.8),
        visible=False
    )

    # Multi-class scatter plot
    scatter_preds = ax_d.scatter(
        [], [], marker='X', s=120, c=[], cmap='Set1',
        label="Predicted POIs", alpha=0.8)
    ax_d.legend()

    for i, win in enumerate(windows, start=1):
        feats = fe_live.transform_window(win)
        x, y_d, y_rf = feats[:, 0], feats[:, 1], feats[:, 2]

        # Multi-class prediction
        cls, confidence, preds = predictor.predict_with_confidence(feats)

        # Check if we need to force POI detection
        forced_cls, forced_conf = predictor.force_poi_if_missing(
            i, len(windows))
        if forced_cls is not None:
            cls = forced_cls
            confidence = forced_conf
            print(
                f"  Forced POI detection: class {cls} (POI {INV_LABEL_MAP[cls]}) at window {i} (confidence: {confidence:.3f})")

        original_poi = INV_LABEL_MAP[cls]

        buffer_t.extend(x)
        buffer_d.extend(y_d)
        buffer_rf.extend(y_rf)
        line_d.set_data(buffer_t, buffer_d)
        line_rf.set_data(buffer_t, buffer_rf)

        for ax in (ax_d, ax_rf):
            ax.relim()
            ax.autoscale_view()

        # Plot POI predictions with class-specific colors
        if cls != BACKGROUND_CLASS:
            pred_time, pred_val = x[-1], y_d[-1]
            old_offsets = scatter_preds.get_offsets()
            old_colors = scatter_preds.get_array()

            # Add new point
            if old_offsets.size == 0:
                new_offsets = np.array([[pred_time, pred_val]])
                new_colors = np.array([cls])
            else:
                new_offsets = np.vstack([old_offsets, [pred_time, pred_val]])
                new_colors = np.append(
                    old_colors, cls) if old_colors.size > 0 else np.array([cls])

            scatter_preds.set_offsets(new_offsets)
            scatter_preds.set_array(new_colors)
            detected_pois.append((i, original_poi, confidence, cls))

        # Update probability bars with class-specific colors
        for j, (bar, p) in enumerate(zip(bar_containers, preds)):
            bar.set_height(p)
            # Highlight the predicted class
            if j == cls:
                bar.set_alpha(1.0)
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
            else:
                bar.set_alpha(0.6)
                bar.set_edgecolor('none')
                bar.set_linewidth(0)

        fig.suptitle(
            f"Multi-Class Model - {sample_path.name} — Window {i}/{len(windows)} — Predicted: POI {original_poi} (conf: {confidence:.2f})")

        if cls != BACKGROUND_CLASS:
            poi_txt.set_text(f"POI {original_poi} (conf: {confidence:.2f})")
            poi_txt.set_visible(True)
        else:
            poi_txt.set_visible(False)

        fig.canvas.draw_idle()
        plt.pause(0.1)

    plt.ioff()

    # Summary for this run
    print(f"  Total POIs detected: {len(detected_pois)}")
    poi_summary = {}
    for window_idx, poi_type, conf, model_class in detected_pois:
        if poi_type not in poi_summary:
            poi_summary[poi_type] = []
        poi_summary[poi_type].append((window_idx, conf))

    for poi_type in sorted(poi_summary.keys()):
        detections = poi_summary[poi_type]
        print(f"    POI {poi_type}: {len(detections)} detections")
        for window_idx, conf in detections:
            print(f"      Window {window_idx}: confidence {conf:.3f}")

    plt.show()
    plt.close(fig)

print(f"\nMulti-class model training and simulation complete!")
print(f"Configuration used:")
print(f"  Target POIs: {TARGET_POIS}")
print(f"  Number of classes: {NUM_CLASSES}")
print(f"  Model saved as: {model_filename}")

# ─── evaluation metrics for multi-class ────────────────────────────────────────


def evaluate_multiclass_performance(model, X_test, y_test):
    """Evaluate multi-class model performance"""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    class_names = [f"BG" if i ==
                   0 else f"POI_{INV_LABEL_MAP[i]}" for i in range(NUM_CLASSES)]
    print("\n" + "="*50)
    print("MULTI-CLASS CLASSIFICATION REPORT:")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Multi-Class Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    return y_pred, y_pred_probs


# Run evaluation on validation set
print("\nEvaluating multi-class model performance...")
y_pred, y_pred_probs = evaluate_multiclass_performance(
    best_model, X_val, y_val)

print("\nKey Features of the Adaptive Multi-Class System:")
print("=" * 60)
print("✓ Configurable TARGET_POIS and LABEL_MAP")
print("✓ Automatic class balancing for all POI types")
print("✓ Multi-class temporal weighted loss function")
print("✓ Enhanced predictor with forced late detection")
print("✓ Color-coded visualization for different POI classes")
print("✓ Comprehensive evaluation metrics")
print("\nTo change POI configuration:")
print("1. Update TARGET_POIS = {1, 4, 5, 6}")
print("2. Update LABEL_MAP = {0: 0, 1: 1, 4: 2, 5: 3, 6: 4}")
print("3. Re-run the script")
