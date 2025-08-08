# ─── IMPROVED MULTI-CLASS POI DETECTION MODEL ─────────────────────────────────────

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

# ─── STABLE CONFIGURATION ─────────────────────────────────────────────────────

WIN_SIZE = 128
DS_FACTOR = 5

# POIs must appear in this exact order during a run: 1 → 4 → 5 → 6
POI_SEQUENCE = [1, 4, 5, 6]
TARGET_POIS = {1, 4, 5, 6}
LABEL_MAP = {0: 0, 1: 1, 4: 2, 5: 3, 6: 4}  # Original POI → Model class
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)
BACKGROUND_CLASS = 0

print(f"Sequential POI Configuration:")
print(f"  Expected sequence: {' → '.join(map(str, POI_SEQUENCE))}")
print(f"  Label mapping: {LABEL_MAP}")
print(f"  Number of classes: {NUM_CLASSES}")

# ─── EXISTING PREPROCESSING FUNCTIONS ─────────────────────────────────────────


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
            target_poi_list = sorted(list(target_pois))
            for poi_type in target_poi_list:
                if poi_type in hits:
                    labels.append(label_map[poi_type])
                    break
        else:
            labels.append(label_map[0])  # Background

    return labels

# ─── IMPROVED FEATURE EXTRACTOR ──────────────────────────────────────────────


class StableFeatureExtractor:
    def __init__(self, window_size: int = 100, scaler_cls=RobustScaler):
        self.window_size = window_size
        # Using RobustScaler for better stability with outliers
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

        # More stable feature engineering
        df["dissipation_change"] = df["Dissipation"].diff().fillna(0)
        df["rf_change"] = df["Resonance_Frequency"].diff().fillna(0)

        # Clip extreme values to prevent instability
        df["dissipation_change"] = np.clip(df["dissipation_change"], -5, 5)
        df["rf_change"] = np.clip(df["rf_change"], -5, 5)

        df["diss_x_rf"] = df["Dissipation"] * df["Resonance_Frequency"]
        df["change_prod"] = df["dissipation_change"] * df["rf_change"]

        # Add temporal position feature
        df["temporal_position"] = np.linspace(0, 1, len(df))

        # Add rolling statistics for stability
        df["diss_rolling_mean"] = df["Dissipation"].rolling(
            window=3, center=True).mean().fillna(df["Dissipation"])
        df["rf_rolling_mean"] = df["Resonance_Frequency"].rolling(
            window=3, center=True).mean().fillna(df["Resonance_Frequency"])

        cols = [
            "Relative_time", "Dissipation", "Resonance_Frequency",
            "dissipation_change", "rf_change", "diss_x_rf", "change_prod",
            "temporal_position", "diss_rolling_mean", "rf_rolling_mean"
        ]
        return df[cols].to_numpy()

# ─── IMPROVED LOSS FUNCTION ──────────────────────────────────────────────────


def stable_multiclass_loss(num_classes, temporal_weight=0.1):
    """
    More stable loss function with reduced temporal penalty
    """
    def loss_fn(y_true, y_pred):
        # Standard categorical crossentropy with label smoothing
        y_pred = tf.clip_by_value(
            y_pred, 1e-7, 1.0 - 1e-7)  # Numerical stability
        base_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False)

        # Gentler temporal penalty
        batch_size = tf.shape(y_true)[0]
        temporal_penalty = tf.linspace(
            1.5, 1.0, batch_size)  # Reduced penalty range

        # Apply penalty when predicting any POI class but true label is background
        false_positive_mask = tf.cast(tf.equal(y_true, 0), tf.float32)

        # Sum probabilities of all POI classes (non-background)
        poi_probs = tf.reduce_sum(
            y_pred[:, 1:], axis=1) if num_classes > 1 else y_pred[:, 0]
        penalty = temporal_weight * temporal_penalty * poi_probs * false_positive_mask

        return base_loss + penalty

    return loss_fn

# ─── IMPROVED MODEL BUILDER ──────────────────────────────────────────────────


def build_stable_tuning_model(hp):
    """More stable model architecture with better regularization"""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(WIN_SIZE, 10)))

    # Always add batch normalization at input
    model.add(tf.keras.layers.BatchNormalization())

    # Reduced number of conv layers for stability
    num_conv_layers = hp.Int(
        'num_conv_layers', min_value=2, max_value=3, default=2)

    for i in range(num_conv_layers):
        filters = hp.Int(f'conv_{i}_filters', min_value=16,
                         max_value=64, step=16, default=32)
        kernel_size = hp.Choice(f'conv_{i}_kernel', values=[3, 5], default=3)

        model.add(tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(
                0.001)  # L2 regularization
        ))

        # Always use batch normalization after conv layers
        model.add(tf.keras.layers.BatchNormalization())

        if i < num_conv_layers - 1:  # Don't pool after last conv layer
            pool_size = hp.Choice(f'pool_{i}_size', values=[2], default=2)
            model.add(tf.keras.layers.MaxPooling1D(pool_size))

    # Global pooling
    pooling_type = hp.Choice('global_pooling', values=[
                             'avg', 'max'], default='avg')
    if pooling_type == 'avg':
        model.add(tf.keras.layers.GlobalAveragePooling1D())
    else:
        model.add(tf.keras.layers.GlobalMaxPooling1D())

    # Reduced dense layers for stability
    num_dense_layers = hp.Int(
        'num_dense_layers', min_value=1, max_value=2, default=1)

    for i in range(num_dense_layers):
        units = hp.Int(f'dense_{i}_units', min_value=32,
                       max_value=128, step=32, default=64)
        model.add(tf.keras.layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ))

        dropout_rate = hp.Float(
            f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1, default=0.3)
        model.add(tf.keras.layers.Dropout(dropout_rate))

    # Output layer
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    # More conservative learning rates
    learning_rate = hp.Float(
        'learning_rate', min_value=1e-4, max_value=5e-3, sampling='LOG', default=1e-3)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ─── IMPROVED DATASET BUILDER ────────────────────────────────────────────────


def build_stable_dataset(
    csv_paths: List[Path],
    window_size: int = WIN_SIZE,
    factor: int = DS_FACTOR,
    target_pois: set = TARGET_POIS,
    label_map: Dict[int, int] = LABEL_MAP
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    fe = StableFeatureExtractor(window_size=window_size)
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
            temporal_pos_list.append(i / max(len(windows) - 1, 1))

    if not X_list:
        raise ValueError(
            "No valid runs found (all POI files missing or unreadable)")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    temporal_positions = np.array(temporal_pos_list, dtype=np.float32)

    return X, y, temporal_positions

# ─── IMPROVED BALANCING FUNCTION ─────────────────────────────────────────────


def stable_balance_dataset(X, y, temporal_positions, balance_ratio=0.8):
    """
    More conservative balancing to prevent overfitting
    """
    print("Balancing multi-class dataset with stability focus...")

    # Separate background and POI samples
    bg_mask = (y == BACKGROUND_CLASS)
    X_bg, y_bg, temp_bg = X[bg_mask], y[bg_mask], temporal_positions[bg_mask]

    # Get all POI samples
    poi_mask = ~bg_mask
    X_poi, y_poi, temp_poi = X[poi_mask], y[poi_mask], temporal_positions[poi_mask]

    bg_count = len(y_bg)

    # Use only a portion of background for better balance
    bg_sample_size = int(bg_count * balance_ratio)
    if bg_sample_size < bg_count:
        indices = np.random.choice(bg_count, bg_sample_size, replace=False)
        X_bg, y_bg = X_bg[indices], y_bg[indices]

    X_balanced_list = [X_bg]
    y_balanced_list = [y_bg]

    print(f"Background samples (after sampling): {len(y_bg)}")

    for poi_class in POI_CLASSES:
        poi_class_mask = (y_poi == poi_class)
        if not np.any(poi_class_mask):
            print(f"Warning: No samples found for POI class {poi_class}")
            continue

        X_class = X_poi[poi_class_mask]
        y_class = y_poi[poi_class_mask]
        temp_class = temp_poi[poi_class_mask]

        print(f"POI class {poi_class} original samples: {len(X_class)}")

        # More conservative resampling
        target_samples = min(len(y_bg) // len(POI_CLASSES), len(X_class) * 3)

        if len(X_class) < target_samples:
            # Upsample
            Xr, yr = resample(X_class, y_class, replace=True,
                              n_samples=target_samples, random_state=42)
        else:
            # Downsample
            Xr, yr = resample(X_class, y_class, replace=False,
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

# ─── IMPROVED TUNER CLASS ────────────────────────────────────────────────────


class StableTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        # More conservative callbacks
        kwargs['callbacks'] = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,  # Increased patience
                restore_best_weights=True,
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,  # Increased patience
                min_lr=1e-6,
                verbose=1
            )
        ]
        return super(StableTuner, self).run_trial(trial, *args, **kwargs)

# ─── MAIN EXECUTION WITH IMPROVEMENTS ────────────────────────────────────────


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data preparation
all_csvs = [
    p for p in Path("content/live/train").rglob("*.csv")
    if not p.name.endswith(("_poi.csv", "_lower.csv"))
]
train_paths, val_paths = train_test_split(
    all_csvs, test_size=0.2, random_state=42)

print("Building stable training dataset...")
X_train, y_train, temporal_pos_train = build_stable_dataset(
    train_paths, target_pois=TARGET_POIS, label_map=LABEL_MAP)

# Stable balancing
X_train, y_train = stable_balance_dataset(X_train, y_train, temporal_pos_train)

print("Building validation dataset...")
X_val, y_val, _ = build_stable_dataset(
    val_paths, target_pois=TARGET_POIS, label_map=LABEL_MAP)

# ─── IMPROVED HYPERPARAMETER TUNING ──────────────────────────────────────────

print(
    f"\nStarting stable hyperparameter tuning for {NUM_CLASSES}-class problem...")

tuner = StableTuner(
    build_stable_tuning_model,
    objective=kt.Objective('val_accuracy', direction='max'),
    max_trials=10,  # Reduced for stability
    directory=f'stable_multiclass_tuning_{NUM_CLASSES}classes',
    project_name=f'stable_multiclass_poi_{NUM_CLASSES}classes'
)

tuner.search(
    X_train, y_train,
    epochs=15,  # Reduced epochs for initial search
    validation_data=(X_val, y_val),
    batch_size=64,  # Larger batch size for stability
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

# ─── FINAL TRAINING WITH STABLE LOSS ─────────────────────────────────────────

print(f"\nTraining final stable {NUM_CLASSES}-class model...")

# Recompile with stable custom loss
stable_loss = stable_multiclass_loss(
    NUM_CLASSES, temporal_weight=0.05)  # Reduced weight
best_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001),  # Fixed conservative LR
    loss=stable_loss,
    metrics=['accuracy']
)

# Create datasets with prefetching
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

# Enhanced callbacks for stability
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        f'best_stable_multiclass_{NUM_CLASSES}classes.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = best_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,  # More epochs with better callbacks
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*50)
print(f"FINAL STABLE {NUM_CLASSES}-CLASS MODEL SUMMARY:")
print("="*50)
best_model.summary()

# Enhanced plotting
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'],
         label='Validation Accuracy', linewidth=2)
plt.title(f'Stable {NUM_CLASSES}-Class Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title(f'Stable {NUM_CLASSES}-Class Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
if 'lr' in history.history:
    plt.plot(history.history['lr'],
             label='Learning Rate', linewidth=2, color='red')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save model
model_filename = f'stable_multiclass_poi_model_{NUM_CLASSES}classes.h5'
best_model.save(model_filename)
print(f"Stable multi-class model saved as '{model_filename}'")

# ─── ENHANCED MULTI-CLASS PREDICTOR ──────────────────────────────────────────


class SequentialPOITracker:
    """Tracks POI detection state to enforce sequential ordering"""

    def __init__(self, poi_sequence: List[int] = POI_SEQUENCE):
        self.poi_sequence = poi_sequence
        self.sequence_map = {poi: idx for idx, poi in enumerate(poi_sequence)}
        self.reset()

    def reset(self):
        """Reset tracker for a new run"""
        self.current_stage = 0  # Index in POI_SEQUENCE
        self.detected_pois = []
        self.detection_history = []
        self.stage_confidence_history = []

    def get_expected_poi(self) -> Optional[int]:
        """Get the next expected POI in sequence"""
        if self.current_stage < len(self.poi_sequence):
            return self.poi_sequence[self.current_stage]
        return None  # All POIs detected

    def get_allowed_pois(self) -> set:
        """Get set of POIs that are currently allowed"""
        if self.current_stage < len(self.poi_sequence):
            # Only the current expected POI and background are allowed
            expected_poi = self.poi_sequence[self.current_stage]
            return {0, expected_poi}  # Background + expected POI
        return {0}  # Only background if all POIs detected

    def is_valid_prediction(self, predicted_poi: int) -> bool:
        """Check if predicted POI is valid given current state"""
        if predicted_poi == 0:  # Background always valid
            return True

        expected_poi = self.get_expected_poi()
        return predicted_poi == expected_poi

    def update_state(self, predicted_poi: int, confidence: float) -> bool:
        """Update tracker state with new prediction. Returns True if POI was accepted."""
        self.detection_history.append((predicted_poi, confidence))

        if predicted_poi == 0:  # Background
            return False

        if self.is_valid_prediction(predicted_poi):
            # Valid POI detected - advance to next stage
            self.detected_pois.append(
                (predicted_poi, confidence, len(self.detection_history)))
            self.current_stage += 1
            print(
                f"  ✅ Sequential POI {predicted_poi} detected (stage {self.current_stage}/{len(self.poi_sequence)})")
            return True
        else:
            # Invalid POI - log but don't accept
            expected = self.get_expected_poi()
            print(
                f"  ❌ Invalid POI {predicted_poi} rejected (expected: {expected})")
            return False

    def get_progress(self) -> Tuple[int, int]:
        """Get current progress (detected, total)"""
        return len(self.detected_pois), len(self.poi_sequence)

    def is_complete(self) -> bool:
        """Check if all POIs in sequence have been detected"""
        return self.current_stage >= len(self.poi_sequence)

# ─── SEQUENTIAL-AWARE PREDICTOR ───────────────────────────────────────────────


class SequentialPOIPredictor:
    """Enhanced predictor that enforces sequential POI ordering"""

    def __init__(self, model, confidence_threshold=0.65, sequence_bonus=0.1):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.sequence_bonus = sequence_bonus  # Bonus for expected POIs
        self.tracker = SequentialPOITracker()
        self.prediction_history = []

    def reset_for_new_run(self):
        """Reset predictor state for a new run"""
        self.tracker.reset()
        self.prediction_history = []

    def predict_with_sequential_constraint(self, features) -> Tuple[int, float, np.ndarray]:
        """Make prediction enforcing sequential constraints"""
        # Get raw model predictions
        raw_probs = self.model.predict(features[np.newaxis, ...], verbose=0)[0]

        # Apply sequential constraints
        adjusted_probs = self._apply_sequential_constraints(raw_probs)

        # Get final prediction
        predicted_class = np.argmax(adjusted_probs)
        confidence = adjusted_probs[predicted_class]
        predicted_poi = INV_LABEL_MAP[predicted_class]

        # Store in history
        self.prediction_history.append({
            'raw_probs': raw_probs.copy(),
            'adjusted_probs': adjusted_probs.copy(),
            'predicted_class': predicted_class,
            'predicted_poi': predicted_poi,
            'confidence': confidence
        })

        return predicted_class, confidence, adjusted_probs

    def _apply_sequential_constraints(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply sequential ordering constraints to raw probabilities"""
        adjusted_probs = raw_probs.copy()
        allowed_pois = self.tracker.get_allowed_pois()
        expected_poi = self.tracker.get_expected_poi()

        # Zero out probabilities for disallowed POIs
        for model_class in range(NUM_CLASSES):
            original_poi = INV_LABEL_MAP[model_class]
            if original_poi not in allowed_pois:
                adjusted_probs[model_class] = 0.0

        # Apply sequence bonus to expected POI
        if expected_poi is not None:
            expected_class = LABEL_MAP[expected_poi]
            adjusted_probs[expected_class] *= (1.0 + self.sequence_bonus)

        # Renormalize probabilities
        prob_sum = np.sum(adjusted_probs)
        if prob_sum > 0:
            adjusted_probs = adjusted_probs / prob_sum
        else:
            # Fallback to uniform distribution over allowed classes
            for model_class in range(NUM_CLASSES):
                original_poi = INV_LABEL_MAP[model_class]
                if original_poi in allowed_pois:
                    adjusted_probs[model_class] = 1.0 / len(allowed_pois)

        return adjusted_probs

    def force_sequential_detection(self, window_position: int, total_windows: int) -> Tuple[Optional[int], Optional[float]]:
        """Enhanced force detection that respects sequential ordering"""
        expected_poi = self.tracker.get_expected_poi()

        if expected_poi is None:  # All POIs already detected
            return None, None

        # More aggressive forcing as we approach the end
        # Earlier forcing for later stages
        force_threshold = 0.75 - (0.1 * self.tracker.current_stage)

        if window_position > force_threshold * total_windows:
            # Look for the expected POI in recent history
            if len(self.prediction_history) >= 3:
                expected_class = LABEL_MAP[expected_poi]
                # Look at last 7 predictions
                recent_entries = self.prediction_history[-7:]

                # Find the highest probability for the expected POI
                max_expected_prob = 0
                for entry in recent_entries:
                    prob = entry['raw_probs'][expected_class]
                    if prob > max_expected_prob:
                        max_expected_prob = prob

                # Lower threshold for forcing, but still reasonable
                # Lower threshold for later stages
                min_force_threshold = 0.15 - \
                    (0.02 * self.tracker.current_stage)

                if max_expected_prob > min_force_threshold:
                    print(
                        f"  🔍 Sequential force detection: POI {expected_poi} (prob: {max_expected_prob:.3f})")
                    return LABEL_MAP[expected_poi], max_expected_prob

        return None, None

# ─── ENHANCED LIVE SIMULATION WITH SEQUENTIAL ORDERING ───────────────────────


def run_sequential_poi_simulation(model, validation_paths: List[Path], feature_extractor):
    """Run live simulation with sequential POI ordering constraints"""

    print(f"\n{'='*80}")
    print(f"🎯 SEQUENTIAL POI DETECTION SIMULATION")
    print(f"{'='*80}")
    print(f"Expected sequence: {' → '.join(map(str, POI_SEQUENCE))}")
    print(f"Found {len(validation_paths)} validation files to simulate")

    # Colors for different POI classes
    colors = plt.cm.Set1(np.linspace(0, 1, NUM_CLASSES))
    class_colors = {i: colors[i] for i in range(NUM_CLASSES)}

    for sample_idx, sample_path in enumerate(validation_paths, 1):
        print(
            f"\n📊 Simulation {sample_idx}/{len(validation_paths)}: {sample_path.name}")
        print("-" * 60)

        # Load ground truth POIs
        poi_path = str(sample_path).replace(".csv", "_poi.csv")
        try:
            orig_pois = np.loadtxt(poi_path, delimiter=",", dtype=int, ndmin=1)
            print(f"Ground truth POIs: {orig_pois}")
        except:
            print(f"⚠️  Could not load POI file {poi_path}, skipping...")
            continue

        # Load and preprocess data
        df_sample = pd.read_csv(sample_path)
        df_ds = downsample_and_rebaseline(df_sample, factor=5, window_size=128)
        windows = make_windows(df_ds, window_size=128)

        print(f"Total windows to process: {len(windows)}")

        # Initialize sequential predictor
        predictor = SequentialPOIPredictor(model, confidence_threshold=0.65)
        predictor.reset_for_new_run()

        # Set up enhanced visualization
        plt.ion()
        fig = plt.figure(figsize=(18, 14))

        # Create enhanced subplot layout
        gs = fig.add_gridspec(5, 2, height_ratios=[
                              2, 2, 1, 1, 1.2], hspace=0.4, wspace=0.3)

        ax_d = fig.add_subplot(gs[0, :])      # Dissipation (full width)
        # Resonance Frequency (full width)
        ax_rf = fig.add_subplot(gs[1, :])
        ax_p = fig.add_subplot(gs[2, 0])      # Raw probabilities
        ax_adj_p = fig.add_subplot(gs[2, 1])  # Adjusted probabilities
        ax_seq = fig.add_subplot(gs[3, :])    # Sequential progress bar
        ax_summary = fig.add_subplot(gs[4, :])  # Enhanced summary

        # Initialize data buffers
        buffer_t, buffer_d, buffer_rf = [], [], []
        raw_prob_history = []
        adj_prob_history = []

        # Initialize plot lines
        line_d, = ax_d.plot([], [], 'b-', linewidth=2,
                            label="Dissipation", alpha=0.8)
        line_rf, = ax_rf.plot([], [], 'g-', linewidth=2,
                              label="Resonance Freq", alpha=0.8)

        # Configure main plots
        for ax, name in [(ax_d, "Dissipation"), (ax_rf, "Resonance Frequency")]:
            ax.set_ylabel("Scaled Value", fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{name} - Sequential Detection",
                         fontsize=12, fontweight='bold')

        ax_rf.set_xlabel("Relative Time (scaled)", fontsize=11)

        # Raw probability bars
        raw_bars = ax_p.bar(range(NUM_CLASSES), [0] * NUM_CLASSES,
                            color=[class_colors[i]
                                   for i in range(NUM_CLASSES)],
                            alpha=0.7, edgecolor='black', linewidth=1)
        ax_p.set_ylim(0, 1)
        ax_p.set_xticks(range(NUM_CLASSES))
        class_labels = [
            f"BG" if i == 0 else f"POI\n{INV_LABEL_MAP[i]}" for i in range(NUM_CLASSES)]
        ax_p.set_xticklabels(class_labels, fontsize=9)
        ax_p.set_ylabel("Probability", fontsize=10)
        ax_p.set_title("Raw Model Probabilities",
                       fontsize=11, fontweight='bold')

        # Adjusted probability bars
        adj_bars = ax_adj_p.bar(range(NUM_CLASSES), [0] * NUM_CLASSES,
                                color=[class_colors[i]
                                       for i in range(NUM_CLASSES)],
                                alpha=0.7, edgecolor='black', linewidth=1)
        ax_adj_p.set_ylim(0, 1)
        ax_adj_p.set_xticks(range(NUM_CLASSES))
        ax_adj_p.set_xticklabels(class_labels, fontsize=9)
        ax_adj_p.set_ylabel("Probability", fontsize=10)
        ax_adj_p.set_title("Sequential Adjusted Probabilities",
                           fontsize=11, fontweight='bold')

        # Sequential progress visualization
        seq_bars = ax_seq.bar(range(len(POI_SEQUENCE)), [0] * len(POI_SEQUENCE),
                              color=['lightgray'] * len(POI_SEQUENCE),
                              edgecolor='black', linewidth=2)
        ax_seq.set_ylim(0, 1)
        ax_seq.set_xticks(range(len(POI_SEQUENCE)))
        ax_seq.set_xticklabels(
            [f"POI {poi}" for poi in POI_SEQUENCE], fontsize=10)
        ax_seq.set_ylabel("Detected", fontsize=10)
        ax_seq.set_title("Sequential POI Progress",
                         fontsize=12, fontweight='bold')

        # Summary text area
        ax_summary.axis('off')
        summary_text = ax_summary.text(0.05, 0.9, "", transform=ax_summary.transAxes,
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(facecolor='lightblue', alpha=0.8, pad=10))

        # POI detection markers
        scatter_preds = ax_d.scatter([], [], marker='X', s=200, c=[], cmap='Set1',
                                     label="🎯 Sequential POIs", alpha=0.9,
                                     edgecolors='black', linewidths=3)
        ax_d.legend(fontsize=10)

        # Alert text for current detection
        alert_text = ax_d.text(0.02, 0.95, "", transform=ax_d.transAxes,
                               fontsize=13, color='red', fontweight='bold',
                               bbox=dict(facecolor='yellow', alpha=0.8, pad=5),
                               visible=False)

        # Main simulation loop
        detected_sequence = []

        for i, win in enumerate(windows, start=1):
            # Extract features and make sequential prediction
            feats = feature_extractor.transform_window(win)
            x, y_d, y_rf = feats[:, 0], feats[:, 1], feats[:, 2]

            # Sequential prediction
            model_class, confidence, adj_probs = predictor.predict_with_sequential_constraint(
                feats)
            predicted_poi = INV_LABEL_MAP[model_class]

            # Get raw probabilities for comparison
            raw_probs = predictor.prediction_history[-1]['raw_probs']

            # Check for forced detection
            forced_class, forced_conf = predictor.force_sequential_detection(
                i, len(windows))
            if forced_class is not None:
                model_class = forced_class
                confidence = forced_conf
                predicted_poi = INV_LABEL_MAP[model_class]
                # Recalculate adjusted probabilities
                adj_probs = np.zeros(NUM_CLASSES)
                adj_probs[model_class] = confidence
                adj_probs[0] = 1 - confidence  # Rest goes to background

            # Update tracker state
            poi_accepted = False
            if predicted_poi != 0:
                poi_accepted = predictor.tracker.update_state(
                    predicted_poi, confidence)

            # Update data buffers
            buffer_t.extend(x)
            buffer_d.extend(y_d)
            buffer_rf.extend(y_rf)
            raw_prob_history.append(raw_probs)
            adj_prob_history.append(adj_probs)

            # Update main plots
            line_d.set_data(buffer_t, buffer_d)
            line_rf.set_data(buffer_t, buffer_rf)

            for ax in (ax_d, ax_rf):
                ax.relim()
                ax.autoscale_view()

            # Update probability bars
            for j, (raw_bar, adj_bar, raw_p, adj_p) in enumerate(zip(raw_bars, adj_bars, raw_probs, adj_probs)):
                raw_bar.set_height(raw_p)
                adj_bar.set_height(adj_p)

                # Highlight current prediction
                if j == model_class:
                    adj_bar.set_alpha(1.0)
                    adj_bar.set_edgecolor('red')
                    adj_bar.set_linewidth(3)
                else:
                    adj_bar.set_alpha(0.6)
                    adj_bar.set_edgecolor('black')
                    adj_bar.set_linewidth(1)

            # Update sequential progress bars
            detected_count, total_count = predictor.tracker.get_progress()
            for j, bar in enumerate(seq_bars):
                if j < detected_count:
                    bar.set_color('green')
                    bar.set_height(1.0)
                elif j == detected_count:
                    bar.set_color('orange')  # Next expected
                    bar.set_height(0.5)
                else:
                    bar.set_color('lightgray')
                    bar.set_height(0.1)

            # Handle POI detection visualization
            if poi_accepted:
                pred_time, pred_val = x[-1], y_d[-1]
                detected_sequence.append((predicted_poi, confidence, i))

                # Update scatter plot
                old_offsets = scatter_preds.get_offsets()
                old_colors = scatter_preds.get_array()

                if old_offsets.size == 0:
                    new_offsets = np.array([[pred_time, pred_val]])
                    new_colors = np.array([model_class])
                else:
                    new_offsets = np.vstack(
                        [old_offsets, [pred_time, pred_val]])
                    new_colors = np.append(
                        old_colors, model_class) if old_colors.size > 0 else np.array([model_class])

                scatter_preds.set_offsets(new_offsets)
                scatter_preds.set_array(new_colors)

                # Show alert
                next_expected = predictor.tracker.get_expected_poi()
                progress_str = f"{detected_count}/{total_count}"
                alert_text.set_text(
                    f"🎯 POI {predicted_poi} DETECTED!\nProgress: {progress_str}\nNext: {next_expected if next_expected else 'Complete!'}")
                alert_text.set_visible(True)
            else:
                alert_text.set_visible(False)

            # Update enhanced summary
            expected_poi = predictor.tracker.get_expected_poi()
            progress_detected, progress_total = predictor.tracker.get_progress()

            sequence_status = " → ".join([
                f"{'✅' if j < progress_detected else '🔄' if j == progress_detected else '⏳'}{poi}"
                for j, poi in enumerate(POI_SEQUENCE)
            ])

            summary_lines = [
                f"📈 Window: {i}/{len(windows)} ({i/len(windows)*100:.1f}%)",
                f"🎯 Sequential Progress: {progress_detected}/{progress_total} POIs detected",
                f"🔄 Expected Next: {expected_poi if expected_poi else 'Complete!'}",
                f"📊 Current: POI {predicted_poi} ({'✅ Accepted' if poi_accepted else '❌ Background/Invalid'})",
                f"🚀 Sequence Status: {sequence_status}",
                f"📋 Detected Order: {' → '.join([str(poi) for poi, _, _ in detected_sequence]) if detected_sequence else 'None yet'}"
            ]
            summary_text.set_text('\n'.join(summary_lines))

            # Update main title
            completion_status = "COMPLETE!" if predictor.tracker.is_complete(
            ) else f"Expecting POI {expected_poi}"
            fig.suptitle(f"🎯 Sequential POI Detection - {sample_path.name}\n"
                         f"Progress: {i}/{len(windows)} | {completion_status}",
                         fontsize=14, fontweight='bold')

            # Refresh display
            fig.canvas.draw_idle()
            plt.pause(0.05)

        plt.ioff()

        # Final analysis for this run
        print(f"\n📋 SEQUENTIAL ANALYSIS FOR {sample_path.name}:")
        print(f"{'='*50}")

        detected_pois = [poi for poi, _, _ in detected_sequence]
        ground_truth_set = set(orig_pois)
        detected_set = set(detected_pois)

        print(f"Expected sequence: {' → '.join(map(str, POI_SEQUENCE))}")
        print(f"Ground truth POIs: {sorted(orig_pois)}")
        print(
            f"Detected sequence: {' → '.join(map(str, detected_pois)) if detected_pois else 'None'}")

        # Sequential validation
        is_valid_sequence = True
        for i, (detected_poi, conf, window) in enumerate(detected_sequence):
            expected_poi = POI_SEQUENCE[i] if i < len(POI_SEQUENCE) else None
            is_correct = detected_poi == expected_poi
            status = "✅" if is_correct else "❌"
            print(f"  {status} Stage {i+1}: Detected POI {detected_poi} (expected {expected_poi}) - conf: {conf:.3f} @ window {window}")
            if not is_correct:
                is_valid_sequence = False

        # Overall assessment
        sequence_complete = len(detected_sequence) == len(POI_SEQUENCE)
        sequence_correct = is_valid_sequence and sequence_complete

        print(f"\n🎯 ASSESSMENT:")
        print(
            f"  Sequential Order: {'✅ Correct' if is_valid_sequence else '❌ Invalid'}")
        print(
            f"  Completeness: {'✅ Complete' if sequence_complete else f'❌ Incomplete ({len(detected_sequence)}/{len(POI_SEQUENCE)})'}")
        print(
            f"  Overall: {'🎉 SUCCESS' if sequence_correct else '⚠️ PARTIAL/FAILED'}")

        # Comparison with ground truth (ignoring order for this metric)
        if detected_set == ground_truth_set:
            print(f"  Content Match: ✅ Perfect match with ground truth")
        elif detected_set.issubset(ground_truth_set):
            print(
                f"  Content Match: ⚠️ Subset of ground truth (missed: {ground_truth_set - detected_set})")
        elif ground_truth_set.issubset(detected_set):
            print(
                f"  Content Match: ⚠️ Superset of ground truth (extra: {detected_set - ground_truth_set})")
        else:
            print(
                f"  Content Match: ❌ Mismatch (missed: {ground_truth_set - detected_set}, extra: {detected_set - ground_truth_set})")

        plt.show()

        # Continue to next simulation
        if sample_idx < len(validation_paths):
            user_input = input(
                f"\n🤔 Continue to next simulation? (y/n/q to quit): ").lower().strip()
            if user_input in ['n', 'no', 'q', 'quit']:
                break

        plt.close(fig)

    print(f"\n{'='*80}")
    print("🎉 SEQUENTIAL POI DETECTION SIMULATION COMPLETE!")
    print(f"{'='*80}")

# ─── EVALUATION METRICS FOR MULTI-CLASS ─────────────────────────────────────


# After training your model and preparing validation data:
validation_paths = [p for p in Path("content/live/valid").rglob("*.csv")
                    if not p.name.endswith("_poi.csv")]

# Initialize feature extractor (assuming you have this from your main code)
feature_extractor = StableFeatureExtractor(window_size=128)
feature_extractor.fit_scalers(train_paths, downsample_factor=5)

# Run sequential simulation
run_sequential_poi_simulation(best_model, validation_paths, feature_extractor)
# def evaluate_stable_multiclass_performance(model, X_test, y_test):
#     """Enhanced evaluation for stable multi-class model"""
#     from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
#     import seaborn as sns

#     # Get predictions
#     y_pred_probs = model.predict(X_test, verbose=0)
#     y_pred = np.argmax(y_pred_probs, axis=1)

#     # Classification report
#     class_names = [f"BG" if i ==
#                    0 else f"POI_{INV_LABEL_MAP[i]}" for i in range(NUM_CLASSES)]
#     print("\n" + "="*60)
#     print("🎯 STABLE MULTI-CLASS CLASSIFICATION REPORT:")
#     print("="*60)
#     print(classification_report(y_test, y_pred, target_names=class_names))

#     # Enhanced confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names,
#                 cbar_kws={'label': 'Count'})
#     plt.title('Enhanced Multi-Class Confusion Matrix',
#               fontsize=14, fontweight='bold')
#     plt.ylabel('True Label', fontsize=12)
#     plt.xlabel('Predicted Label', fontsize=12)
#     plt.tight_layout()
#     plt.show()

#     # Per-class performance metrics
#     precision, recall, f1, support = precision_recall_fscore_support(
#         y_test, y_pred, average=None)

#     print("\n📊 PER-CLASS PERFORMANCE METRICS:")
#     print("-" * 50)
#     for i, class_name in enumerate(class_names):
#         print(f"{class_name:>8}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, "
#               f"F1={f1[i]:.3f}, Support={support[i]}")

#     return y_pred, y_pred_probs


# # Run enhanced evaluation
# print("\n🔍 Evaluating stable multi-class model performance...")
# y_pred, y_pred_probs = evaluate_stable_multiclass_performance(
#     best_model, X_val, y_val)
