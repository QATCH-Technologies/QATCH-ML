from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import Tuple, List
import json
import joblib
from pathlib import Path  # you already import Path above, ok to keep
from datetime import datetime
import shutil
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from v4_dp import DP
import warnings
warnings.filterwarnings('ignore')
# --- top-level helper (NOT inside the class) ---------------------------------


def _prepare_one_dataset(
    data_path: str,
    poi_path: str,
    window_size: int,
    augmentations_per_poi: int,
    is_validation: bool,
    poi_indices: List[int],
    child_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single (data_path, poi_path) pair:
    - compute features
    - robust-normalize per-file
    - extract/augment windows around POIs
    Returns (X_windows_file, y_positions_file)
    """
    rng = np.random.default_rng(child_seed)

    df = pd.read_csv(data_path)
    # Heavy step — parallelizing this is the main win:
    features_df = DP.gen_features(df)
    n = len(features_df)
    if n == 0:
        return np.empty((0, window_size, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)

    features_array = features_df.values.astype(np.float32)

    # Per-file robust scaling (median + IQR)
    q1 = np.percentile(features_array, 25, axis=0)
    q3 = np.percentile(features_array, 75, axis=0)
    iqr = q3 - q1 + 1e-8
    med = np.median(features_array, axis=0)
    features_normalized = (features_array - med) / iqr
    features_normalized = np.clip(features_normalized, -5, 5)

    poi_positions_all = pd.read_csv(poi_path, header=None).values.flatten()
    poi_positions = poi_positions_all[poi_indices]

    X_windows_file = []
    y_positions_file = []

    # Validation uses a single centered sample per-POI
    local_augs = 1 if is_validation else augmentations_per_poi

    for poi_pos in poi_positions:
        if not np.isfinite(poi_pos):
            continue
        poi_pos = int(poi_pos)
        if poi_pos < 0 or poi_pos >= n:
            continue

        for aug_i in range(local_augs):
            # Choose window start so the POI is inside the window
            if n <= window_size:
                start_idx = 0
            else:
                if is_validation:
                    start_idx = max(
                        0, min(poi_pos - window_size // 2, n - window_size))
                else:
                    max_start = min(poi_pos, n - window_size)
                    min_start = max(0, poi_pos - window_size + 1)
                    if aug_i == 0:
                        start_idx = max(
                            0, min(poi_pos - window_size // 2, n - window_size))
                    else:
                        start_idx = int(rng.integers(min_start, max_start + 1))

            end_idx = min(n, start_idx + window_size)
            window = features_normalized[start_idx:end_idx].copy()

            # Mirror padding (fall back to edge if degenerate length)
            if len(window) < window_size:
                pad = window_size - len(window)
                mode = 'reflect' if len(window) > 1 else 'edge'
                window = np.pad(window, ((0, pad), (0, 0)), mode=mode)

            rel_pos = poi_pos - start_idx
            rel_pos = int(np.clip(rel_pos, 0, window_size - 1))

            # Light training-time augmentation
            if not is_validation and aug_i > 0:
                if rng.random() < 0.5:
                    window *= rng.uniform(0.95, 1.05)
                if rng.random() < 0.3:
                    window += rng.normal(0, 0.02,
                                         window.shape).astype(window.dtype)

            X_windows_file.append(window)
            y_positions_file.append(rel_pos)

    if len(X_windows_file) == 0:
        return np.empty((0, window_size, features_normalized.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.asarray(X_windows_file, dtype=np.float32), np.asarray(y_positions_file, dtype=np.int32)


class Localizer:
    """Refined POI detector with outlier handling and confidence estimation."""

    def __init__(self, window_size=128, feature_dim=None):
        """Initialize the refined detector."""
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = RobustScaler()
        self.history = None
        self.confidence_threshold = 0.7  # For filtering uncertain predictions

    def create_advanced_features(self, X):
        """Create advanced signal processing features."""
        batch_size, window_size, n_features = X.shape
        advanced_features = []

        for i in range(batch_size):
            window_features = []

            # Process each feature channel
            for j in range(n_features):
                signal_1d = X[i, :, j]

                # 1. Gradient features
                grad1 = np.gradient(signal_1d)
                grad2 = np.gradient(grad1)

                # 2. Peak detection with prominence
                peaks, properties = signal.find_peaks(
                    signal_1d, prominence=0.1, distance=3)
                peak_indicator = np.zeros_like(signal_1d)
                if len(peaks) > 0:
                    peak_indicator[peaks] = properties['prominences'] if 'prominences' in properties else 1

                # 3. Valley detection
                valleys, valley_props = signal.find_peaks(
                    -signal_1d, prominence=0.1, distance=3)
                valley_indicator = np.zeros_like(signal_1d)
                if len(valleys) > 0:
                    valley_indicator[valleys] = 1

                # 4. Local statistics (rolling window)
                window_len = 7
                padding = window_len // 2
                padded_signal = np.pad(signal_1d, padding, mode='edge')

                local_mean = np.array([
                    np.mean(padded_signal[i:i+window_len])
                    for i in range(len(signal_1d))
                ])
                local_std = np.array([
                    np.std(padded_signal[i:i+window_len])
                    for i in range(len(signal_1d))
                ])

                # 5. Signal energy
                local_energy = np.array([
                    np.sum(padded_signal[i:i+window_len]**2)
                    for i in range(len(signal_1d))
                ])

                # 6. Zero-crossing rate
                zero_crossings = np.zeros_like(signal_1d)
                zero_crossings[1:] = np.abs(
                    np.sign(signal_1d[1:]) - np.sign(signal_1d[:-1])) / 2

                # Stack all features
                feature_stack = np.stack([
                    grad1,
                    grad2,
                    peak_indicator,
                    valley_indicator,
                    signal_1d - local_mean,  # Deviation from local mean
                    local_std,
                    local_energy,
                    zero_crossings
                ], axis=-1)

                window_features.append(feature_stack)

            # Average across feature channels
            window_features = np.mean(np.array(window_features), axis=0)
            advanced_features.append(window_features)

        advanced_features = np.array(advanced_features, dtype=np.float32)

        # Concatenate with original features
        X_enhanced = np.concatenate([X, advanced_features], axis=-1)

        return X_enhanced

    def create_model(self):
        """Create refined model with confidence estimation."""
        if self.feature_dim is None:
            raise ValueError("Feature dimension must be set")
        enhanced_dim = self.feature_dim + 8
        inp = tf.keras.Input(shape=(self.window_size, enhanced_dim))
        pos_encoding = self.create_learnable_position_encoding()

        # Initial feature projection
        embedding_dim = 256
        x = tf.keras.layers.Dense(embedding_dim, activation='gelu')(inp)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        # Add positional information
        x = x + pos_encoding

        # Multi-scale CNN feature extraction
        cnn_features = []

        # Dilated convolutions for different receptive fields
        for dilation_rate in [1, 2, 4, 8]:
            conv = tf.keras.layers.Conv1D(
                64, 3,
                dilation_rate=dilation_rate,
                padding='same',
                activation='gelu'
            )(x)
            conv = tf.keras.layers.BatchNormalization()(conv)
            cnn_features.append(conv)

        # Merge CNN features
        x_cnn = tf.keras.layers.Concatenate()(cnn_features)
        x_cnn = tf.keras.layers.Dense(embedding_dim)(x_cnn)
        x = tf.keras.layers.Add()([x, x_cnn])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # Transformer encoder with residual connections
        for layer_idx in range(6):  # 6 transformer layers
            # Store input for residual
            residual = x

            # Multi-head attention with different head configurations
            num_heads = [4, 8, 8, 16, 16, 32][layer_idx]

            attn = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embedding_dim // num_heads,
                dropout=0.1
            )(x, x)

            x = tf.keras.layers.Add()([residual, attn])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

            # Feed-forward with gating
            ffn_dim = embedding_dim * 4

            # Split pathway
            gate = tf.keras.layers.Dense(ffn_dim, activation='sigmoid')(x)
            transform = tf.keras.layers.Dense(ffn_dim, activation='gelu')(x)

            gated = gate * transform
            gated = tf.keras.layers.Dropout(0.1)(gated)
            gated = tf.keras.layers.Dense(embedding_dim)(gated)

            x = tf.keras.layers.Add()([x, gated])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # Dual output heads for position and confidence

        # Position prediction head
        # Path 1: Attention-based scoring
        query_token = tf.keras.layers.Embedding(input_dim=1, output_dim=embedding_dim, name="position_query_token")(
            tf.zeros((1, 1), dtype=tf.int32)
        )  # shape: (1, 1, embedding_dim)

        # Tile to (batch, 1, embedding_dim)
        query_vector = tf.keras.layers.Lambda(
            lambda t: tf.tile(t[0], [tf.shape(t[1])[0], 1, 1])
        )([query_token, x])

        position_attn, attn_weights = tf.keras.layers.MultiHeadAttention(
            num_heads=32, key_dim=8, dropout=0.1, name="position_attn"
        )(query_vector, x, return_attention_scores=True)

        # Average attention weights across heads
        attn_scores = tf.reduce_mean(attn_weights, axis=1)
        attn_scores = tf.squeeze(attn_scores, axis=1)

        # Path 2: Conv-based position scoring
        conv_scores = tf.keras.layers.Conv1D(64, 1, activation='gelu')(x)
        conv_scores = tf.keras.layers.Conv1D(1, 1)(conv_scores)
        conv_scores = tf.keras.layers.Flatten()(conv_scores)

        # Path 3: Global context
        global_context = tf.keras.layers.GlobalAveragePooling1D()(x)
        global_dense = tf.keras.layers.Dense(
            512, activation='gelu')(global_context)
        global_dense = tf.keras.layers.Dropout(0.2)(global_dense)
        global_scores = tf.keras.layers.Dense(self.window_size)(global_dense)

        # Combine paths with learnable weights
        position_logits = tf.keras.layers.Add()([
            attn_scores * 128 * 0.4,
            conv_scores * 0.3,
            global_scores * 0.3
        ])

        # Temperature scaling
        temperature = 0.25  # Lower temperature for sharper peaks
        position_logits = position_logits / temperature

        # Position output
        position_output = tf.keras.layers.Softmax(
            name='position')(position_logits)

        # Confidence estimation head
        confidence_features = tf.keras.layers.GlobalMaxPooling1D()(x)
        confidence_dense = tf.keras.layers.Dense(
            256, activation='gelu')(confidence_features)
        confidence_dense = tf.keras.layers.Dropout(0.3)(confidence_dense)
        confidence_dense = tf.keras.layers.Dense(
            128, activation='gelu')(confidence_dense)
        confidence_output = tf.keras.layers.Dense(
            1, activation='sigmoid', name='confidence')(confidence_dense)

        model = tf.keras.Model(
            inputs=inp,
            outputs=[position_output, confidence_output],
            name="refined_poi_detector"
        )

        return model

    def create_learnable_position_encoding(self):
        """Create learnable position embeddings."""
        # Learnable embeddings
        pos_embedding = tf.keras.layers.Embedding(
            self.window_size, 256
        )(tf.range(self.window_size))

        return pos_embedding

    def create_learnable_query(self):
        """Create learnable query for attention."""
        query = tf.Variable(
            tf.random.normal([1, 1, 256]),
            trainable=True,
            name='position_query'
        )
        return tf.tile(query, [tf.shape(query)[0], 1, 1])

    def compile_model(self, model, learning_rate=0.001, warmup_steps=100, decay_steps=2000, alpha=0.05):
        """Compile with multi-task loss and a manual Warmup→Cosine LR, applied via callback."""

        # ---------- losses ----------
        def position_loss(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

            label_smoothing = 0.05
            y_true_smooth = y_true * (1 - label_smoothing) + \
                label_smoothing / self.window_size

            gamma = 2.0
            alpha_f = 0.5

            ce = -y_true_smooth * tf.math.log(y_pred)
            p_t = tf.reduce_sum(y_true * y_pred, axis=1, keepdims=True)
            focal_weight = alpha_f * tf.pow(1.0 - p_t, gamma)
            focal_loss = focal_weight * ce

            positions = tf.cast(tf.range(self.window_size), tf.float32)
            true_pos = tf.reduce_sum(y_true * positions, axis=1)
            pred_pos = tf.reduce_sum(y_pred * positions, axis=1)
            position_error = tf.keras.losses.huber(true_pos, pred_pos)

            entropy = -tf.reduce_sum(y_pred *
                                     tf.math.log(y_pred + 1e-7), axis=1)
            entropy_reg = 0.01 * entropy

            return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1) + 0.5 * position_error - entropy_reg)

        def confidence_loss(y_true, y_pred):
            return tf.keras.losses.binary_crossentropy(tf.ones_like(y_pred), y_pred)

        def accuracy_within_n(n):
            def metric(y_true, y_pred):
                true_pos = tf.argmax(y_true, axis=1)
                pred_pos = tf.argmax(y_pred, axis=1)
                return tf.reduce_mean(tf.cast(tf.abs(true_pos - pred_pos) <= n, tf.float32))
            metric.__name__ = f'acc_within_{n}'
            return metric

        # ---------- LR schedule logic (kept), but applied via a callback ----------
        class WarmupThenCosine:
            def __init__(self, init_lr, warmup_steps, decay_steps, alpha):
                self.init_lr = float(init_lr)
                self.warmup_steps = int(warmup_steps)
                self.decay_steps = int(decay_steps)
                self.alpha = float(alpha)
                self.cosine = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=self.init_lr,
                    decay_steps=self.decay_steps,
                    alpha=self.alpha
                )

            @tf.function
            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                warm = tf.cast(self.warmup_steps, tf.float32)
                warmup_lr = self.init_lr * (step + 1.0) / (warm + 1.0)
                cosine_lr = self.cosine(tf.maximum(step - warm, 0.0))
                return tf.where(step < warm, warmup_lr, cosine_lr)

        self._lr_schedule = WarmupThenCosine(
            learning_rate, warmup_steps, decay_steps, alpha)

        class WarmupCosineCallback(tf.keras.callbacks.Callback):
            def __init__(self, schedule):
                super().__init__()
                self.schedule = schedule

            def on_train_batch_begin(self, batch, logs=None):
                step = tf.cast(self.model.optimizer.iterations, tf.float32)
                lr = self.schedule(step)
                # ensure a Python float ends up in the optimizer var
                tf.keras.backend.set_value(
                    self.model.optimizer.learning_rate, float(lr.numpy()))

            def on_epoch_begin(self, epoch, logs=None):
                # also set at epoch boundaries for nice LR in logs
                step = tf.cast(self.model.optimizer.iterations, tf.float32)
                lr = self.schedule(step)
                tf.keras.backend.set_value(
                    self.model.optimizer.learning_rate, float(lr.numpy()))

        self._lr_callback = WarmupCosineCallback(self._lr_schedule)

        # IMPORTANT: give optimizer a scalar LR (not a schedule object)
        optimizer = tfa.optimizers.AdamW(
            learning_rate=float(learning_rate),
            weight_decay=1e-5,
            clipnorm=1.0
        )

        model.compile(
            optimizer=optimizer,
            loss={'position': position_loss, 'confidence': confidence_loss},
            loss_weights={'position': 1.0, 'confidence': 0.1},
            metrics={
                'position': ['accuracy',
                             accuracy_within_n(0),
                             accuracy_within_n(1),
                             accuracy_within_n(3),
                             accuracy_within_n(5),
                             accuracy_within_n(10)],
                'confidence': ['mae']
            }
        )
        return model

    def prepare_data(self, data_dir: str, num_datasets: int = None,
                     augmentations_per_poi: int = 10,
                     is_validation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Refined data preparation with parallel per-file processing."""
        poi_indices = [0, 1, 3, 4, 5]

        # Materialize for length + stable ordering
        dataset_tuples = list(DP.load_content(data_dir, num_datasets))
        if len(dataset_tuples) == 0:
            return (np.empty((0, self.window_size, self.feature_dim or 0), dtype=np.float32),
                    np.empty((0, self.window_size), dtype=np.float32))

        # Validation uses 1 augmentation per-POI
        eff_augs = 1 if is_validation else augmentations_per_poi

        # Reproducible per-file RNG seeds
        base_seed = 123 if is_validation else 42
        ss = np.random.SeedSequence(base_seed)
        child_seeds = ss.spawn(len(dataset_tuples))

        # Parallel per-file processing (DP.gen_features + windowing)
        X_chunks = [None] * len(dataset_tuples)
        y_chunks = [None] * len(dataset_tuples)

        max_workers = max(1, (os.cpu_count() or 1) - 1)  # leave one core free
        desc = f"Preparing {'validation' if is_validation else 'training'} data"

        with ProcessPoolExecutor(max_workers=max_workers) as ex, tqdm(total=len(dataset_tuples), desc=desc) as pbar:
            futures = {}
            for i, (data_path, poi_path) in enumerate(dataset_tuples):
                futures[ex.submit(
                    _prepare_one_dataset,
                    data_path,
                    poi_path,
                    self.window_size,
                    eff_augs,
                    is_validation,
                    poi_indices,
                    # child seed as plain int for portability
                    int(child_seeds[i].generate_state(1, dtype=np.uint64)[0])
                )] = i

            for fut in as_completed(futures):
                i = futures[fut]
                Xi, yi = fut.result()
                X_chunks[i] = Xi
                y_chunks[i] = yi
                pbar.update(1)

        # Concatenate in original order for determinism
        X_windows = [Xi for Xi in X_chunks if Xi is not None and len(Xi)]
        y_positions = [yi for yi in y_chunks if yi is not None and len(yi)]

        if not X_windows:
            return (np.empty((0, self.window_size, self.feature_dim or 0), dtype=np.float32),
                    np.empty((0, self.window_size), dtype=np.float32))

        X = np.concatenate(X_windows, axis=0).astype(np.float32)
        y = np.concatenate(y_positions, axis=0).astype(np.int32)

        if self.feature_dim is None:
            self.feature_dim = X.shape[2]

        # Global normalization on the aggregated set
        X_reshaped = X.reshape(-1, X.shape[-1])
        if is_validation:
            X_norm = self.scaler.transform(X_reshaped).astype(np.float32)
        else:
            X_norm = self.scaler.fit_transform(X_reshaped).astype(np.float32)
        X = X_norm.reshape(X.shape)

        # Optional feature expansion
        X = self.create_advanced_features(X)

        # One-hot encode positions
        y_onehot = tf.keras.utils.to_categorical(
            y, num_classes=self.window_size).astype(np.float32)

        print(f"Created {len(X)} windows (workers={max_workers})")
        return X, y_onehot

    def train(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=32):
        """Train with refined strategy."""

        # Create model
        print("Creating model...")
        self.model = self.create_model()
        self.model = self.compile_model(self.model, learning_rate=0.002)

        # Callbacks
        callbacks = [
            self._lr_callback,  # <— add this FIRST so LR is set before other callbacks act
            tf.keras.callbacks.EarlyStopping(
                patience=25,
                restore_best_weights=True,
                monitor='val_position_acc_within_1',
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                monitor='val_position_loss',
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_refined_model.h5',
                save_best_only=True,
                monitor='val_position_acc_within_1',
                mode='max',
                verbose=1
            )
        ]

        # Prepare confidence targets (dummy for now)
        confidence_train = np.ones((len(y_train), 1), dtype=np.float32)
        confidence_val = np.ones((len(y_val), 1), dtype=np.float32)

        # Train
        print("\nTraining model...")
        self.history = self.model.fit(
            X_train,
            {'position': y_train, 'confidence': confidence_train},
            validation_data=(
                X_val, {'position': y_val, 'confidence': confidence_val}),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def predict_with_confidence(self, X):
        """Predict with confidence filtering."""
        position_pred, confidence_pred = self.model.predict(X, verbose=0)

        # Get positions
        positions = np.argmax(position_pred, axis=1)

        # Filter by confidence
        high_confidence_mask = confidence_pred.squeeze() > self.confidence_threshold

        return positions, confidence_pred.squeeze(), high_confidence_mask

    def evaluate_refined(self, X_test, y_test):
        """Refined evaluation with confidence analysis."""
        positions, confidences, high_conf_mask = self.predict_with_confidence(
            X_test)
        true_positions = np.argmax(y_test, axis=1)

        # Overall metrics
        errors = np.abs(positions - true_positions)

        print("\n=== Overall Metrics ===")
        print(f"Total samples: {len(errors)}")
        print(f"MAE: {np.mean(errors):.2f} points")
        print(f"Median Error: {np.median(errors):.1f} points")
        print(f"Exact Match: {np.mean(errors == 0):.1%}")
        print(f"Within 1 point: {np.mean(errors <= 1):.1%}")
        print(f"Within 3 points: {np.mean(errors <= 3):.1%}")
        print(f"Within 5 points: {np.mean(errors <= 5):.1%}")

        # High confidence metrics
        if np.any(high_conf_mask):
            high_conf_errors = errors[high_conf_mask]
            print(
                f"\n=== High Confidence Predictions ({np.sum(high_conf_mask)}/{len(errors)}) ===")
            print(f"MAE: {np.mean(high_conf_errors):.2f} points")
            print(f"Exact Match: {np.mean(high_conf_errors == 0):.1%}")
            print(f"Within 1 point: {np.mean(high_conf_errors <= 1):.1%}")

        # Error distribution analysis
        print("\n=== Error Distribution ===")
        print(f"25th percentile: {np.percentile(errors, 25):.1f} points")
        print(f"50th percentile: {np.percentile(errors, 50):.1f} points")
        print(f"75th percentile: {np.percentile(errors, 75):.1f} points")
        print(f"90th percentile: {np.percentile(errors, 90):.1f} points")
        print(f"95th percentile: {np.percentile(errors, 95):.1f} points")
        print(f"99th percentile: {np.percentile(errors, 99):.1f} points")

        return errors, confidences

    def save(
        self,
        out_dir: str = "artifacts",
        filename: str = "refined_poi_detector"
    ):
        """
        Save the trained Keras model (.h5) and the fitted scaler (.joblib).
        """
        if self.model is None:
            raise RuntimeError(
                "No model to save. Train or load a model first.")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model_path = out_path / f"{filename}.h5"
        scaler_path = out_path / f"{filename}_scaler.joblib"

        # Save full model in HDF5 format
        self.model.save(model_path)

        # Save fitted scaler
        joblib.dump(self.scaler, scaler_path)

        print(f"Saved model to:   {model_path}")
        print(f"Saved scaler to:  {scaler_path}")


def main():
    """Main pipeline."""

    # Initialize
    detector = Localizer(window_size=128)

    # Prepare training data
    print("Preparing training data...")
    X_train_full, y_train_full = detector.prepare_data(
        'content/static/train',
        num_datasets=np.inf,
        augmentations_per_poi=10,
        is_validation=False
    )

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=np.argmax(y_train_full, axis=1)
    )

    print(f"Training: {X_train.shape}, Validation: {X_val.shape}")

    # Train
    history = detector.train(
        X_train, y_train,
        X_val, y_val,
        epochs=150,
        batch_size=32
    )
    # Test on separate data
    print("\nPreparing test data...")
    X_test, y_test = detector.prepare_data(
        'content/static/valid',
        num_datasets=np.inf,
        augmentations_per_poi=1,
        is_validation=True
    )

    # Evaluate
    print("\nEvaluating on test set...")
    errors, confidences = detector.evaluate_refined(X_test, y_test)

    return detector


if __name__ == "__main__":
    detector = main()
    detector.save("localizer", "v4_localizer")
