import numpy as np
import pandas as pd
import os
import random
import pickle
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import time
import threading
from queue import Queue
import logging
from sklearn.utils import resample

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


@dataclass
class ModelConfig:
    """Configuration for the neural network model"""
    input_size: int = 78  # 39 features × 2 signals (Dissipation + Resonance_Frequency)
    hidden_sizes: List[int] = None
    num_classes: int = 4   # now: no_stop, early, normal, late
    dropout_rate: float = 0.3
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 15
    validation_split: float = 0.2
    l2_regularization: float = 0.001
    window_size: int = 200  # Window size for extracting samples around stop position

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 128, 64, 32]


class FeatureExtractor:
    """Advanced feature engineering for signal data"""

    def __init__(self, window_size: int = 50):
        self.scalers = {}
        self.feature_names = []
        self.window_size = window_size
        self.fitted = False

    def extract_temporal_features(self, signal: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Extract comprehensive temporal features from signal"""
        if len(signal) == 0:
            return np.zeros(self.get_feature_count())

        features = []

        # Basic statistics
        features.extend([
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.median(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
            np.ptp(signal),  # peak-to-peak
            skew(signal),
            kurtosis(signal)
        ])

        # Signal dynamics
        if len(signal) > 1:
            diff1 = np.diff(signal)
            features.extend([
                np.mean(diff1),
                np.std(diff1),
                np.max(diff1),
                np.min(diff1),
                np.sum(diff1 > 0) / len(diff1),  # fraction increasing
                np.sum(diff1 < 0) / len(diff1),  # fraction decreasing
            ])

            if len(signal) > 2:
                diff2 = np.diff(diff1)
                features.extend([
                    np.mean(diff2),
                    np.std(diff2),
                    np.mean(np.abs(diff2))
                ])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0] * 9)

        # Zero crossings
        mean_centered = signal - np.mean(signal)
        zero_crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)
        features.append(zero_crossings / max(len(signal) - 1, 1))

        # Peak analysis
        if len(signal) > 5:
            peaks, peak_props = find_peaks(signal, height=np.mean(signal))
            valleys, valley_props = find_peaks(-signal,
                                               height=-np.mean(signal))

            features.extend([
                len(peaks) / len(signal),
                len(valleys) / len(signal),
                np.mean(peak_props['peak_heights']) if len(peaks) > 0 else 0,
                np.std(peak_props['peak_heights']) if len(peaks) > 1 else 0
            ])
        else:
            features.extend([0, 0, 0, 0])

        # Frequency domain features (using FFT)
        if len(signal) > 8:
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal))
            power_spectrum = np.abs(fft) ** 2

            # Dominant frequency
            dominant_freq_idx = np.argmax(
                power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_freq = freqs[dominant_freq_idx]

            features.extend([
                dominant_freq,
                np.sum(power_spectrum[:len(power_spectrum)//4]) /
                np.sum(power_spectrum),  # low freq power
                np.sum(power_spectrum[len(power_spectrum)//4:len(power_spectrum)//2]) / np.sum(
                    power_spectrum),  # high freq power
                np.mean(power_spectrum),
                np.std(power_spectrum)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])

        # Windowed statistics
        if len(signal) >= self.window_size:
            n_windows = len(signal) // self.window_size
            windowed_means = []
            windowed_stds = []

            for i in range(n_windows):
                start_idx = i * self.window_size
                end_idx = (i + 1) * self.window_size
                window = signal[start_idx:end_idx]
                windowed_means.append(np.mean(window))
                windowed_stds.append(np.std(window))

            features.extend([
                np.mean(windowed_means),
                np.std(windowed_means),
                np.mean(windowed_stds),
                np.std(windowed_stds),
                np.max(windowed_means) - np.min(windowed_means)
            ])
        else:
            features.extend([np.mean(signal), 0, np.std(signal), 0, 0])

        # Trend analysis
        if len(time) == len(signal) and len(signal) > 1:
            trend_coeff = np.polyfit(time, signal, 1)[0]
            features.append(trend_coeff)
        else:
            features.append(0)

        return np.array(features, dtype=np.float32)

    def get_feature_count(self) -> int:
        """Get the total number of features extracted"""
        # 39 features per signal × 2 signals (Dissipation + Resonance_Frequency)
        return 78  # Total features from both signals

    def fit_scalers(self, features: np.ndarray) -> None:
        """Fit scalers on training data"""
        self.scalers['robust'] = RobustScaler()
        self.scalers['standard'] = StandardScaler()

        self.scalers['robust'].fit(features)
        self.scalers['standard'].fit(features)
        self.fitted = True

    def transform_features(self, features: np.ndarray, scaler_type: str = 'robust') -> np.ndarray:
        """Transform features using fitted scaler"""
        if not self.fitted:
            raise ValueError("Scalers not fitted. Call fit_scalers first.")
        return self.scalers[scaler_type].transform(features)


class SignalDataset:
    """Dataset class for signal data"""

    def __init__(self, data_dir: str, feature_extractor: FeatureExtractor, config: ModelConfig, downsample_factor: int = 5):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.config = config
        self.downsample_factor = downsample_factor
        self.label_encoder = LabelEncoder()

    def downsample_data(self, data_df: pd.DataFrame, poi_indices: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Downsample data and adjust POI indices accordingly"""
        # Downsample main data
        downsampled_df = data_df.iloc[::self.downsample_factor].copy(
        ).reset_index(drop=True)

        # Map POI indices to downsampled space
        mapped_poi_indices = []
        for poi_idx in poi_indices:
            # Find the closest downsampled index
            downsampled_idx = poi_idx // self.downsample_factor
            if downsampled_idx < len(downsampled_df):
                mapped_poi_indices.append(downsampled_idx)

        return downsampled_df, np.array(mapped_poi_indices)

    def extract_sliding_windows(
        self,
        data_df: pd.DataFrame,
        stop_idx: int,
        window_size: int = 200,
        stride: int = 100
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        """Slide a window across the full series and label each window."""
        diss = data_df['Dissipation'].values
        reso = data_df['Resonance_Frequency'].values
        time = data_df['Relative_time'].values

        windows = []
        half = window_size // 2

        for start in range(0, len(data_df) - window_size + 1, stride):
            end = start + window_size
            win_d = diss[start:end]
            win_r = reso[start:end]
            win_t = time[start:end]

            rel_stop = stop_idx - start

            if rel_stop < 0 or rel_stop >= window_size:
                label = 'no_stop'
            else:
                # define regions: [0–0.3W), [0.3W–0.7W), [0.7W–W)
                p30 = int(0.3 * window_size)
                p70 = int(0.7 * window_size)
                if rel_stop < p30:
                    label = 'stop_early'
                elif rel_stop < p70:
                    label = 'stop_normal'
                else:
                    label = 'stop_late'

            windows.append((win_d, win_r, win_t, label))

        return windows

    def load_and_process_data(self, max_runs: int = np.inf, window_size: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process all data with windowing approach"""
        logger.info("Loading data files...")

        # Use the provided load_all_runs function
        runs = self.load_all_runs(self.data_dir, max_runs)

        all_features = []
        all_labels = []

        logger.info(f"Processing {len(runs)} runs with windowing approach...")

        for i, (data_csv, poi_csv) in enumerate(runs):
            try:
                # Load data - only specific columns
                required_cols = ['Dissipation',
                                 'Resonance_Frequency', 'Relative_time']
                data_df = pd.read_csv(data_csv, usecols=required_cols)

                # Verify required columns exist
                missing_cols = [
                    col for col in required_cols if col not in data_df.columns]
                if missing_cols:
                    logger.warning(
                        f"Missing columns {missing_cols} in {data_csv}")
                    continue

                # Load POI data (headerless, stop position is final entry)
                poi_data = pd.read_csv(poi_csv, header=None)
                if len(poi_data) == 0:
                    logger.warning(f"Empty POI file {poi_csv}")
                    continue

                # Get stop position (final entry in POI)
                stop_position_original = int(
                    poi_data.iloc[-1, 0])  # Last row, first column

                # Downsample data and adjust POI
                data_df_downsampled, _ = self.downsample_data(
                    data_df, np.array([stop_position_original]))
                stop_position_downsampled = stop_position_original // self.downsample_factor
                first_win = data_df_downsampled.iloc[:window_size]
                baseline_diss = first_win['Dissipation'].mean()
                baseline_res = first_win['Resonance_Frequency'].mean()

                data_df_downsampled['Dissipation'] -= baseline_diss
                data_df_downsampled['Resonance_Frequency'] -= baseline_res
                # Ensure stop position is within bounds
                if stop_position_downsampled >= len(data_df_downsampled):
                    stop_position_downsampled = len(data_df_downsampled) - 1

                # Extract multiple windows around stop position
                stride = window_size // 2  # or any stride you like
                windows = self.extract_sliding_windows(
                    data_df_downsampled,
                    stop_position_downsampled,
                    window_size,
                    stride=stride
                )
                # Process each window
                for dissipation, resonance_freq, rel_time, label in windows:
                    # Extract features from both signals
                    features_dissipation = self.feature_extractor.extract_temporal_features(
                        dissipation, rel_time)
                    features_resonance = self.feature_extractor.extract_temporal_features(
                        resonance_freq, rel_time)

                    # Combine features
                    combined_features = np.concatenate(
                        [features_dissipation, features_resonance])

                    all_features.append(combined_features)
                    all_labels.append(label)

                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(runs)} runs")

            except Exception as e:
                logger.warning(f"Error processing {data_csv}: {e}")
                continue

        if not all_features:
            raise ValueError("No valid data found")

        # Convert to arrays
        X = np.vstack(all_features)
        y = np.array(all_labels)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")
        logger.info(
            f"Label distribution: {dict(zip(self.label_encoder.classes_, np.bincount(y_encoded)))}")

        return X, y_encoded

    def load_all_runs(self, data_dir: str,
                      max_runs: int = np.inf,
                      bin_sec: float = 15.0,
                      per_bin: int = None,
                      mode: str = "undersample",
                      assign_strategy: str = "median"):
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
            per_bin = min(int(np.median(sizes) * 1.5), max(sizes))

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


class SignalClassifier:
    """TensorFlow-based neural network for signal classification"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.history = None

    def build_model(self) -> Model:
        """Build the neural network model"""
        inputs = keras.Input(
            shape=(self.config.input_size,), name='signal_features')

        x = inputs

        # Add hidden layers
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            x = layers.Dense(
                hidden_size,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(
                    self.config.l2_regularization),
                name=f'dense_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            x = layers.Dropout(self.config.dropout_rate,
                               name=f'dropout_{i+1}')(x)

        # Output layer
        outputs = layers.Dense(
            self.config.num_classes,
            activation='softmax',
            name='output'
        )(x)

        model = Model(inputs=inputs, outputs=outputs, name='signal_classifier')

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """Train the model (with training-set upsampling of minority classes)"""

        # 1) Upsample minority classes in the training set
        train_df = pd.DataFrame(X_train)
        train_df['label'] = y_train
        # majority class size
        max_count = train_df['label'].value_counts().max()

        upsampled = []
        for cls, grp in train_df.groupby('label'):
            if len(grp) < max_count:
                grp = resample(
                    grp,
                    replace=True,
                    n_samples=max_count,
                    random_state=42
                )
            upsampled.append(grp)
        train_balanced = pd.concat(upsampled).sample(frac=1, random_state=42)

        # Split back into arrays
        X_train = train_balanced.drop('label', axis=1).values
        y_train = train_balanced['label'].values

        # 2) Build model
        self.model = self.build_model()
        logger.info("Model architecture:")
        self.model.summary()

        # 3) Prepare callbacks
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.early_stopping_patience // 2,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_signal_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # 4) Split validation set if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=self.config.validation_split,
                stratify=y_train,
                random_state=42
            )

        logger.info(
            f"Training on {len(X_train)} samples (balanced), validating on {len(X_val)} samples")

        # 5) Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.num_epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )

        logger.info("Training completed!")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes, predictions

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 label_names: List[str] = None) -> Dict[str, Any]:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get predictions
        y_pred, y_prob = self.predict(X_test)

        # Calculate metrics
        test_loss, test_accuracy, test_top_k = self.model.evaluate(
            X_test, y_test, verbose=0)

        # Classification report
        if label_names is None:
            label_names = [f'Class_{i}' for i in range(
                self.config.num_classes)]

        class_report = classification_report(
            y_test, y_pred, target_names=label_names, output_dict=True)

        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top_k_accuracy': test_top_k,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_prob
        }

        return results

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history found")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'],
                 label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class LiveSimulator:
    """Live simulation buffer for real-time prediction"""

    def __init__(self, classifier: SignalClassifier, feature_extractor: FeatureExtractor,
                 buffer_size: int = 1000, update_interval: float = 0.1):
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.buffer_size = buffer_size
        self.update_interval = update_interval

        self.dissipation_buffer = deque(maxlen=buffer_size)
        self.resonance_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)
        self.prediction_queue = Queue()
        self.running = False
        self.thread = None

    def add_data_point(self,
                       dissipation: float,
                       resonance_freq: float,
                       timestamp: float = None):
        """Add a new point, but convert absolute time→relative_time internally."""
        if timestamp is None:
            timestamp = time.time()

        # On the very first point, lock in our start time
        if not hasattr(self, 'start_time'):
            self.start_time = timestamp

        # Store diss/reso as before, but re‑base time to 0
        rel_ts = timestamp - self.start_time
        self.dissipation_buffer.append(dissipation)
        self.resonance_buffer.append(resonance_freq)
        self.time_buffer.append(rel_ts)

    def start_simulation(self, data_source: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Start live simulation using provided data (dissipation, resonance_freq, time)"""
        if self.running:
            logger.warning("Simulation already running")
            return

        self.running = True
        self.data_source = data_source
        self.thread = threading.Thread(target=self._simulation_loop)
        self.thread.start()
        logger.info("Live simulation started")

    def stop_simulation(self):
        """Stop the live simulation"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Live simulation stopped")

    def _simulation_loop(self):
        """Main simulation loop with on-the-fly rebaseline."""
        dissipation_data, resonance_data, time_data = self.data_source
        idx = 0

        # We'll compute the baseline once we have 50 samples
        baseline_window = 50

        while self.running and idx < len(dissipation_data):
            # 1) Add the new (raw) data point
            self.add_data_point(
                dissipation_data[idx],
                resonance_data[idx],
                time_data[idx]
            )

            # 2) Once we have enough points, compute the baseline exactly once
            if (not hasattr(self, 'baseline_diss')) \
                    and len(self.dissipation_buffer) >= baseline_window:
                arr_d = np.array(list(self.dissipation_buffer))
                arr_r = np.array(list(self.resonance_buffer))
                self.baseline_diss = arr_d.mean()
                self.baseline_res = arr_r.mean()
                logger.info(f"Computed baseline → diss: {self.baseline_diss:.4f}, "
                            f"reso: {self.baseline_res:.4f}")

            # 3) When we have ≥50, prepare data for prediction
            if len(self.dissipation_buffer) >= baseline_window:
                try:
                    # subtract off the run‐baseline
                    dissipation_array = (np.array(list(self.dissipation_buffer))
                                         - self.baseline_diss)
                    resonance_array = (np.array(list(self.resonance_buffer))
                                       - self.baseline_res)
                    time_array = np.array(list(self.time_buffer))

                    # Extract & scale features as before
                    features_dissipation = self.feature_extractor.extract_temporal_features(
                        dissipation_array, time_array)
                    features_resonance = self.feature_extractor.extract_temporal_features(
                        resonance_array,   time_array)

                    combined_features = np.concatenate(
                        [features_dissipation, features_resonance])
                    features_scaled = self.feature_extractor.transform_features(
                        combined_features.reshape(1, -1))

                    # Predict
                    pred_class, pred_prob = self.classifier.predict(
                        features_scaled)

                    # Enqueue the result
                    self.prediction_queue.put({
                        'timestamp':      time_array[-1],
                        'predicted_class': pred_class[0],
                        'probabilities':   pred_prob[0],
                        'buffer_size':     len(self.dissipation_buffer),
                        'dissipation_current': dissipation_array[-1],
                        'resonance_current':   resonance_array[-1]
                    })

                except Exception as e:
                    logger.error(f"Prediction error: {e}")

            idx += 1
            time.sleep(self.update_interval)

    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Get the latest prediction from the queue"""
        if not self.prediction_queue.empty():
            return self.prediction_queue.get()
        return None

# Main training and evaluation pipeline


class SignalClassificationPipeline:
    """Complete pipeline for signal stop position classification"""

    def __init__(self, data_dir: str, config: ModelConfig = None):
        self.data_dir = data_dir
        self.config = config or ModelConfig()

        self.feature_extractor = FeatureExtractor()
        self.dataset = SignalDataset(
            data_dir, self.feature_extractor, self.config)
        self.classifier = SignalClassifier(self.config)
        self.simulator = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def run_training_pipeline(self, max_runs: int = np.inf, test_size: float = 0.2, window_size: int = 200):
        """Run the complete training pipeline"""
        logger.info("Starting signal classification pipeline...")

        # Load and process data
        X, y = self.dataset.load_and_process_data(max_runs, window_size)

        # Update config input size based on actual features
        self.config.input_size = X.shape[1]
        self.classifier.config.input_size = X.shape[1]

        # Fit feature scalers
        self.feature_extractor.fit_scalers(X)

        # Scale features
        X_scaled = self.feature_extractor.transform_features(X)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, stratify=y, random_state=42
        )

        logger.info(f"Training set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")

        # Train model
        self.classifier.train(self.X_train, self.y_train)

        # Evaluate model
        results = self.classifier.evaluate(self.X_test, self.y_test,
                                           list(self.dataset.label_encoder.classes_))

        logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"Test Loss: {results['test_loss']:.4f}")

        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, results['predictions'],
                                    target_names=list(self.dataset.label_encoder.classes_)))

        # Plot results
        self.classifier.plot_training_history()
        self.plot_evaluation_results(results)

        return results

    def plot_evaluation_results(self, results: Dict[str, Any]):
        """Plot evaluation results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Confusion matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues')
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # Feature importance (using model weights)
        if hasattr(self.classifier.model, 'layers'):
            first_layer_weights = self.classifier.model.layers[1].get_weights()[
                0]
            feature_importance = np.mean(np.abs(first_layer_weights), axis=1)

            top_features = np.argsort(feature_importance)[-10:]
            ax2.barh(range(len(top_features)),
                     feature_importance[top_features])
            ax2.set_title('Top 10 Feature Importance (First Layer Weights)')
            ax2.set_xlabel('Average Absolute Weight')
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels([f'Feature {i}' for i in top_features])

        plt.tight_layout()
        plt.show()

    def setup_live_simulation(self,
                              run_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                              poi_indices: np.ndarray):
        """
        Prepare and start live simulation with the same downsampling & POI mapping
        that we used for training.
        Returns:
        simulator           – the LiveSimulator running on downsampled data
        actual_stop_time_ds – the downsampled-relative_time of your POI
        """
        dissipation_data, resonance_data, time_data = run_data

        # 1. Convert incoming timestamps into relative_time (start at zero)
        time_rel = time_data - time_data[0]

        # 2. Build a temporary DataFrame, downsample + map POIs
        df = pd.DataFrame({
            'Dissipation':           dissipation_data,
            'Resonance_Frequency':   resonance_data,
            'Relative_time':         time_rel
        })
        df_ds, mapped_pois = self.dataset.downsample_data(df, poi_indices)

        # 3. Extract the downsampled arrays
        diss_ds = df_ds['Dissipation'].values
        reso_ds = df_ds['Resonance_Frequency'].values
        time_ds = df_ds['Relative_time'].values

        # 4. Compute the true stop time in downsampled coordinates
        actual_stop_time_ds = None
        if len(mapped_pois):
            poi_idx_ds = mapped_pois[-1]
            # clamp if needed
            poi_idx_ds = min(poi_idx_ds, len(time_ds)-1)
            actual_stop_time_ds = time_ds[poi_idx_ds]

        # 5. Start the simulator on this downsampled / re‑based data
        self.simulator = LiveSimulator(self.classifier, self.feature_extractor)
        self.simulator.start_simulation((diss_ds, reso_ds, time_ds))

        logger.info(f"Live simulation with downsample_factor={self.dataset.downsample_factor}, "
                    f"mapped POI at t={actual_stop_time_ds:.2f}s")
        return self.simulator, actual_stop_time_ds


def live_plot(simulator, label_encoder, actual_stop_time=None):
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top: dissipation vs time
    diss_line, = ax1.plot([], [], lw=1.5, label='Dissipation')
    est_stop_line = ax1.axvline(
        0, color='C2', ls='--', lw=2, label='Estimated Stop')
    if actual_stop_time is not None:
        actual_stop_line = ax1.axvline(actual_stop_time,
                                       color='C3', ls=':', lw=2, label='Actual Stop')
    ax1.set_ylabel('Dissipation')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Bottom: probabilities
    classes = list(label_encoder.classes_)
    bar_containers = ax2.bar(classes, [0]*len(classes))
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y')

    plt.tight_layout()

    try:
        while simulator.running:
            times = list(simulator.time_buffer)
            diss = list(simulator.dissipation_buffer)
            if times:
                diss_line.set_data(times, diss)
                ax1.relim()
                ax1.autoscale_view()

            pred = simulator.get_latest_prediction()
            if pred:
                t = pred['timestamp']
                # update the vertical line with a length‑2 sequence
                est_stop_line.set_xdata([t, t])

                for bar, h in zip(bar_containers, pred['probabilities']):
                    bar.set_height(h)

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(simulator.update_interval)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Configuration
    config = ModelConfig(
        hidden_sizes=[256, 128, 64, 32],
        num_classes=3,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=100,
        window_size=200  # Window size around stop position
    )

    # Initialize pipeline
    data_directory = "content/live/train"  # Update this path
    pipeline = SignalClassificationPipeline(data_directory, config)

    # Run training with windowing
    results = pipeline.run_training_pipeline(max_runs=1000, window_size=200)

    # Setup live simulation

    # assume you know the true stop time for this run (in same units as time_data)
    real_poi = pd.read_csv(
        "content/static/valid/02251/MM230816W2_QV1858_EL1_2_3rd_poi.csv", header=None).values
    real_data = pd.read_csv(
        "content/static/valid/02251/MM230816W2_QV1858_EL1_2_3rd.csv")
    r_time = real_data["Relative_time"].values
    diss = real_data["Dissipation"].values
    r_freq = real_data["Resonance_Frequency"].values
    stop_idx = int(real_poi[-1, 0])  # original index

    # Launch the live sim using downsampling + re‑based time
    simulator, ds_true_stop = pipeline.setup_live_simulation(
        run_data=(diss, r_freq, r_time),
        poi_indices=np.array([stop_idx])
    )

    # And now plot, passing the downsampled stop‐time
    live_plot(simulator,
              pipeline.dataset.label_encoder,
              actual_stop_time=ds_true_stop)
    # Save pipeline
    pipeline.save_pipeline()
