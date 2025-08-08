from joblib import Parallel, delayed
from tqdm import tqdm
import keras_tuner as kt
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
from v4_dp import DP
warnings.filterwarnings('ignore')


class POIDetectionSystem:
    """
    A TensorFlow-based system for detecting Points of Interest in viscosity sensor data.
    Uses multi-label classification to handle overlapping POIs and temporal relationships.
    """

    def __init__(self, window_size: int = 50, stride: int = 10, tolerance: int = 100):
        """
        Initialize the Enhanced POI Detection System.

        Args:
            window_size: Size of the sliding window for feature extraction
            stride: Step size for sliding window
            tolerance: Acceptable distance (in points) for POI detection
        """
        self.window_size = window_size
        self.stride = stride
        self.tolerance = tolerance
        self.model = None
        self.scaler = StandardScaler()
        self.feature_dim = None
        self.best_hp: Optional[kt.HyperParameters] = None

    def load_and_prepare_data(self, data_dir: str, num_datasets: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data using DP.load_content and prepare for training.

        Args:
            data_dir: Directory containing data files
            num_datasets: Number of datasets to load

        Returns:
            X: Feature array
            y: Multi-label array (shape: [n_samples, 6] for POI 0,1,2,4,5,6)
        """
        # Load data pairs
        if num_datasets is None:
            file_pairs = DP.load_content(data_dir)
        else:
            file_pairs = DP.load_content(data_dir, num_datasets)

        all_features = []
        all_labels = []

        def process_file_pair(data_file, poi_file, self_instance):
            """
            Contains all the processing logic for a single data_file and poi_file pair.
            """
            # 1. Read data file (using a faster engine)
            # Use a faster engine like pyarrow
            df = pd.read_csv(data_file, engine='pyarrow')

            # 2. Generate features using DP.gen_features
            # Assuming DP is accessible or passed into this scope
            features_df = DP.gen_features(df)

            # 3. Read and process POI file
            poi_indices = pd.read_csv(poi_file, header=None).values.flatten()
            poi_indices = poi_indices[[0, 1, 3, 4, 5]]

            # 4. Create sliding windows and multi-labels
            # Note: Pass 'self' as an argument if the method depends on instance state
            windows, labels = self_instance._create_windows_and_multilabels(
                features_df, poi_indices
            )

            return windows, labels

        results = Parallel(n_jobs=-1)(
            delayed(process_file_pair)(data_file, poi_file, self)
            for data_file, poi_file in tqdm(file_pairs, desc="<Processing files>")
        )

        all_features, all_labels = zip(*results)
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        # # Concatenate all data
        # X = np.concatenate(all_features, axis=0)
        # y = np.concatenate(all_labels, axis=0)
        return X, y

    def _create_windows_and_multilabels(self, features_df: pd.DataFrame, poi_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows and corresponding multi-labels with sequential awareness.

        Args:
            features_df: DataFrame with generated features
            poi_indices: Array of POI indices

        Returns:
            windows: Array of feature windows
            labels: Multi-label array where each row indicates presence of POIs
        """
        features = features_df.values
        n_samples = len(features)

        windows = []
        labels = []

        # Map POI positions to their indices
        poi_map = {
            1: poi_indices[0],  # POI 1
            2: poi_indices[1],  # POI 2
            4: poi_indices[2],  # POI 4 (skipping 3)
            5: poi_indices[3],  # POI 5
            6: poi_indices[4]   # POI 6
        }

        for i in range(0, n_samples - self.window_size, self.stride):
            window = features[i:i + self.window_size]
            windows.append(window)

            # Create multi-label for this window
            window_center = i + self.window_size // 2
            window_label = np.zeros(6)

            # Check each POI
            is_poi = False
            for poi_class in [1, 2, 4, 5, 6]:
                poi_position = poi_map.get(poi_class)
                if poi_position is not None and abs(window_center - poi_position) <= self.tolerance:
                    # Map POI class to label index
                    label_idx = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}[poi_class]
                    window_label[label_idx] = 1
                    is_poi = True

            # Set non-POI flag if no POIs detected
            if not is_poi:
                window_label[0] = 1

            labels.append(window_label)

        return np.array(windows), np.array(labels)

    def _build_model(self, hp: kt.engine.hyperparameters.HyperParameters) -> Model:
        """Build model with hyperparameter tuning support."""
        inputs = layers.Input(shape=(self.window_size, self.feature_dim))

        # Tune conv filters and kernel size
        conv_filters = hp.Int('conv_filters', min_value=32,
                              max_value=128, step=32)
        kernel_size = hp.Choice('kernel_size', values=[3, 5, 7])
        x = layers.Conv1D(conv_filters, kernel_size=kernel_size,
                          activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)

        # Tune additional conv layer
        conv_filters_2 = hp.Int('conv_filters_2', 64, 256, step=64)
        x = layers.Conv1D(conv_filters_2, kernel_size=3,
                          activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)

        # Tune LSTM units
        lstm_units = hp.Int('lstm_units', 64, 256, step=64)
        x = layers.Bidirectional(layers.LSTM(
            lstm_units, return_sequences=True))(x)
        x = layers.Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1))(x)
        x = layers.Bidirectional(layers.LSTM(
            hp.Int('lstm_units_2', 32, 128, step=32)))(x)
        x = layers.Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1))(x)

        # Tune dense layer size
        dense_units = hp.Int('dense_units', 64, 256, step=64)
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp.Float('dropout_3', 0.1, 0.5, step=0.1))(x)

        # Output layer
        outputs = layers.Dense(6, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Tune learning rate
        lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['binary_accuracy', keras.metrics.AUC(name='auc')]
        )
        return model

    def tune(self, X: np.ndarray, y: np.ndarray, max_trials: int = 10,
             executions_per_trial: int = 1, directory: str = 'kt_tuner',
             project_name: str = 'poi_detection') -> kt.Tuner:
        """Perform hyperparameter search and store best_hp."""
        self.feature_dim = X.shape[-1]
        # normalize
        flat = X.reshape(-1, self.feature_dim)
        flat = self.scaler.fit_transform(flat)
        Xn = flat.reshape(X.shape)

        X_train, X_val, y_train, y_val = train_test_split(
            Xn, y, test_size=0.2, random_state=42)

        auc_obj = kt.Objective("val_auc", direction="max")

        tuner = kt.Hyperband(
            self._build_model,
            objective=auc_obj,
            max_epochs=15,
            factor=3,
            executions_per_trial=executions_per_trial,
            directory=directory,
            project_name=project_name
        )

        stop = keras.callbacks.EarlyStopping('val_loss', patience=5)
        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            sample_weight=self.compute_sample_weights(y_train),
            epochs=50,
            batch_size=32,
            callbacks=[stop]
        )
        self.best_hp = tuner.get_best_hyperparameters(1)[0]
        self.model = tuner.hypermodel.build(self.best_hp)
        return tuner

    def compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Compute sample weights to handle class imbalance in multi-label setting.
        """
        weights = np.ones(len(y))
        label_counts = np.sum(y, axis=0)
        total_samples = len(y)

        label_weights = {}
        for i in range(6):
            if label_counts[i] > 0:
                label_weights[i] = total_samples / (6 * label_counts[i])
                if i > 0:  # POI classes
                    label_weights[i] *= 10
            else:
                label_weights[i] = 1.0

        for i in range(len(y)):
            sample_weight = 1.0
            for j in range(6):
                if y[i, j] == 1:
                    sample_weight = max(sample_weight, label_weights[j])
            weights[i] = sample_weight

        return weights

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
              batch_size: int = 32, validation_split: float = 0.2) -> keras.callbacks.History:
        """Train using either the tuned model or default architecture."""
        # report distribution
        names = ['Non-POI', 'POI1', 'POI2', 'POI4', 'POI5', 'POI6']
        for i, nm in enumerate(names):
            print(f"{nm}: {y[:,i].sum()} samples")

        # normalize
        flat = X.reshape(-1, X.shape[-1])
        flat = self.scaler.transform(flat)
        Xn = flat.reshape(X.shape)

        # split
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xn, y, test_size=validation_split, random_state=42)

        if not self.best_hp:
            self.model = self._build_model(kt.HyperParameters())

        sw = self.compute_sample_weights(y_tr)
        callbacks = [
            keras.callbacks.EarlyStopping(
                'val_loss', patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                'val_loss', factor=0.5, patience=5, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5', 'val_loss', save_best_only=True)
        ]
        return self.model.fit(X_tr, y_tr,
                              validation_data=(X_val, y_val),
                              sample_weight=sw,
                              epochs=epochs,
                              batch_size=batch_size,
                              callbacks=callbacks)

    def predict_pois(self, df: pd.DataFrame, threshold: float = 0.5,
                     adaptive_thresholds: Dict[int, float] = None,
                     use_nms: bool = True, nms_window: int = 30,
                     enforce_sequential: bool = True,
                     enforce_relative_gaps: bool = True) -> Dict[int, int]:
        """
        Predict POI locations with sequential and relative time gap constraints.

        Args:
            df: DataFrame with sensor data (must include 'Relative_time' column for gap constraints)
            threshold: Default probability threshold for POI detection
            adaptive_thresholds: Optional dict of POI-specific thresholds
            use_nms: Whether to use non-maximum suppression
            nms_window: Window size for non-maximum suppression
            enforce_sequential: Enforce sequential ordering of POIs
            enforce_relative_gaps: Apply relative time gap constraints

        Returns:
            Dictionary mapping POI number to predicted index
        """
        # Generate features
        features_df = DP.gen_features(df)
        features = features_df.values

        # Create windows
        windows = []
        window_positions = []

        for i in range(0, len(features) - self.window_size, self.stride):
            window = features[i:i + self.window_size]
            windows.append(window)
            window_positions.append(i + self.window_size // 2)

        windows = np.array(windows)

        # Normalize
        windows_reshaped = windows.reshape(-1, windows.shape[-1])
        windows_reshaped = self.scaler.transform(windows_reshaped)
        windows = windows_reshaped.reshape(windows.shape)

        # Predict
        predictions = self.model.predict(windows, verbose=0)

        # Set adaptive thresholds
        if adaptive_thresholds is None:
            adaptive_thresholds = {
                1: 0.5,   # POI-1
                2: 0.5,   # POI-2
                4: 0.7,   # POI-4
                5: 0.6,   # POI-5
                6: 0.65   # POI-6
            }

        # Initial POI detection
        poi_candidates = self._find_poi_candidates(
            predictions, window_positions, adaptive_thresholds, use_nms, nms_window)

        # Apply constraints
        if enforce_sequential:
            poi_candidates = self._enforce_sequential_constraints(
                poi_candidates, predictions, window_positions)

        if enforce_relative_gaps and 'Relative_time' in df.columns:
            poi_candidates = self._enforce_relative_gap_constraints(
                poi_candidates, df)

        return poi_candidates

    def _non_maximum_suppression(self, probs: np.ndarray, mask: np.ndarray,
                                 window: int) -> np.ndarray:
        """Apply non-maximum suppression to reduce duplicate detections."""
        peaks = []
        probs_copy = probs.copy()

        while True:
            valid_indices = np.where(mask & (probs_copy > 0))[0]
            if len(valid_indices) == 0:
                break

            max_idx = valid_indices[np.argmax(probs_copy[valid_indices])]
            peaks.append(max_idx)

            start = max(0, max_idx - window // 2)
            end = min(len(probs_copy), max_idx + window // 2 + 1)
            probs_copy[start:end] = 0

        return np.array(peaks)

    def _find_peaks(self, probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Find local maxima in probability array."""
        peaks = []
        for i in range(1, len(probs) - 1):
            if mask[i] and probs[i] >= probs[i-1] and probs[i] >= probs[i+1]:
                peaks.append(i)

        if len(mask) > 0:
            if mask[0] and (len(probs) == 1 or probs[0] >= probs[1]):
                peaks.insert(0, 0)
            if mask[-1] and probs[-1] >= probs[-2]:
                peaks.append(len(probs) - 1)

        return np.array(peaks) if peaks else np.array([])

    def _find_poi_candidates(self, predictions: np.ndarray, window_positions: List[int],
                             thresholds: Dict[int, float], use_nms: bool,
                             nms_window: int) -> Dict[int, int]:
        """Find initial POI candidates from predictions."""
        poi_locations = {}
        poi_indices = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}

        for poi_num, pred_idx in poi_indices.items():
            poi_probs = predictions[:, pred_idx]
            poi_threshold = thresholds.get(poi_num, 0.5)

            above_threshold = poi_probs > poi_threshold
            if np.any(above_threshold):
                if use_nms:
                    peak_indices = self._non_maximum_suppression(
                        poi_probs, above_threshold, nms_window)
                else:
                    peak_indices = self._find_peaks(poi_probs, above_threshold)

                if len(peak_indices) > 0:
                    best_idx = peak_indices[np.argmax(poi_probs[peak_indices])]
                    poi_locations[poi_num] = window_positions[best_idx]

        return poi_locations

    def _enforce_sequential_constraints(self, poi_candidates: Dict[int, int],
                                        predictions: np.ndarray,
                                        window_positions: List[int]) -> Dict[int, int]:
        """
        Enforce sequential ordering: POI4 requires POI1&2, POI5 requires POI4, POI6 requires POI5.
        """
        validated_pois = {}

        # POI1 and POI2 can exist independently
        if 1 in poi_candidates:
            validated_pois[1] = poi_candidates[1]
        if 2 in poi_candidates:
            validated_pois[2] = poi_candidates[2]

        # POI4 requires both POI1 and POI2
        if 4 in poi_candidates:
            if 1 in validated_pois and 2 in validated_pois:
                # Also check that POI4 comes after POI2
                if poi_candidates[4] > validated_pois[2]:
                    validated_pois[4] = poi_candidates[4]

        # POI5 requires POI4
        if 5 in poi_candidates:
            if 4 in validated_pois:
                # Check that POI5 comes after POI4
                if poi_candidates[5] > validated_pois[4]:
                    validated_pois[5] = poi_candidates[5]

        # POI6 requires POI5
        if 6 in poi_candidates:
            if 5 in validated_pois:
                # Check that POI6 comes after POI5
                if poi_candidates[6] > validated_pois[5]:
                    validated_pois[6] = poi_candidates[6]

        return validated_pois

    def _enforce_relative_gap_constraints(self, poi_candidates: Dict[int, int],
                                          df: pd.DataFrame) -> Dict[int, int]:
        """
        Apply relative time gap constraints:
        - Gap between POI1 and POI2 must be the smallest
        - Gap between POI4 and POI2 >= gap between POI1 and POI2
        - Gap between POI5 and POI4 >= gap between POI4 and POI2 (if POI4 and POI5 exist)
        - Gap between POI6 and POI5 >= gap between POI5 and POI4 (if POI5 and POI6 exist)

        Args:
            poi_candidates: Dictionary of POI candidates
            df: DataFrame with 'Relative_time' column

        Returns:
            Refined POI candidates after applying gap constraints
        """
        refined_pois = poi_candidates.copy()

        # Check if Relative_time exists
        if 'Relative_time' not in df.columns:
            print(
                "Warning: 'Relative_time' column not found. Skipping relative gap constraints.")
            return refined_pois

        # Helper function to get time at a given index
        def get_time(idx):
            if idx < len(df):
                return df['Relative_time'].iloc[idx]
            return None

        # Calculate time gaps
        time_gaps = {}

        # Gap between POI1 and POI2 (should be smallest)
        if 1 in refined_pois and 2 in refined_pois:
            time1 = get_time(refined_pois[1])
            time2 = get_time(refined_pois[2])
            if time1 is not None and time2 is not None:
                gap_1_2 = abs(time2 - time1)
                time_gaps['1-2'] = gap_1_2

                # Gap between POI4 and POI2
                if 4 in refined_pois:
                    time4 = get_time(refined_pois[4])
                    if time4 is not None:
                        gap_2_4 = abs(time4 - time2)
                        time_gaps['2-4'] = gap_2_4

                        # Constraint: gap(2-4) >= gap(1-2)
                        if gap_2_4 < gap_1_2:
                            print(
                                f"Removing POI4: gap(2-4)={gap_2_4:.3f} < gap(1-2)={gap_1_2:.3f}")
                            del refined_pois[4]
                        else:
                            # Gap between POI5 and POI4
                            if 5 in refined_pois:
                                time5 = get_time(refined_pois[5])
                                if time5 is not None:
                                    gap_4_5 = abs(time5 - time4)
                                    time_gaps['4-5'] = gap_4_5

                                    # Constraint: gap(4-5) >= gap(2-4)
                                    if gap_4_5 < gap_2_4:
                                        print(
                                            f"Removing POI5: gap(4-5)={gap_4_5:.3f} < gap(2-4)={gap_2_4:.3f}")
                                        del refined_pois[5]
                                    else:
                                        # Gap between POI6 and POI5
                                        if 6 in refined_pois:
                                            time6 = get_time(refined_pois[6])
                                            if time6 is not None:
                                                gap_5_6 = abs(time6 - time5)
                                                time_gaps['5-6'] = gap_5_6

                                                # Constraint: gap(5-6) >= gap(4-5)
                                                if gap_5_6 < gap_4_5:
                                                    print(
                                                        f"Removing POI6: gap(5-6)={gap_5_6:.3f} < gap(4-5)={gap_4_5:.3f}")
                                                    del refined_pois[6]

        # If POI5 exists but POI4 was removed, remove POI5
        if 5 in refined_pois and 4 not in refined_pois:
            del refined_pois[5]

        # If POI6 exists but POI5 was removed, remove POI6
        if 6 in refined_pois and 5 not in refined_pois:
            del refined_pois[6]

        return refined_pois

    def evaluate_on_test_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, any]:
        """Evaluate multi-label model performance on test data."""
        from sklearn.metrics import classification_report, multilabel_confusion_matrix

        # Normalize test data
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_reshaped = self.scaler.transform(X_test_reshaped)
        X_test = X_test_reshaped.reshape(X_test.shape)

        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate per-label metrics
        label_names = ['Non-POI', 'POI-1', 'POI-2', 'POI-4', 'POI-5', 'POI-6']

        # Multi-label confusion matrices
        mcm = multilabel_confusion_matrix(y_test, y_pred)

        metrics = {
            'predictions_proba': y_pred_proba,
            'predictions': y_pred,
            'multi_confusion_matrix': mcm
        }

        # Print summary
        print("\n=== Enhanced Multi-Label Model Evaluation Results ===")

        # Per-label performance
        print("\nPer-Label Performance:")
        for i, name in enumerate(label_names):
            tn, fp, fn, tp = mcm[i].ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision +
                                             recall) if (precision + recall) > 0 else 0

            print(f"  {name}:")
            print(f"    Precision: {precision:.3f}")
            print(f"    Recall: {recall:.3f}")
            print(f"    F1-Score: {f1:.3f}")
            print(f"    True Positives: {tp}, False Positives: {fp}")

            metrics[f'{name}_precision'] = precision
            metrics[f'{name}_recall'] = recall
            metrics[f'{name}_f1'] = f1

        # Overall metrics
        print("\nOverall Metrics:")
        exact_match = np.mean(np.all(y_pred == y_test, axis=1))
        print(f"  Exact Match Accuracy: {exact_match:.3f}")

        # POI detection performance
        y_test_has_poi = np.any(y_test[:, 1:], axis=1)
        y_pred_has_poi = np.any(y_pred[:, 1:], axis=1)
        poi_accuracy = np.mean(y_test_has_poi == y_pred_has_poi)
        print(f"  POI Detection Accuracy: {poi_accuracy:.3f}")

        metrics['exact_match_accuracy'] = exact_match
        metrics['poi_detection_accuracy'] = poi_accuracy

        return metrics

    def visualize_predictions(self, df: pd.DataFrame, save_path: str = None,
                              show_constraints: bool = True):
        """
        Visualize multi-label POI predictions with constraint enforcement indicators.
        """
        import matplotlib.pyplot as plt

        # Get predictions with and without constraints
        predicted_pois_full = self.predict_pois(
            df, enforce_sequential=True, enforce_relative_gaps=True)
        predicted_pois_no_constraints = self.predict_pois(
            df, enforce_sequential=False, enforce_relative_gaps=False)

        features_df = DP.gen_features(df)

        # Determine x-axis: use Relative_time if available, else use index
        use_time = 'Relative_time' in df.columns
        x_values = df['Relative_time'].values if use_time else np.arange(
            len(df))
        x_label = 'Relative Time' if use_time else 'Sample Index'

        fig, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=True)

        # Dissipation
        if 'Dissipation' in df.columns:
            axes[0].plot(x_values, df['Dissipation'].values,
                         'b-', alpha=0.7, lw=0.5)
            axes[0].set_ylabel('Dissipation')
            axes[0].grid(True, alpha=0.3)

        # Resonance Frequency
        if 'Resonance_Frequency' in df.columns:
            axes[1].plot(x_values, df['Resonance_Frequency'].values,
                         'g-', alpha=0.7, lw=0.5)
            axes[1].set_ylabel('Resonance Frequency')
            axes[1].grid(True, alpha=0.3)

        # POI probabilities
        features = features_df.values
        windows, window_positions = [], []
        window_x_values = []  # Store x-axis values for windows

        for i in range(0, len(features) - self.window_size, self.stride):
            windows.append(features[i:i + self.window_size])
            pos = i + self.window_size // 2
            window_positions.append(pos)
            if use_time and pos < len(df):
                window_x_values.append(df['Relative_time'].iloc[pos])
            else:
                window_x_values.append(pos)

        windows = np.array(windows)
        ws = windows.shape[:2]
        ss = windows.shape[-1]
        windows = self.scaler.transform(
            windows.reshape(-1, ss)).reshape(ws + (ss,))
        predictions = self.model.predict(windows, verbose=0)

        poi_colors = {1: 'red', 2: 'orange',
                      4: 'yellow', 5: 'green', 6: 'blue'}
        poi_indices = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
        half_win = 128 // 2
        for poi_num, idx in predicted_pois_full.items():
            # compute start/end indices
            start_idx = max(0, idx - half_win)
            end_idx = min(len(df) - 1, idx + half_win)
            # convert to x-values
            if use_time:
                x_start = df['Relative_time'].iloc[start_idx]
                x_end = df['Relative_time'].iloc[end_idx]
            else:
                x_start, x_end = start_idx, end_idx

            # shade the region on all axes
            for ax in axes:
                ax.axvspan(x_start, x_end,
                           color=poi_colors.get(poi_num, 'gray'),
                           alpha=0.1,
                           label=f'128-pt window POI-{poi_num}' if poi_num == list(predicted_pois_full)[0] else "")
        # optionally add a legend entry for the shaded window
        axes[0].legend(loc='upper left')

        for poi_num, idx in poi_indices.items():
            axes[2].plot(window_x_values, predictions[:, idx],
                         color=poi_colors[poi_num], alpha=0.6,
                         label=f'POI-{poi_num}', lw=1)

        axes[2].set_ylabel('POI Probabilities')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0.5, color='black', linestyle=':', alpha=0.5)

        # Combined POI confidence
        any_conf = 1 - predictions[:, 0]
        axes[3].plot(window_x_values, any_conf, 'r-', alpha=0.7, lw=1)
        axes[3].set_ylabel('Any POI Confidence')
        axes[3].grid(True, alpha=0.3)

        # Constraint enforcement visualization
        if show_constraints:
            # Convert POI positions to x-axis values for plotting
            def get_x_value(idx):
                if use_time and idx < len(df):
                    return df['Relative_time'].iloc[idx]
                return idx

            # Show rejected candidates
            for poi_num in poi_indices.keys():
                if poi_num in predicted_pois_no_constraints and poi_num not in predicted_pois_full:
                    x_pos = get_x_value(predicted_pois_no_constraints[poi_num])
                    axes[4].scatter(x_pos, poi_num, color='red', s=100, marker='x',
                                    label='Rejected' if poi_num == 1 else '')

            # Show accepted POIs
            for poi_num, idx in predicted_pois_full.items():
                x_pos = get_x_value(idx)
                axes[4].scatter(x_pos, poi_num, color=poi_colors[poi_num], s=100, marker='o',
                                label='Accepted' if poi_num == 1 else '')

            axes[4].set_ylabel('POI Constraints')
            axes[4].set_yticks([1, 2, 4, 5, 6])
            axes[4].set_yticklabels(
                ['POI-1', 'POI-2', 'POI-4', 'POI-5', 'POI-6'])
            axes[4].grid(True, alpha=0.3)
            if poi_num == 1:  # Only add legend once
                axes[4].legend(loc='upper right')

        # Draw vertical lines for final predictions
        for poi_num, idx in predicted_pois_full.items():
            x_pos = get_x_value(idx)
            for ax in axes:
                ax.axvline(x_pos, color=poi_colors.get(poi_num, 'black'),
                           linestyle='--', alpha=0.8)

        # Add relative gap indicators if using Relative_time
        if use_time and len(predicted_pois_full) > 1:
            # Sort POIs by their position
            sorted_pois = sorted(
                predicted_pois_full.items(), key=lambda x: x[1])

            # Draw gap indicators
            for i in range(len(sorted_pois) - 1):
                poi1_num, poi1_idx = sorted_pois[i]
                poi2_num, poi2_idx = sorted_pois[i + 1]

                time1 = df['Relative_time'].iloc[poi1_idx]
                time2 = df['Relative_time'].iloc[poi2_idx]
                gap = time2 - time1

                # Draw horizontal line showing gap
                mid_time = (time1 + time2) / 2
                axes[4].annotate('', xy=(time2, poi2_num), xytext=(time1, poi1_num),
                                 arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.5))
                axes[4].text(mid_time, (poi1_num + poi2_num) / 2, f'{gap:.2f}',
                             ha='center', va='center', fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        axes[0].set_title(
            'POI Detection with Sequential & Relative Gap Constraints')
        axes[-1].set_xlabel(x_label)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        # Print gap analysis if using Relative_time
        if use_time and len(predicted_pois_full) > 1:
            print("\n=== Relative Time Gap Analysis ===")
            sorted_pois = sorted(
                predicted_pois_full.items(), key=lambda x: x[1])

            gaps = {}
            for i in range(len(sorted_pois) - 1):
                poi1_num, poi1_idx = sorted_pois[i]
                poi2_num, poi2_idx = sorted_pois[i + 1]

                time1 = df['Relative_time'].iloc[poi1_idx]
                time2 = df['Relative_time'].iloc[poi2_idx]
                gap = time2 - time1

                gap_name = f"POI{poi1_num}-POI{poi2_num}"
                gaps[gap_name] = gap
                print(f"{gap_name}: {gap:.3f}")

            # Check constraints
            print("\nConstraint Checks:")
            if 'POI1-POI2' in gaps:
                print(
                    f"POI1-POI2 gap: {gaps['POI1-POI2']:.3f} (should be smallest)")

                if 'POI2-POI4' in gaps:
                    if gaps['POI2-POI4'] >= gaps['POI1-POI2']:
                        print(
                            f"✓ POI2-POI4 ({gaps['POI2-POI4']:.3f}) >= POI1-POI2 ({gaps['POI1-POI2']:.3f})")
                    else:
                        print(
                            f"✗ POI2-POI4 ({gaps['POI2-POI4']:.3f}) < POI1-POI2 ({gaps['POI1-POI2']:.3f})")

                if 'POI4-POI5' in gaps and 'POI2-POI4' in gaps:
                    if gaps['POI4-POI5'] >= gaps['POI2-POI4']:
                        print(
                            f"✓ POI4-POI5 ({gaps['POI4-POI5']:.3f}) >= POI2-POI4 ({gaps['POI2-POI4']:.3f})")
                    else:
                        print(
                            f"✗ POI4-POI5 ({gaps['POI4-POI5']:.3f}) < POI2-POI4 ({gaps['POI2-POI4']:.3f})")

                if 'POI5-POI6' in gaps and 'POI4-POI5' in gaps:
                    if gaps['POI5-POI6'] >= gaps['POI4-POI5']:
                        print(
                            f"✓ POI5-POI6 ({gaps['POI5-POI6']:.3f}) >= POI4-POI5 ({gaps['POI4-POI5']:.3f})")
                    else:
                        print(
                            f"✗ POI5-POI6 ({gaps['POI5-POI6']:.3f}) < POI4-POI5 ({gaps['POI4-POI5']:.3f})")

        return predicted_pois_full

    def save(self,
             model_path: str = 'poi_model',
             scaler_path: str = 'scaler.joblib') -> None:
        """
        Save the trained Keras model and the fitted StandardScaler to disk.

        Args:
            model_path:    Path (file or directory) to save the Keras model. 
                           If extension is “.h5” it will save to HDF5; 
                           otherwise a TensorFlow SavedModel directory is created.
            scaler_path:   File path (should end in .joblib) to dump the scaler.
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")

        # 1) Save the TensorFlow/Keras model
        #    - .h5 if you want a single file, or
        #    - a folder if you want the SavedModel format.
        self.model.save(model_path)

        # 2) Save the scaler
        import joblib
        joblib.dump(self.scaler, scaler_path)

        print(
            f"Model saved to {model_path!r}\n Scaler saved to {scaler_path!r}")


# Enhanced example usage
if __name__ == "__main__":
    # Initialize system
    poi_system = POIDetectionSystem(window_size=128, stride=16, tolerance=64)

    # Load and prepare data
    print("Loading data...")
    X, y = poi_system.load_and_prepare_data(
        "content/static/train", num_datasets=800)
    tuner = poi_system.tune(X, y)
    # After tuning, train best model
    history = poi_system.train(X, y, epochs=100, batch_size=32)
    # Evaluate
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    metrics = poi_system.evaluate_on_test_data(X_test, y_test)

    # Test on new data
    new_content = DP.load_content(data_dir='content/static/test')
    poi_system.save(model_path="v4_model.h5", scaler_path="v4_scaler")
    for d, p in new_content:
        df_new = pd.read_csv(d)

        # Randomly trim the data to simulate partial runs
        trim_loc = random.randint(min(2000, len(df_new) - 1), len(df_new))
        df_new = df_new.iloc[:trim_loc]

        # Predict with new relative gap constraints
        predicted_pois = poi_system.predict_pois(
            df_new, enforce_relative_gaps=True)
        print(f"\nPredicted POI locations: {predicted_pois}")

        # Visualize with gap analysis
        poi_system.visualize_predictions(
            df_new, save_path="poi_gap_constraints_predictions.png")
