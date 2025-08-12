import matplotlib.pyplot as plt
from v4_dp import DP
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EnhancedPOIPredictor:
    """
    Enhanced POI Predictor that uses a Localizer model to refine predictions.
    First uses the main model to find windows containing POIs, then uses
    the localizer to pinpoint exact locations within those windows.
    """

    def __init__(self,
                 model_path: str,
                 scaler_path: str,
                 localizer_model_path: str = None,
                 localizer_scaler_path: str = None,
                 window_size: int = 128,
                 stride: int = 16,
                 tolerance: int = 64):
        """
        Initialize the Enhanced POI Predictor with main model and localizer.

        Args:
            model_path: Path to the main .h5 model file
            scaler_path: Path to the main .pkl scaler file
            localizer_model_path: Path to the localizer .h5 model file
            localizer_scaler_path: Path to the localizer .pkl scaler file
            window_size: Size of the sliding window for feature extraction
            stride: Step size for sliding window
            tolerance: Acceptable distance (in points) for POI detection
        """
        self.window_size = window_size
        self.stride = stride
        self.tolerance = tolerance

        # Load the main model and scaler
        print(f"Loading main model from {model_path}...")
        self.model = keras.models.load_model(model_path)

        print(f"Loading main scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)

        # Load the localizer model if provided
        self.localizer_model = None
        self.localizer_scaler = None
        if localizer_model_path and localizer_scaler_path:
            print(f"Loading localizer model from {localizer_model_path}...")
            self.localizer_model = keras.models.load_model(
                localizer_model_path, compile=False)

            print(f"Loading localizer scaler from {localizer_scaler_path}...")
            self.localizer_scaler = joblib.load(localizer_scaler_path)

            print("Localizer loaded successfully!")

        # POI configuration
        self.poi_names = {
            0: 'Non-POI',
            1: 'POI-1',
            2: 'POI-2',
            4: 'POI-4',
            5: 'POI-5',
            6: 'POI-6'
        }

        # Mapping from POI number to prediction index
        self.poi_indices = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}

        print("Enhanced Predictor initialized successfully!")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from raw data."""
        return DP.gen_features(df)

    def create_advanced_features_for_localizer(self, X):
        """
        Create advanced signal processing features for the localizer.
        This mirrors the feature creation in the Localizer class.
        """
        from scipy import signal

        batch_size, window_size, n_features = X.shape
        advanced_features = []

        for i in range(batch_size):
            window_features = []

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
                    peak_indicator[peaks] = properties.get(
                        'prominences', np.ones(len(peaks)))

                # 3. Valley detection
                valleys, _ = signal.find_peaks(-signal_1d,
                                               prominence=0.1, distance=3)
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
                    signal_1d - local_mean,
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

    def refine_position_with_localizer(self, window_data: np.ndarray,
                                       window_start_idx: int,
                                       confidence_threshold: float = 0.7) -> Tuple[int, float, bool]:
        """
        Use the localizer model to refine the POI position within a window.

        Args:
            window_data: The normalized window data (1, window_size, n_features)
            window_start_idx: The starting index of this window in the original data
            confidence_threshold: Minimum confidence for accepting the refined position

        Returns:
            Tuple of (refined_global_index, confidence, is_high_confidence)
        """
        if self.localizer_model is None:
            # If no localizer, return the center of the window
            return window_start_idx + self.window_size // 2, 1.0, True

        # Prepare the window for the localizer
        # First apply localizer's scaler
        window_reshaped = window_data.reshape(-1, window_data.shape[-1])
        window_normalized = self.localizer_scaler.transform(
            window_reshaped).astype(np.float32)
        window_normalized = window_normalized.reshape(window_data.shape)

        # Create advanced features
        window_enhanced = self.create_advanced_features_for_localizer(
            window_normalized)

        # Get localizer prediction
        position_pred, confidence_pred = self.localizer_model.predict(
            window_enhanced, verbose=0)
        # Get the predicted position within the window
        local_position = np.argmax(position_pred[0])
        confidence = float(confidence_pred[0, 0])

        # Convert to global index
        global_position = window_start_idx + local_position

        # Check if confidence meets threshold
        is_high_confidence = confidence > confidence_threshold

        return global_position, confidence, is_high_confidence

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[int], List[float], np.ndarray]:
        """
        Preprocess data for prediction, returning both normalized and raw windows.

        Returns:
            Tuple of (normalized_windows, window_positions, window_times, raw_windows)
        """
        # Generate features
        features_df = self.generate_features(df)
        features = features_df.values

        # Create sliding windows
        windows = []
        raw_windows = []
        window_positions = []
        window_times = []

        for i in range(0, len(features) - self.window_size, self.stride):
            window = features[i:i + self.window_size]
            windows.append(window)
            raw_windows.append(window.copy())  # Keep raw for localizer

            # Store start position of window (not center)
            window_positions.append(i)

            # Store time if available
            if 'Relative_time' in df.columns and i < len(df):
                window_times.append(
                    df['Relative_time'].iloc[i + self.window_size // 2])
            else:
                window_times.append(i + self.window_size // 2)

        windows = np.array(windows)
        raw_windows = np.array(raw_windows)

        # Normalize windows for main model
        original_shape = windows.shape
        windows_flat = windows.reshape(-1, windows.shape[-1])
        windows_normalized = self.scaler.transform(windows_flat)
        windows_normalized = windows_normalized.reshape(original_shape)

        return windows_normalized, window_positions, window_times, raw_windows

    def predict_best_with_refinement(self,
                                     df: pd.DataFrame,
                                     use_nms: bool = True,
                                     nms_window: int = 30,
                                     adaptive_thresholds: Optional[Dict[int,
                                                                        float]] = None,
                                     use_localizer: bool = True) -> Dict[int, Dict]:
        """
        Predict the single best location for each POI with localizer refinement.

        Args:
            df: DataFrame with sensor data
            use_nms: Whether to use non-maximum suppression
            nms_window: Window size for NMS
            adaptive_thresholds: Optional POI-specific thresholds
            use_localizer: Whether to use the localizer for refinement

        Returns:
            Dictionary mapping POI numbers to refined prediction info
        """
        # Default adaptive thresholds
        if adaptive_thresholds is None:
            adaptive_thresholds = {
                1: 0.5,
                2: 0.5,
                4: 0.7,
                5: 0.6,
                6: 0.65
            }

        # Preprocess data (get both normalized and raw windows)
        windows, positions, times, raw_windows = self.preprocess_data(df)

        # Get main model predictions
        predictions = self.model.predict(windows, verbose=0)

        # Find best prediction for each POI
        best_predictions = {}

        for poi_num, pred_idx in self.poi_indices.items():
            poi_probs = predictions[:, pred_idx]
            threshold = adaptive_thresholds.get(poi_num, 0.5)

            # Apply threshold
            above_threshold = poi_probs >= threshold

            if np.any(above_threshold):
                if use_nms:
                    # Apply non-maximum suppression
                    peak_indices = self._non_maximum_suppression(
                        poi_probs, above_threshold, nms_window
                    )
                else:
                    # Find all peaks above threshold
                    peak_indices = np.where(above_threshold)[0]

                if len(peak_indices) > 0:
                    # Select peak with highest confidence
                    best_idx = peak_indices[np.argmax(poi_probs[peak_indices])]

                    # Get the window for refinement
                    best_window = raw_windows[best_idx:best_idx+1]
                    window_start = positions[best_idx]

                    if use_localizer and self.localizer_model is not None:
                        # Use localizer to refine position within the window
                        refined_position, localizer_confidence, is_high_conf = \
                            self.refine_position_with_localizer(
                                best_window, window_start)

                        best_predictions[poi_num] = {
                            'data_index': refined_position,
                            'coarse_index': window_start + self.window_size // 2,
                            'window_start': window_start,
                            'window_end': window_start + self.window_size,
                            'main_confidence': float(poi_probs[best_idx]),
                            'localizer_confidence': localizer_confidence,
                            'combined_confidence': float(poi_probs[best_idx]) * localizer_confidence,
                            'is_refined': True,
                            'is_high_confidence': is_high_conf,
                            'time': times[best_idx] if times else None,
                            'window_index': best_idx
                        }
                    else:
                        # No localizer, use center of window
                        best_predictions[poi_num] = {
                            'data_index': window_start + self.window_size // 2,
                            'coarse_index': window_start + self.window_size // 2,
                            'window_start': window_start,
                            'window_end': window_start + self.window_size,
                            'main_confidence': float(poi_probs[best_idx]),
                            'localizer_confidence': None,
                            'combined_confidence': float(poi_probs[best_idx]),
                            'is_refined': False,
                            'is_high_confidence': True,
                            'time': times[best_idx] if times else None,
                            'window_index': best_idx
                        }

        return best_predictions

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

            # Suppress nearby values
            start = max(0, max_idx - window // 2)
            end = min(len(probs_copy), max_idx + window // 2 + 1)
            probs_copy[start:end] = 0

        return np.array(peaks)

    def get_enhanced_prediction_summary(self,
                                        df: pd.DataFrame,
                                        apply_constraints: bool = True,
                                        use_localizer: bool = True) -> str:
        """
        Get a formatted summary of predictions with localizer refinement details.

        Args:
            df: DataFrame with sensor data
            apply_constraints: Whether to apply constraints
            use_localizer: Whether to use localizer refinement

        Returns:
            Formatted string summary with refinement details
        """
        # Get best predictions with refinement
        predictions = self.predict_best_with_refinement(
            df, use_localizer=use_localizer)

        if apply_constraints:
            predictions = self.apply_constraints(predictions, df)

        # Build summary
        summary = []
        summary.append("=" * 70)
        summary.append(
            "ENHANCED POI PREDICTION SUMMARY (WITH LOCALIZER REFINEMENT)")
        summary.append("=" * 70)

        for poi_num in sorted(self.poi_indices.keys()):
            summary.append(
                f"\n{self.poi_names.get(poi_num, f'POI-{poi_num}')}:")

            if poi_num in predictions:
                pred = predictions[poi_num]
                summary.append(f"  Best Prediction:")
                summary.append(f"    Refined Position: {pred['data_index']}")

                if pred['is_refined']:
                    summary.append(
                        f"    Coarse Position:  {pred['coarse_index']} (before refinement)")
                    summary.append(
                        f"    Refinement Shift: {pred['data_index'] - pred['coarse_index']:+d} points")

                summary.append(
                    f"    Window Range: [{pred['window_start']}, {pred['window_end']})")
                summary.append(
                    f"    Main Model Confidence: {pred['main_confidence']:.3f}")

                if pred['localizer_confidence'] is not None:
                    summary.append(
                        f"    Localizer Confidence:  {pred['localizer_confidence']:.3f}")
                    summary.append(
                        f"    Combined Confidence:   {pred['combined_confidence']:.3f}")
                    summary.append(
                        f"    High Confidence: {'Yes' if pred['is_high_confidence'] else 'No'}")

                if pred.get('time') is not None:
                    summary.append(f"    Time: {pred['time']:.3f}")
            else:
                summary.append(f"  No detection above threshold")

        summary.append("\n" + "=" * 70)

        # Add refinement statistics
        if use_localizer and self.localizer_model is not None:
            summary.append("\nREFINEMENT STATISTICS:")
            summary.append("-" * 40)

            refinement_shifts = []
            for poi_num, pred in predictions.items():
                if pred['is_refined']:
                    shift = pred['data_index'] - pred['coarse_index']
                    refinement_shifts.append(shift)
                    summary.append(
                        f"  POI-{poi_num}: shifted by {shift:+d} points")

            if refinement_shifts:
                summary.append(
                    f"\n  Average shift: {np.mean(refinement_shifts):+.1f} points")
                summary.append(
                    f"  Max shift: {max(refinement_shifts):+d} points")
                summary.append(
                    f"  Min shift: {min(refinement_shifts):+d} points")

        return "\n".join(summary)

    def apply_constraints(self, predictions: Dict[int, Dict], df: pd.DataFrame,
                          enforce_sequential: bool = True,
                          enforce_gaps: bool = True) -> Dict[int, Dict]:
        """Apply sequential and temporal constraints to predictions."""
        validated = predictions.copy()

        if enforce_sequential:
            validated = self._enforce_sequential_constraints(validated)

        if enforce_gaps and 'Relative_time' in df.columns:
            validated = self._enforce_gap_constraints(validated, df)

        return validated

    def _enforce_sequential_constraints(self, predictions: Dict[int, Dict]) -> Dict[int, Dict]:
        """Enforce POI sequential ordering rules."""
        validated = {}

        # POI1 and POI2 can exist independently
        if 1 in predictions:
            validated[1] = predictions[1]
        if 2 in predictions:
            validated[2] = predictions[2]

        # POI4 requires both POI1 and POI2, and must come after POI2
        if 4 in predictions:
            if 1 in validated and 2 in validated:
                if predictions[4]['data_index'] > validated[2]['data_index']:
                    validated[4] = predictions[4]

        # POI5 requires POI4 and must come after it
        if 5 in predictions:
            if 4 in validated:
                if predictions[5]['data_index'] > validated[4]['data_index']:
                    validated[5] = predictions[5]

        # POI6 requires POI5 and must come after it
        if 6 in predictions:
            if 5 in validated:
                if predictions[6]['data_index'] > validated[5]['data_index']:
                    validated[6] = predictions[6]

        return validated

    def _enforce_gap_constraints(self, predictions: Dict[int, Dict],
                                 df: pd.DataFrame) -> Dict[int, Dict]:
        """Apply relative time gap constraints."""
        refined = predictions.copy()

        if 'Relative_time' not in df.columns:
            return refined

        # Helper to get time at index
        def get_time(idx):
            if idx < len(df):
                return df['Relative_time'].iloc[idx]
            return None

        # Check gap constraints
        if 1 in refined and 2 in refined:
            time1 = get_time(refined[1]['data_index'])
            time2 = get_time(refined[2]['data_index'])

            if time1 is not None and time2 is not None:
                gap_1_2 = abs(time2 - time1)

                # Check POI4
                if 4 in refined:
                    time4 = get_time(refined[4]['data_index'])
                    if time4 is not None:
                        gap_2_4 = abs(time4 - time2)
                        if gap_2_4 < gap_1_2:
                            del refined[4]
                        else:
                            # Check POI5
                            if 5 in refined:
                                time5 = get_time(refined[5]['data_index'])
                                if time5 is not None:
                                    gap_4_5 = abs(time5 - time4)
                                    if gap_4_5 < gap_2_4:
                                        del refined[5]
                                    else:
                                        # Check POI6
                                        if 6 in refined:
                                            time6 = get_time(
                                                refined[6]['data_index'])
                                            if time6 is not None:
                                                gap_5_6 = abs(time6 - time5)
                                                if gap_5_6 < gap_4_5:
                                                    del refined[6]

        # Remove dependent POIs if parent was removed
        if 5 in refined and 4 not in refined:
            del refined[5]
        if 6 in refined and 5 not in refined:
            del refined[6]

        return refined


# Example usage
if __name__ == "__main__":
    # Initialize enhanced predictor with both models
    predictor = EnhancedPOIPredictor(
        model_path=r"QModel\src\models\static_v4\v4_model.h5",
        scaler_path=r"QModel\src\models\static_v4\v4_scaler.joblib",
        localizer_model_path=r"QModel\src\models\static_v4\v4_localizer.h5",
        localizer_scaler_path=r"QModel\src\models\static_v4\v4_localizer_scaler.joblib",
        window_size=128,
        stride=16,
        tolerance=64
    )

    content = DP.load_content('content/static/valid')

    for d, p in content:
        df_test = pd.read_csv(d)
        import random
        max_rows = random.randint(0, len(df_test))
        df_test = df_test.iloc[:max_rows]

        # Read POI data and truncate to match df_test length
        poi_test = pd.read_csv(p, header=None).values
        poi_test = poi_test[poi_test <= max_rows]

        # Get enhanced predictions with localizer refinement
        print("\n" + predictor.get_enhanced_prediction_summary(df_test,
                                                               apply_constraints=True,
                                                               use_localizer=True))

        # Plot results with refinement visualization
        predictions = predictor.predict_best_with_refinement(
            df_test, use_localizer=False)

        plt.figure(figsize=(14, 8))
        plt.plot(df_test['Relative_time'], df_test['Dissipation'],
                 label='Dissipation Curve', alpha=0.7)

        # Plot predictions with refinement info
        for poi_num, pred in predictions.items():
            r_time = df_test.loc[pred['data_index'], 'Relative_time']
            diss_val = df_test.loc[pred['data_index'], 'Dissipation']

            # Plot refined position
            plt.scatter(r_time, diss_val,
                        s=100, marker='o',
                        label=f'POI-{poi_num} (Refined)',
                        zorder=5)

            # If refined, also show original coarse position
            if pred['is_refined'] and pred['coarse_index'] != pred['data_index']:
                coarse_time = df_test.loc[pred['coarse_index'],
                                          'Relative_time']
                coarse_diss = df_test.loc[pred['coarse_index'], 'Dissipation']
                plt.scatter(coarse_time, coarse_diss,
                            s=50, marker='x', alpha=0.5,
                            label=f'POI-{poi_num} (Coarse)')

                # Draw arrow from coarse to refined
                plt.annotate('', xy=(r_time, diss_val),
                             xytext=(coarse_time, coarse_diss),
                             arrowprops=dict(arrowstyle='->', alpha=0.3, lw=1))

            # Show window boundaries
            window_start_time = df_test.loc[pred['window_start'],
                                            'Relative_time']
            window_end_time = df_test.loc[min(
                pred['window_end']-1, len(df_test)-1), 'Relative_time']
            plt.axvspan(window_start_time, window_end_time,
                        alpha=0.1, label=f'POI-{poi_num} Window')

        # Plot ground truth
        r_time = df_test["Relative_time"].values
        for idx in poi_test:
            plt.axvline(r_time[idx], color='red', alpha=0.3, linestyle='--')

        plt.xlabel('Relative_time')
        plt.ylabel('Dissipation')
        plt.title('POI Predictions with Localizer Refinement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
