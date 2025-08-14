import matplotlib.pyplot as plt
from v4_dp import DP
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class EnhancedLocalizer:
    """
    Fine-tuning model for precise POI localization within windows.
    """

    def __init__(self):
        self._classifier = None
        self._regressor = None
        self._confidence_model = None
        self._scaler = None
        self._feature_importance = None

    def load(self, filepath='v4_localizer'):
        """Load all model components."""
        self._classifier = xgb.Booster()
        self._classifier.load_model(f"{filepath}_classifier.xgb")

        self._regressor = xgb.Booster()
        self._regressor.load_model(f"{filepath}_regressor.xgb")

        self._confidence_model = xgb.Booster()
        self._confidence_model.load_model(f"{filepath}_confidence.xgb")

        self._scaler = joblib.load(f"{filepath}_scaler.pkl")
        self._feature_importance = joblib.load(f"{filepath}_importance.pkl")
        print(f"Localizer models loaded with prefix: {filepath}")

    def predict_with_confidence(self, features_scaled: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Make prediction with confidence scoring and refinement.

        Args:
            features_scaled: Scaled features for a single window

        Returns:
            Tuple of (final_prediction, confidence, class_probabilities)
        """
        dtest = xgb.DMatrix(features_scaled)

        # Get predictions from multiple models
        class_probs = self._classifier.predict(dtest)
        if len(class_probs.shape) == 1:
            class_probs = class_probs.reshape(1, -1)
        class_probs = class_probs[0]

        regression_pred = float(self._regressor.predict(dtest)[0])
        confidence = float(self._confidence_model.predict(dtest)[0])

        # Weighted ensemble prediction
        class_pred = np.argmax(class_probs)
        top_k = 5  # Consider top-k predictions
        top_k_indices = np.argsort(class_probs)[-top_k:][::-1]

        # Refine prediction based on confidence and regression
        if confidence > 0.7:
            # High confidence - use classifier directly
            final_pred = class_pred
        else:
            # Lower confidence - blend with regression
            regression_pred_clipped = np.clip(regression_pred, 0, 127)

            # Find closest top-k prediction to regression
            distances = np.abs(top_k_indices - regression_pred_clipped)
            closest_idx = top_k_indices[np.argmin(distances)]

            # Weighted average
            final_pred = 0.7 * class_pred + 0.3 * closest_idx
            final_pred = int(np.round(final_pred))

        return final_pred, confidence, class_probs

    def predict_offset(self, window_data: np.ndarray) -> Tuple[int, float]:
        """
        Predict the offset within the window for precise POI location.

        Args:
            window_data: Normalized window data (256 x n_features) already preprocessed

        Returns:
            Tuple of (offset_from_window_start, confidence)
        """
        # Window data is already normalized from the main model
        # Flatten it for the XGBoost models
        features = window_data.flatten()
        features_scaled = features.reshape(1, -1)

        # Use the sophisticated prediction method
        final_pred, confidence, class_probs = self.predict_with_confidence(
            features_scaled)

        # Ensure prediction is within window bounds
        final_pred = int(np.clip(final_pred, 0, 127))

        return final_pred, float(confidence)


class POIPredictor:
    """
    POI Predictor with optional fine-tuning capability.
    """

    def __init__(self,
                 model_path: str,
                 scaler_path: str,
                 localizer_path: Optional[str] = None,
                 window_size: int = 256,
                 stride: int = 16,
                 tolerance: int = 64):
        """
        Initialize the POI Predictor with a pre-trained model.

        Args:
            model_path: Path to the .h5 model file
            scaler_path: Path to the .pkl scaler file  
            localizer_path: Optional path prefix for fine-tuning model
            window_size: Size of the sliding window for feature extraction
            stride: Step size for sliding window
            tolerance: Acceptable distance (in points) for POI detection
        """
        self.window_size = window_size
        self.stride = stride
        self.tolerance = tolerance

        # Load the main model and scaler
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)

        print(f"Loading scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)

        # Load fine-tuning model if provided
        self.localizer = None
        if localizer_path:
            print(f"Loading enhanced localizer from {localizer_path}...")
            self.localizer = EnhancedLocalizer()
            self.localizer.load(localizer_path)
            print("Fine-tuning model loaded successfully!")

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

        print("Predictor initialized successfully!")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from raw data."""
        return DP.gen_features(df)

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[int], List[float]]:
        """
        Preprocess data for prediction.

        Returns:
            Tuple of (normalized_windows, window_positions, window_times)
        """
        # Generate features
        features_df = self.generate_features(df)
        features = features_df.values

        # Create sliding windows
        windows = []
        window_positions = []
        window_times = []

        for i in range(0, len(features) - self.window_size, self.stride):
            window = features[i:i + self.window_size]
            windows.append(window)

            # Store start position of window (not center)
            window_positions.append(i)

            # Store time if available (at center of window)
            center_pos = i + self.window_size // 2
            if 'Relative_time' in df.columns and center_pos < len(df):
                window_times.append(df['Relative_time'].iloc[center_pos])
            else:
                window_times.append(center_pos)

        windows = np.array(windows)

        # Normalize windows
        original_shape = windows.shape
        windows_flat = windows.reshape(-1, windows.shape[-1])
        windows_normalized = self.scaler.transform(windows_flat)
        windows_normalized = windows_normalized.reshape(original_shape)

        return windows_normalized, window_positions, window_times

    def fine_tune_position(self, window_data: np.ndarray,
                           window_start_position: int) -> Tuple[int, float]:
        """
        Fine-tune the POI position within a window using the localizer.

        Args:
            window_data: Normalized window data (256 x n_features)
            window_start_position: Start position of window in original data

        Returns:
            Tuple of (refined_position_in_original_data, fine_tune_confidence)
        """
        if self.localizer is None:
            # No fine-tuning available, return center of window
            return window_start_position + self.window_size // 2, 1.0

        # Get offset within window from localizer
        offset, confidence = self.localizer.predict_offset(window_data)

        # Calculate refined position in original data
        refined_position = window_start_position + offset

        return refined_position, confidence

    def predict_with_confidence(self,
                                df: pd.DataFrame,
                                top_k: int = 3,
                                min_confidence: float = 0.3,
                                use_fine_tuning: bool = True) -> Dict[int, List[Dict]]:
        """
        Predict POI locations with confidence scores and optional fine-tuning.

        Args:
            df: DataFrame with sensor data
            top_k: Number of top predictions to return per POI
            min_confidence: Minimum confidence threshold
            use_fine_tuning: Whether to apply fine-tuning model

        Returns:
            Dictionary mapping POI numbers to lists of prediction dictionaries
        """
        # Preprocess data
        windows, positions, times = self.preprocess_data(df)

        # Get model predictions
        predictions = self.model.predict(windows, verbose=0)

        # Collect predictions for each POI
        poi_predictions = {}

        for poi_num, pred_idx in self.poi_indices.items():
            poi_probs = predictions[:, pred_idx]

            # Find indices above minimum confidence
            valid_indices = np.where(poi_probs >= min_confidence)[0]

            if len(valid_indices) > 0:
                # Sort by confidence
                sorted_indices = valid_indices[np.argsort(
                    poi_probs[valid_indices])[::-1]]

                # Take top K
                top_indices = sorted_indices[:top_k]

                # Create prediction list
                predictions_list = []
                for idx in top_indices:
                    # Get window start position and coarse position (center)
                    window_start = positions[idx]
                    coarse_position = window_start + self.window_size // 2
                    confidence = float(poi_probs[idx])

                    # Apply fine-tuning if available and requested
                    if use_fine_tuning and self.localizer:
                        refined_pos, fine_conf = self.fine_tune_position(
                            windows[idx], window_start
                        )
                        final_position = refined_pos
                        final_confidence = confidence * fine_conf
                    else:
                        final_position = coarse_position
                        final_confidence = confidence

                    pred_dict = {
                        'window_index': idx,
                        'data_index': final_position,
                        'coarse_index': coarse_position,
                        'confidence': final_confidence,
                        'coarse_confidence': confidence,
                        'time': times[idx] if times else None,
                        'fine_tuned': use_fine_tuning and self.localizer is not None
                    }
                    predictions_list.append(pred_dict)

                poi_predictions[poi_num] = predictions_list
            else:
                poi_predictions[poi_num] = []

        return poi_predictions

    def predict_best(self,
                     df: pd.DataFrame,
                     use_nms: bool = True,
                     nms_window: int = 30,
                     use_fine_tuning: bool = True,
                     adaptive_thresholds: Optional[Dict[int, float]] = None) -> Dict[int, Dict]:
        """
        Predict the single best location for each POI with fine-tuning.
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

        # Preprocess data
        windows, positions, times = self.preprocess_data(df)

        # Get model predictions
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
                    window_start = positions[best_idx]
                    coarse_position = window_start + self.window_size // 2
                    confidence = float(poi_probs[best_idx])

                    # Apply fine-tuning
                    if use_fine_tuning and self.localizer:
                        refined_pos, fine_conf = self.fine_tune_position(
                            windows[best_idx], window_start
                        )
                        final_position = refined_pos
                        final_confidence = confidence * fine_conf
                    else:
                        final_position = coarse_position
                        final_confidence = confidence

                    best_predictions[poi_num] = {
                        'data_index': final_position,
                        'coarse_index': coarse_position,
                        'confidence': final_confidence,
                        'coarse_confidence': confidence,
                        'time': times[best_idx] if times else None,
                        'window_index': best_idx,
                        'fine_tuned': use_fine_tuning and self.localizer is not None
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

    def apply_constraints(self,
                          predictions: Dict[int, Dict],
                          df: pd.DataFrame,
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

    def get_prediction_summary(self,
                               df: pd.DataFrame,
                               apply_constraints: bool = True,
                               use_fine_tuning: bool = True) -> str:
        """Get a formatted summary of predictions with fine-tuning info."""
        # Get best predictions
        predictions = self.predict_best(df, use_fine_tuning=use_fine_tuning)

        if apply_constraints:
            predictions = self.apply_constraints(predictions, df)

        # Get top K for additional context
        top_k_predictions = self.predict_with_confidence(
            df, top_k=3, use_fine_tuning=use_fine_tuning
        )

        # Build summary
        summary = []
        summary.append("=" * 60)
        summary.append("POI PREDICTION SUMMARY")
        if use_fine_tuning and self.localizer:
            summary.append("(With Fine-Tuning Enabled)")
        summary.append("=" * 60)

        for poi_num in sorted(self.poi_indices.keys()):
            summary.append(
                f"\n{self.poi_names.get(poi_num, f'POI-{poi_num}')}:")

            if poi_num in predictions:
                pred = predictions[poi_num]
                summary.append(f"  Best Prediction:")
                summary.append(f"    Final Index: {pred['data_index']}")

                if pred.get('fine_tuned') and 'coarse_index' in pred:
                    summary.append(f"    Coarse Index: {pred['coarse_index']}")
                    offset = pred['data_index'] - pred['coarse_index']
                    summary.append(f"    Fine-tune Offset: {offset:+d}")

                summary.append(f"    Confidence: {pred['confidence']:.3f}")

                if pred.get('time') is not None:
                    summary.append(f"    Time: {pred['time']:.3f}")

                # Add alternative predictions if available
                if poi_num in top_k_predictions and len(top_k_predictions[poi_num]) > 1:
                    summary.append(f"  Alternative Predictions:")
                    for i, alt in enumerate(top_k_predictions[poi_num][1:3], 1):
                        summary.append(f"    #{i+1}: Index {alt['data_index']}, "
                                       f"Confidence {alt['confidence']:.3f}")
            else:
                summary.append(f"  No detection above threshold")

                # Show best candidate even if below threshold
                if poi_num in top_k_predictions and top_k_predictions[poi_num]:
                    best = top_k_predictions[poi_num][0]
                    summary.append(f"  Best candidate: Index {best['data_index']}, "
                                   f"Confidence {best['confidence']:.3f} (below threshold)")

        summary.append("\n" + "=" * 60)

        # Add gap analysis if time information available
        if 'Relative_time' in df.columns and len(predictions) > 1:
            summary.append("\nTEMPORAL GAP ANALYSIS:")
            summary.append("-" * 40)

            sorted_pois = sorted(predictions.items(),
                                 key=lambda x: x[1]['data_index'])

            for i in range(len(sorted_pois) - 1):
                poi1_num, poi1_data = sorted_pois[i]
                poi2_num, poi2_data = sorted_pois[i + 1]

                if poi1_data.get('time') and poi2_data.get('time'):
                    gap = poi2_data['time'] - poi1_data['time']
                    summary.append(
                        f"  POI-{poi1_num} to POI-{poi2_num}: {gap:.3f}")

        return "\n".join(summary)


if __name__ == "__main__":
    # Initialize predictor with both models
    predictor = POIPredictor(
        model_path=r"QModel\src\models\static_v4\v4_model.h5",
        scaler_path=r"QModel\src\models\static_v4\v4_scaler.joblib",
        localizer_path=r"QModel\src\models\static_v4\v4_localizer",  # Add localizer
        window_size=256,
        stride=16,
        tolerance=64
    )

    content = DP.load_content('content/static/valid')
    for d, p in content:
        df_test = pd.read_csv(d)
        poi_test = pd.read_csv(p, header=None).values

        # Get predictions with fine-tuning
        print("\n" + "="*60)
        print("TOP K PREDICTIONS PER POI (WITH FINE-TUNING)")
        print("="*60)
        top_predictions = predictor.predict_with_confidence(
            df_test, top_k=3, use_fine_tuning=True
        )
        for poi_num, preds in top_predictions.items():
            print(f"\nPOI-{poi_num}:")
            for i, pred in enumerate(preds, 1):
                print(f"  #{i}: Index {pred['data_index']} "
                      f"(coarse: {pred['coarse_index']}), "
                      f"Confidence {pred['confidence']:.3f}")

        # Compare with and without fine-tuning
        print("\n" + "="*60)
        print("COMPARISON: WITH vs WITHOUT FINE-TUNING")
        print("="*60)

        # Without fine-tuning
        predictions_no_ft = predictor.predict_best(
            df_test, use_fine_tuning=False)
        # With fine-tuning
        predictions_with_ft = predictor.predict_best(
            df_test, use_fine_tuning=True)

        for poi_num in sorted(predictor.poi_indices.keys()):
            print(f"\nPOI-{poi_num}:")
            if poi_num in predictions_no_ft:
                print(
                    f"  Without FT: Index {predictions_no_ft[poi_num]['data_index']}")
            if poi_num in predictions_with_ft:
                print(
                    f"  With FT:    Index {predictions_with_ft[poi_num]['data_index']}")
                if poi_num in predictions_no_ft:
                    diff = predictions_with_ft[poi_num]['data_index'] - \
                        predictions_no_ft[poi_num]['data_index']
                    print(f"  Adjustment: {diff:+d} points")

        # Get formatted summary
        print("\n" + predictor.get_prediction_summary(
            df_test, apply_constraints=True, use_fine_tuning=True
        ))

        # ————— Enhanced Plotting —————
        plt.figure(figsize=(14, 8))

        # Create two subplots
        plt.subplot(2, 1, 1)
        plt.plot(df_test['Relative_time'], df_test['Dissipation'],
                 label='Dissipation Curve', alpha=0.7, color='gray')

        # Plot predictions
        markers = ['o', 's', 'v', '^', 'D', 'x']
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

        for idx, (poi_num, preds) in enumerate(top_predictions.items()):
            if preds:  # Only plot if there are predictions
                # Plot fine-tuned positions
                best_pred = preds[0]  # Best prediction
                if best_pred['data_index'] < len(df_test):
                    time_fine = df_test.loc[best_pred['data_index'],
                                            'Relative_time']
                    diss_fine = df_test.loc[best_pred['data_index'],
                                            'Dissipation']

                    # Plot coarse position
                    if best_pred['coarse_index'] < len(df_test):
                        time_coarse = df_test.loc[best_pred['coarse_index'],
                                                  'Relative_time']
                        diss_coarse = df_test.loc[best_pred['coarse_index'],
                                                  'Dissipation']

                        # Coarse prediction (hollow)
                        plt.scatter(time_coarse, diss_coarse,
                                    marker=markers[idx % len(markers)],
                                    facecolors='none',
                                    edgecolors=colors[idx % len(colors)],
                                    s=150, linewidth=2,
                                    label=f'POI {poi_num} Coarse')

                        # Fine-tuned prediction (filled)
                        plt.scatter(time_fine, diss_fine,
                                    marker=markers[idx % len(markers)],
                                    color=colors[idx % len(colors)],
                                    s=100,
                                    label=f'POI {poi_num} Fine')

                        # Arrow from coarse to fine
                        # Only show arrow if there's adjustment
                        if abs(time_fine - time_coarse) > 0.01:
                            plt.annotate('', xy=(time_fine, diss_fine),
                                         xytext=(time_coarse, diss_coarse),
                                         arrowprops=dict(arrowstyle='->',
                                                         color=colors[idx %
                                                                      len(colors)],
                                                         alpha=0.5, lw=1))

        # Plot ground truth
        r_time = df_test["Relative_time"].values
        for idx in poi_test:
            if idx[0] < len(r_time):
                plt.axvline(r_time[idx[0]], color='black',
                            linestyle='--', alpha=0.3, label='Ground Truth' if idx[0] == poi_test[0][0] else '')

        plt.xlabel('Relative Time')
        plt.ylabel('Dissipation')
        plt.title('POI Predictions: Coarse vs Fine-tuned')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # Second subplot: Show adjustment distances
        plt.subplot(2, 1, 2)
        poi_nums = []
        adjustments = []

        for poi_num in sorted(predictor.poi_indices.keys()):
            if poi_num in predictions_with_ft and poi_num in predictions_no_ft:
                poi_nums.append(f'POI-{poi_num}')
                adj = predictions_with_ft[poi_num]['data_index'] - \
                    predictions_no_ft[poi_num]['data_index']
                adjustments.append(adj)

        if adjustments:
            bars = plt.bar(poi_nums, adjustments, color=colors[:len(poi_nums)])
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.xlabel('POI')
            plt.ylabel('Fine-tuning Adjustment (points)')
            plt.title('Fine-tuning Adjustments from Coarse Predictions')
            plt.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, val in zip(bars, adjustments):
                plt.text(bar.get_x() + bar.get_width()/2, val,
                         f'{val:+d}', ha='center', va='bottom' if val > 0 else 'top')

        plt.tight_layout()
        plt.show()
