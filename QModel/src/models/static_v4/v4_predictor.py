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


class POIPredictor:
    """
    A class for predicting Points of Interest using a pre-trained model.
    Loads model and scaler from files and provides predictions with confidence scores.
    """

    def __init__(self,
                 model_path: str,
                 scaler_path: str,
                 window_size: int = 128,
                 stride: int = 16,
                 tolerance: int = 64):
        """
        Initialize the POI Predictor with a pre-trained model.

        Args:
            model_path: Path to the .h5 model file
            scaler_path: Path to the .pkl scaler file
            window_size: Size of the sliding window for feature extraction
            stride: Step size for sliding window
            tolerance: Acceptable distance (in points) for POI detection
        """
        self.window_size = window_size
        self.stride = stride
        self.tolerance = tolerance

        # Load the model and scaler
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)

        print(f"Loading scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)

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
        """
        Generate features from raw data.
        This is a simplified version - replace with your actual DP.gen_features logic.

        Args:
            df: Raw data DataFrame

        Returns:
            DataFrame with generated features
        """
        # Import your actual feature generation module here
        # For now, using a placeholder that assumes features are already in the DataFrame
        return DP.gen_features(df)

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[int], List[float]]:
        """
        Preprocess data for prediction.

        Args:
            df: Input DataFrame with sensor data

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

            # Store center position of window
            center_pos = i + self.window_size // 2
            window_positions.append(center_pos)

            # Store time if available
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

    def predict_with_confidence(self,
                                df: pd.DataFrame,
                                top_k: int = 3,
                                min_confidence: float = 0.3) -> Dict[int, List[Dict]]:
        """
        Predict POI locations with confidence scores, returning top K predictions per POI.

        Args:
            df: DataFrame with sensor data
            top_k: Number of top predictions to return per POI
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary mapping POI numbers to lists of prediction dictionaries
            Each prediction dict contains: {index, confidence, time (if available)}
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
                    pred_dict = {
                        'window_index': idx,
                        'data_index': positions[idx],
                        'confidence': float(poi_probs[idx]),
                        'time': times[idx] if times else None
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
                     adaptive_thresholds: Optional[Dict[int, float]] = None) -> Dict[int, Dict]:
        """
        Predict the single best location for each POI with confidence.

        Args:
            df: DataFrame with sensor data
            use_nms: Whether to use non-maximum suppression
            nms_window: Window size for NMS
            adaptive_thresholds: Optional POI-specific thresholds

        Returns:
            Dictionary mapping POI numbers to prediction info
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

                    best_predictions[poi_num] = {
                        'data_index': positions[best_idx],
                        'confidence': float(poi_probs[best_idx]),
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

    def apply_constraints(self,
                          predictions: Dict[int, Dict],
                          df: pd.DataFrame,
                          enforce_sequential: bool = True,
                          enforce_gaps: bool = True) -> Dict[int, Dict]:
        """
        Apply sequential and temporal constraints to predictions.

        Args:
            predictions: Dictionary of POI predictions
            df: Original DataFrame (for time information)
            enforce_sequential: Apply sequential ordering constraints
            enforce_gaps: Apply relative time gap constraints

        Returns:
            Filtered predictions after applying constraints
        """
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
                               apply_constraints: bool = True) -> str:
        """
        Get a formatted summary of predictions.

        Args:
            df: DataFrame with sensor data
            apply_constraints: Whether to apply constraints

        Returns:
            Formatted string summary
        """
        # Get best predictions
        predictions = self.predict_best(df)

        if apply_constraints:
            predictions = self.apply_constraints(predictions, df)

        # Get top K for additional context
        top_k_predictions = self.predict_with_confidence(df, top_k=3)

        # Build summary
        summary = []
        summary.append("=" * 60)
        summary.append("POI PREDICTION SUMMARY")
        summary.append("=" * 60)

        for poi_num in sorted(self.poi_indices.keys()):
            summary.append(
                f"\n{self.poi_names.get(poi_num, f'POI-{poi_num}')}:")

            if poi_num in predictions:
                pred = predictions[poi_num]
                summary.append(f"  Best Prediction:")
                summary.append(f"    Data Index: {pred['data_index']}")
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
    # Initialize predictor with pre-trained model
    predictor = POIPredictor(
        model_path=r"QModel\src\models\static_v4\v4_model.h5",
        scaler_path=r"QModel\src\models\static_v4\v4_scaler.joblib",
        window_size=128,
        stride=16,
        tolerance=64
    )
    content = DP.load_content('content/static/valid')
    for d, p in content:
        df_test = pd.read_csv(d)
        poi_test = pd.read_csv(p, header=None).values

        # Get predictions with confidence scores
        print("\n" + "="*60)
        print("TOP K PREDICTIONS PER POI")
        print("="*60)
        top_predictions = predictor.predict_with_confidence(df_test, top_k=3)
        for poi_num, preds in top_predictions.items():
            print(f"\nPOI-{poi_num}:")
            for i, pred in enumerate(preds, 1):
                print(
                    f"  #{i}: Index {pred['data_index']}, Confidence {pred['confidence']:.3f}")

        # Get best predictions with constraints
        print("\n" + "="*60)
        print("BEST PREDICTIONS (WITH CONSTRAINTS)")
        print("="*60)
        best_predictions = predictor.predict_best(df_test)
        constrained_predictions = predictor.apply_constraints(
            best_predictions, df_test)

        for poi_num, pred in constrained_predictions.items():
            print(f"POI-{poi_num}: Index {pred['data_index']}, "
                  f"Confidence {pred['confidence']:.3f}")

        # Get formatted summary
        print("\n" + predictor.get_prediction_summary(df_test, apply_constraints=True))

        # ————— Plotting —————
        # Plot the dissipation curve
        plt.figure()
        plt.plot(df_test['Relative_time'], df_test['Dissipation'],
                 label='Dissipation Curve')

        # Overlay the top-K points for each POI
        markers = ['o', 's', 'v', '^', 'D', 'x']  # as many as you need
        for idx, (poi_num, preds) in enumerate(top_predictions.items()):
            # extract times and dissipation values at the predicted indices
            times = [df_test.loc[p['data_index'], 'Relative_time']
                     for p in preds]
            diss_vals = [df_test.loc[p['data_index'], 'Dissipation']
                         for p in preds]
            plt.scatter(times,
                        diss_vals,
                        marker=markers[idx % len(markers)],
                        label=f'POI {poi_num} Top {len(preds)}')
        r_time = df_test["Relative_time"].values
        for idx in poi_test:
            plt.axvline(r_time[idx])

        plt.xlabel('Relative_time')
        plt.ylabel('Dissipation')
        plt.title('Dissipation Curve with Top-K POI Predictions')
        plt.legend()
        plt.tight_layout()
        plt.show()
