import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import joblib
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
from fusion_dataprocessor import FusionDataprocessor

warnings.filterwarnings('ignore')


class V4ClfModel(nn.Module):
    """PyTorch model for POI detection - inference version."""

    def __init__(self,
                 window_size: int,
                 feature_dim: int,
                 conv_filters_1: int = 64,
                 conv_filters_2: int = 128,
                 kernel_size: int = 3,
                 lstm_units_1: int = 128,
                 lstm_units_2: int = 64,
                 dense_units: int = 128,
                 dropout_rate: float = 0.3):
        super(V4ClfModel, self).__init__()

        self.window_size = window_size
        self.feature_dim = feature_dim

        # Convolutional layers
        self.conv1 = nn.Conv1d(feature_dim, conv_filters_1,
                               kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(conv_filters_1)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(
            conv_filters_1, conv_filters_2, 3, padding='same')
        self.bn2 = nn.BatchNorm1d(conv_filters_2)
        self.pool2 = nn.MaxPool1d(2)

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(conv_filters_2, lstm_units_1,
                             batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.lstm2 = nn.LSTM(lstm_units_1 * 2, lstm_units_2,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Dense layers
        self.dense = nn.Linear(lstm_units_2 * 2, dense_units)
        self.bn3 = nn.BatchNorm1d(dense_units)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer (6 classes: Non-POI, POI1, POI2, POI4, POI5, POI6)
        self.output = nn.Linear(dense_units, 6)

    def forward(self, x):
        # Input shape: (batch_size, window_size, feature_dim)
        # Conv1d expects: (batch_size, feature_dim, window_size)
        x = x.transpose(1, 2)

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Back to (batch_size, seq_len, features) for LSTM
        x = x.transpose(1, 2)

        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Take the last output from LSTM
        x = x[:, -1, :]

        # Dense layers
        x = F.relu(self.bn3(self.dense(x)))
        x = self.dropout3(x)

        # Output with sigmoid activation for multi-label classification
        x = torch.sigmoid(self.output(x))

        return x


class V4ClfPredictor:
    """
    Standalone predictor for POI detection in viscosity sensor data.
    Loads trained model assets and performs inference with visualization.
    """

    def __init__(self,
                 model_path: str = 'v4_model_pytorch.pth',
                 scaler_path: str = 'v4_scaler_pytorch.joblib',
                 config_path: str = 'v4_config_pytorch.json',
                 device: str = None):
        """
        Initialize the POI Predictor.

        Args:
            model_path: Path to the saved PyTorch model state dict
            scaler_path: Path to the saved StandardScaler
            config_path: Path to the saved model configuration
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Set device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model and configuration
        self.load_model(model_path, scaler_path, config_path)

    def load_model(self, model_path: str, scaler_path: str, config_path: str) -> None:
        """Load the trained model, scaler, and configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.window_size = config['window_size']
        self.stride = config['stride']
        self.tolerance = config['tolerance']
        self.feature_dim = config['feature_dim']
        self.best_params = config.get('best_params')

        # Load scaler
        self.scaler = joblib.load(scaler_path)

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)

        # Create model with best parameters if available
        if self.best_params:
            model_params = {k: v for k, v in self.best_params.items()
                            if k not in ['learning_rate', 'batch_size']}
            self.model = self._create_model(**model_params)
        else:
            self.model = self._create_model()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Scaler loaded from {scaler_path}")
        print(f"✓ Configuration loaded from {config_path}")

    def _create_model(self, **params) -> V4ClfModel:
        """Create model with given parameters."""
        return V4ClfModel(
            window_size=self.window_size,
            feature_dim=self.feature_dim,
            **params
        ).to(self.device)

    def predict(self,
                data_path: str = None,
                df: pd.DataFrame = None,
                threshold: float = 0.5,
                adaptive_thresholds: Dict[int, float] = None,
                enforce_constraints: bool = True) -> Dict[str, any]:
        """
        Predict POIs for a single datafile.

        Args:
            data_path: Path to CSV file (if df not provided)
            df: DataFrame with sensor data (if data_path not provided)
            threshold: Default threshold for POI detection
            adaptive_thresholds: Custom thresholds per POI type
            enforce_constraints: Whether to apply sequential/temporal constraints

        Returns:
            Dictionary containing:
                - 'poi_locations': Dict of POI numbers and their indices
                - 'poi_count': Total number of POIs detected
                - 'poi_binary': Binary array indicating POI presence [POI1, POI2, POI4, POI5, POI6]
                - 'probabilities': Raw model output probabilities
                - 'dataframe': The processed dataframe
        """
        # Load data if path provided
        if data_path is not None:
            df = pd.read_csv(data_path)
            print(f"Loaded data from {data_path}: {len(df)} samples")
        elif df is None:
            raise ValueError("Either data_path or df must be provided")

        # Generate features
        features_df = FusionDataprocessor.gen_features(df)
        features = features_df.values

        # Create windows
        windows = []
        window_positions = []

        n = len(features)
        w = self.window_size
        s = self.stride

        for i in range(0, n, s):
            # compute slice
            end = i + w
            if end <= n:
                # full window fits
                window = features[i:end]
            else:
                # pad with the last row value to reach window_size
                pad_len = end - n
                last_val = features[-1]
                # replicate last value for padding
                pad = np.repeat(last_val[np.newaxis, :], pad_len, axis=0)
                window = np.vstack([features[i:n], pad])
            windows.append(window)

            # window center (use min to clamp to end of data)
            center = min(i + w // 2, n - 1)
            window_positions.append(center)

        if len(windows) == 0:
            print("Warning: Not enough data for prediction")
            return {
                'poi_locations': {},
                'poi_count': 0,
                'poi_binary': np.zeros(5),
                'probabilities': None,
                'dataframe': df
            }

        windows = np.array(windows)

        # Normalize
        windows_reshaped = windows.reshape(-1, windows.shape[-1])
        windows_reshaped = self.scaler.transform(windows_reshaped)
        windows = windows_reshaped.reshape(windows.shape)

        # Get predictions
        with torch.no_grad():
            windows_tensor = torch.FloatTensor(windows).to(self.device)
            predictions = self.model(windows_tensor).cpu().numpy()

        # Set adaptive thresholds
        if adaptive_thresholds is None:
            adaptive_thresholds = {
                1: 0.5,   # POI-1
                2: 0.5,   # POI-2
                4: 0.01,   # POI-4
                5: 0.1,   # POI-5
                6: 0.1   # POI-6
            }

        # Find POI candidates
        poi_locations = self._find_poi_candidates(
            predictions, window_positions, adaptive_thresholds)

        # Apply constraints if requested
        if enforce_constraints:
            poi_locations = self._enforce_sequential_constraints(poi_locations)
            if 'Relative_time' in df.columns:
                poi_locations = self._enforce_relative_gap_constraints(
                    poi_locations, df)

        # Create binary output
        poi_binary = np.zeros(5)
        for poi_num in [1, 2, 4, 5, 6]:
            if poi_num in poi_locations:
                idx = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}[poi_num]
                poi_binary[idx] = 1

        # Calculate POI count
        poi_count = len(poi_locations)

        # Print results
        print(f"\n=== Prediction Results ===")
        print(f"POIs detected: {poi_count}")
        print(f"POI locations: {poi_locations}")
        print(
            f"Binary output [POI1, POI2, POI4, POI5, POI6]: {poi_binary.astype(int)}")

        return {
            'poi_locations': poi_locations,
            'poi_count': poi_count,
            'poi_binary': poi_binary,
            'probabilities': predictions,
            'window_positions': window_positions,
            'dataframe': df
        }

    def _find_poi_candidates(self, predictions: np.ndarray,
                             window_positions: List[int],
                             thresholds: Dict[int, float]) -> Dict[int, int]:
        """Find initial POI candidates from predictions."""
        poi_locations = {}
        poi_indices = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}

        for poi_num, pred_idx in poi_indices.items():
            poi_probs = predictions[:, pred_idx]
            poi_threshold = thresholds.get(poi_num, 0.5)

            above_threshold = poi_probs > poi_threshold
            if np.any(above_threshold):
                # Find peak
                peak_indices = self._find_peaks(poi_probs, above_threshold)

                if len(peak_indices) > 0:
                    best_idx = peak_indices[np.argmax(poi_probs[peak_indices])]
                    poi_locations[poi_num] = window_positions[best_idx]

        return poi_locations

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

    def _enforce_sequential_constraints(self, poi_candidates: Dict[int, int]) -> Dict[int, int]:
        """Enforce sequential ordering constraints."""
        validated_pois = {}

        # POI1 and POI2 can exist independently
        if 1 in poi_candidates:
            validated_pois[1] = poi_candidates[1]
        if 2 in poi_candidates:
            validated_pois[2] = poi_candidates[2]

        # POI4 requires both POI1 and POI2
        if 4 in poi_candidates:
            if 1 in validated_pois and 2 in validated_pois:
                if poi_candidates[4] > validated_pois[2]:
                    validated_pois[4] = poi_candidates[4]

        # POI5 requires POI4
        if 5 in poi_candidates:
            if 4 in validated_pois:
                if poi_candidates[5] > validated_pois[4]:
                    validated_pois[5] = poi_candidates[5]

        # POI6 requires POI5
        if 6 in poi_candidates:
            if 5 in validated_pois:
                if poi_candidates[6] > validated_pois[5]:
                    validated_pois[6] = poi_candidates[6]

        return validated_pois

    def _enforce_relative_gap_constraints(self, poi_candidates: Dict[int, int],
                                          df: pd.DataFrame) -> Dict[int, int]:
        """Apply relative time gap constraints."""
        refined_pois = poi_candidates.copy()

        if 'Relative_time' not in df.columns:
            return refined_pois

        def get_time(idx):
            if idx < len(df):
                return df['Relative_time'].iloc[idx]
            return None

        # Check gap constraints
        if 1 in refined_pois and 2 in refined_pois:
            time1 = get_time(refined_pois[1])
            time2 = get_time(refined_pois[2])

            if time1 is not None and time2 is not None:
                gap_1_2 = abs(time2 - time1)

                # Check POI4
                if 4 in refined_pois:
                    time4 = get_time(refined_pois[4])
                    if time4 is not None:
                        gap_2_4 = abs(time4 - time2)
                        if gap_2_4 < gap_1_2:
                            del refined_pois[4]

        # Remove dependent POIs if parent removed
        if 5 in refined_pois and 4 not in refined_pois:
            del refined_pois[5]
        if 6 in refined_pois and 5 not in refined_pois:
            del refined_pois[6]

        return refined_pois

    def visualize(self,
                  prediction_result: Dict,
                  save_path: str = None,
                  show_plot: bool = True) -> None:
        """
        Visualize POI predictions.

        Args:
            prediction_result: Output from predict() method
            save_path: Path to save the figure
            show_plot: Whether to display the plot
        """
        df = prediction_result['dataframe']
        poi_locations = prediction_result['poi_locations']
        probabilities = prediction_result['probabilities']
        window_positions = prediction_result['window_positions']

        # Determine x-axis
        use_time = 'Relative_time' in df.columns
        x_values = df['Relative_time'].values if use_time else np.arange(
            len(df))
        x_label = 'Relative Time' if use_time else 'Sample Index'

        # Window x-values
        window_x_values = []
        for pos in window_positions:
            if use_time and pos < len(df):
                window_x_values.append(df['Relative_time'].iloc[pos])
            else:
                window_x_values.append(pos)

        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

        # Colors for POIs
        poi_colors = {1: 'red', 2: 'orange',
                      4: 'yellow', 5: 'green', 6: 'blue'}

        # Plot 1: Dissipation
        if 'Dissipation' in df.columns:
            axes[0].plot(x_values, df['Dissipation'].values,
                         'b-', alpha=0.7, lw=0.5)
            axes[0].set_ylabel('Dissipation')
            axes[0].grid(True, alpha=0.3)

        # Plot 2: Resonance Frequency
        if 'Resonance_Frequency' in df.columns:
            axes[1].plot(x_values, df['Resonance_Frequency'].values,
                         'g-', alpha=0.7, lw=0.5)
            axes[1].set_ylabel('Resonance Frequency')
            axes[1].grid(True, alpha=0.3)

        # Plot 3: POI Probabilities
        if probabilities is not None:
            poi_indices = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
            for poi_num, idx in poi_indices.items():
                axes[2].plot(window_x_values, probabilities[:, idx],
                             color=poi_colors[poi_num], alpha=0.6,
                             label=f'POI-{poi_num}', lw=1)

            axes[2].set_ylabel('POI Probabilities')
            axes[2].legend(loc='upper right')
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(0.5, color='black', linestyle=':', alpha=0.5)

            # Any POI confidence
            any_conf = 1 - probabilities[:, 0]
            axes[3].plot(window_x_values, any_conf, 'r-', alpha=0.7, lw=1)
            axes[3].set_ylabel('Any POI Confidence')
            axes[3].grid(True, alpha=0.3)

        # Add vertical lines for detected POIs
        for poi_num, idx in poi_locations.items():
            if use_time and idx < len(df):
                x_pos = df['Relative_time'].iloc[idx]
            else:
                x_pos = idx

            for ax in axes:
                ax.axvline(x_pos, color=poi_colors.get(poi_num, 'black'),
                           linestyle='--', alpha=0.8, label=f'POI-{poi_num}' if ax == axes[0] else '')

        # Add shaded regions for POI windows
        half_win = 64  # Half of 128 window size
        for poi_num, idx in poi_locations.items():
            start_idx = max(0, idx - half_win)
            end_idx = min(len(df) - 1, idx + half_win)

            if use_time:
                x_start = df['Relative_time'].iloc[start_idx]
                x_end = df['Relative_time'].iloc[end_idx]
            else:
                x_start, x_end = start_idx, end_idx

            for ax in axes:
                ax.axvspan(x_start, x_end, color=poi_colors.get(poi_num, 'gray'),
                           alpha=0.1)

        # Labels and title
        axes[0].set_title(
            f'POI Detection Results - {len(poi_locations)} POIs Detected')
        axes[-1].set_xlabel(x_label)

        if poi_locations:
            axes[0].legend(loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        if show_plot:
            plt.show()

        # Print summary
        print("\n=== Detection Summary ===")
        for poi_num in sorted(poi_locations.keys()):
            idx = poi_locations[poi_num]
            if use_time and idx < len(df):
                time_val = df['Relative_time'].iloc[idx]
                print(f"POI-{poi_num}: Index {idx} (Time: {time_val:.3f})")
            else:
                print(f"POI-{poi_num}: Index {idx}")


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = V4ClfPredictor(
        model_path='fusion/fusion_models/v4_model_pytorch.pth',
        scaler_path='fusion/fusion_models/v4_scaler_pytorch.joblib',
        config_path='fusion/fusion_models/v4_config_pytorch.json'
    )

    # Example 1: Predict from file path
    result = predictor.predict(
        data_path='M250505W3_IGG_NOSUC_20C_3rd.csv',
        enforce_constraints=True
    )

    # Visualize results
    predictor.visualize(result, save_path='poi_prediction.png')
    # Access results
    print(f"Number of POIs: {result['poi_count']}")
    print(f"Binary output: {result['poi_binary']}")
    print(f"POI locations: {result['poi_locations']}")
