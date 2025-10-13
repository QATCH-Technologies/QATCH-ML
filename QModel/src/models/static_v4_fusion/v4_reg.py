import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from fusion_dataprocessor import FusionDataprocessor
import warnings

warnings.filterwarnings('ignore')


class V4RegModel(nn.Module):
    """Neural network for universal POI detection using regression - inference version"""

    def __init__(self, window_size: int, feature_dim: int, config: Dict[str, any]):
        super(V4RegModel, self).__init__()

        # Extract hyperparameters
        conv_filters = config.get('conv_filters', 64)
        kernel_size = config.get('kernel_size', 5)
        conv_filters_2 = config.get('conv_filters_2', 128)
        lstm_units = config.get('lstm_units', 128)
        lstm_units_2 = config.get('lstm_units_2', 64)
        dense_units = config.get('dense_units', 128)
        dropout_1 = config.get('dropout_1', 0.3)
        dropout_2 = config.get('dropout_2', 0.3)
        dropout_3 = config.get('dropout_3', 0.3)

        # Convolutional layers
        self.conv1 = nn.Conv1d(feature_dim, conv_filters,
                               kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(conv_filters, conv_filters_2,
                               kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(conv_filters_2)
        self.pool2 = nn.MaxPool1d(2)

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(conv_filters_2, lstm_units,
                             batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_1)

        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units_2,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout_2)

        # Dense layers
        self.fc1 = nn.Linear(lstm_units_2 * 2, dense_units)
        self.bn3 = nn.BatchNorm1d(dense_units)
        self.dropout3 = nn.Dropout(dropout_3)

        # Regression output layer (sigmoid for 0-1 range)
        self.output = nn.Linear(dense_units, 1)

    def forward(self, x):
        # Input shape: (batch, window_size, features)
        # Transpose for Conv1d: (batch, features, window_size)
        x = x.transpose(1, 2)

        # Conv layers
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))

        # Transpose back for LSTM: (batch, sequence, features)
        x = x.transpose(1, 2)

        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Take last output
        x = x[:, -1, :]

        # Dense layers
        x = torch.relu(self.bn3(self.fc1(x)))
        x = self.dropout3(x)

        # Regression output with sigmoid to constrain to [0, 1]
        x = torch.sigmoid(self.output(x))

        return x.squeeze()


class V4RegPredictor:
    """
    Standalone predictor for single POI detection using regression-based approach.
    Predicts continuous overlap parameter (OP) values and finds POI position.
    """

    def __init__(self,
                 model_path: str,
                 poi_type: str = None,
                 device: str = None):
        """
        Initialize the Regression POI Predictor.

        Args:
            model_path: Path to the saved model checkpoint
            poi_type: Type of POI this model predicts (e.g., 'POI1', 'POI2', etc.)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Set device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Extract POI type from model path if not provided
        if poi_type is None:
            # Try to extract from filename (e.g., "poi_model_small_window_0.pth" -> POI1)
            path_stem = Path(model_path).stem
            if path_stem[-1].isdigit():
                poi_idx = int(path_stem[-1])
                poi_map = {0: 'POI1', 1: 'POI2',
                           3: 'POI4', 4: 'POI5', 5: 'POI6'}
                self.poi_type = poi_map.get(poi_idx, f'POI{poi_idx}')
            else:
                self.poi_type = 'POI'
        else:
            self.poi_type = poi_type

        # Load model
        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load the trained model and parameters."""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load configuration
        self.config = checkpoint['config']
        self.window_size = checkpoint['window_size']
        self.stride = checkpoint['stride']
        self.tolerance = checkpoint['tolerance']
        self.feature_dim = checkpoint['feature_dim']
        self.gaussian_sigma = checkpoint.get('gaussian_sigma', 2.0)
        self.peak_threshold = checkpoint.get('peak_threshold', 0.3)

        # Load scaler parameters
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']

        # Create and load model
        self.model = V4RegModel(
            self.window_size, self.feature_dim, self.config
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ Model loaded from {model_path}")
        print(f"  POI Type: {self.poi_type}")
        print(f"  Window size: {self.window_size}")
        print(f"  Stride: {self.stride}")
        print(f"  Tolerance: {self.tolerance}")
        print(f"  Gaussian σ: {self.gaussian_sigma}")
        print(f"  Peak threshold: {self.peak_threshold}")

    def predict(self,
                data_path: str = None,
                df: pd.DataFrame = None,
                apply_smoothing: bool = True,
                custom_threshold: float = None) -> Dict[str, any]:
        """
        Predict POI position for a single datafile.

        Args:
            data_path: Path to CSV file (if df not provided)
            df: DataFrame with sensor data (if data_path not provided)
            apply_smoothing: Whether to apply Gaussian smoothing
            custom_threshold: Override default peak detection threshold

        Returns:
            Dictionary containing:
                - 'poi_position': Detected POI position (index)
                - 'poi_confidence': Confidence value (0-1)
                - 'op_values': Raw overlap parameter predictions
                - 'op_smoothed': Smoothed OP values (if smoothing applied)
                - 'window_positions': Window center positions
                - 'all_peaks': All detected peaks (if multiple found)
                - 'dataframe': The processed dataframe
        """
        # Load data if path provided
        if data_path is not None:
            df = pd.read_csv(data_path)
            print(f"Loaded data from {data_path}: {len(df)} samples")
        elif df is None:
            raise ValueError("Either data_path or df must be provided")

        # Generate features
        features_df = FusionDataprocessor.get_features(df)
        features = features_df.values
        n_samples = len(features)

        # Create sliding windows
        windows = []
        window_positions = []

        for i in range(0, n_samples - self.window_size + 1, self.stride):
            window = features[i:i + self.window_size]
            windows.append(window)
            window_center = i + self.window_size // 2
            window_positions.append(window_center)

        if len(windows) == 0:
            print(
                f"Warning: File too short (need at least {self.window_size} samples)")
            return {
                'poi_position': None,
                'poi_confidence': 0.0,
                'op_values': np.array([]),
                'op_smoothed': np.array([]),
                'window_positions': np.array([]),
                'all_peaks': [],
                'dataframe': df
            }

        windows = np.array(windows)
        window_positions = np.array(window_positions)

        # Normalize features
        flat = windows.reshape(-1, self.feature_dim)
        flat = self.scaler.transform(flat)
        windows_normalized = flat.reshape(windows.shape)

        # Get predictions (raw OP values)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(windows_normalized).to(self.device)
            op_values = self.model(X_tensor).cpu().numpy()

        # Apply smoothing if requested
        if apply_smoothing:
            op_smoothed = gaussian_filter1d(
                op_values, sigma=self.gaussian_sigma)
        else:
            op_smoothed = op_values.copy()

        # Find peaks
        threshold = custom_threshold if custom_threshold is not None else self.peak_threshold
        peaks, properties = find_peaks(
            op_smoothed,
            height=threshold,
            distance=self.tolerance // self.stride
        )

        # Determine primary POI position
        if len(peaks) > 0:
            # Select peak with highest confidence
            peak_heights = op_smoothed[peaks]
            best_peak_idx = np.argmax(peak_heights)
            poi_position = window_positions[peaks[best_peak_idx]]
            poi_confidence = float(peak_heights[best_peak_idx])
            all_peaks = [(window_positions[p], float(op_smoothed[p]))
                         for p in peaks]
        else:
            poi_position = None
            poi_confidence = 0.0
            all_peaks = []

        # Print results
        print(f"\n=== {self.poi_type} Prediction Results ===")
        if poi_position is not None:
            print(f"POI detected at position: {poi_position}")
            print(f"Confidence: {poi_confidence:.3f}")
            if len(all_peaks) > 1:
                print(f"Additional peaks found: {len(all_peaks) - 1}")
        else:
            print("No POI detected")
            max_op = np.max(op_smoothed) if len(op_smoothed) > 0 else 0
            print(f"Max OP value: {max_op:.3f} (threshold: {threshold:.3f})")

        return {
            'poi_position': poi_position,
            'poi_confidence': poi_confidence,
            'op_values': op_values,
            'op_smoothed': op_smoothed if apply_smoothing else None,
            'window_positions': window_positions,
            'all_peaks': all_peaks,
            'dataframe': df
        }

    def visualize(self,
                  prediction_result: Dict,
                  save_path: str = None,
                  show_plot: bool = True,
                  figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Visualize POI prediction results.

        Args:
            prediction_result: Output from predict() method
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            figsize: Figure size
        """
        df = prediction_result['dataframe']
        poi_position = prediction_result['poi_position']
        poi_confidence = prediction_result['poi_confidence']
        op_values = prediction_result['op_values']
        op_smoothed = prediction_result['op_smoothed']
        window_positions = prediction_result['window_positions']
        all_peaks = prediction_result['all_peaks']

        # Extract dissipation for plotting
        if 'Dissipation' in df.columns:
            dissipation = df['Dissipation'].values
        elif 'dissipation' in df.columns:
            dissipation = df['dissipation'].values
        else:
            dissipation = df.iloc[:, 0].values

        # Create figure
        n_plots = 3 if op_smoothed is not None else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize,
                                 height_ratios=[3] + [1] * (n_plots - 1))

        # Plot 1: Signal with POI marker
        axes[0].plot(dissipation, 'b-', linewidth=1,
                     alpha=0.7, label='Dissipation')
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Dissipation')
        axes[0].set_title(
            f'{self.poi_type} Detection Results - Regression Method')
        axes[0].grid(True, alpha=0.3)

        # Add POI marker
        if poi_position is not None:
            axes[0].axvline(x=poi_position, color='red', linestyle='--',
                            linewidth=2, alpha=0.8, label=f'{self.poi_type} Detected')
            axes[0].text(poi_position, axes[0].get_ylim()[1] * 0.95,
                         f'{self.poi_type}\n{poi_confidence:.2f}',
                         rotation=0, ha='center', va='top',
                         bbox=dict(boxstyle='round',
                                   facecolor='white', alpha=0.8),
                         fontsize=10, color='red')

            # Add shaded region around POI
            shade_start = max(0, poi_position - self.tolerance)
            shade_end = min(len(dissipation), poi_position + self.tolerance)
            axes[0].axvspan(shade_start, shade_end, color='red', alpha=0.1)

        # Add markers for additional peaks if present
        for i, (pos, conf) in enumerate(all_peaks[1:] if poi_position else all_peaks):
            axes[0].axvline(x=pos, color='orange', linestyle=':',
                            linewidth=1, alpha=0.5)
            axes[0].text(pos, axes[0].get_ylim()[0] +
                         (axes[0].get_ylim()[1] -
                          axes[0].get_ylim()[0]) * 0.05,
                         f'{conf:.2f}', rotation=90, va='bottom',
                         fontsize=8, color='orange')

        axes[0].legend(loc='upper right')

        # Plot 2: Raw OP values
        axes[1].plot(window_positions, op_values, 'g-', linewidth=1.5,
                     alpha=0.7, label='Raw OP values')
        axes[1].set_xlabel('Window Position')
        axes[1].set_ylabel('OP Value')
        axes[1].set_title('Overlap Parameter (OP) Predictions - Raw')
        axes[1].set_ylim([0, 1.05])
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=self.peak_threshold, color='red', linestyle='--',
                        alpha=0.5, label=f'Threshold ({self.peak_threshold:.2f})')
        axes[1].legend(loc='upper right')

        # Plot 3: Smoothed OP values with peaks (if smoothing applied)
        if op_smoothed is not None and n_plots > 2:
            axes[2].plot(window_positions, op_smoothed, 'b-', linewidth=1.5,
                         label=f'Smoothed OP (σ={self.gaussian_sigma})')

            # Mark all peaks
            for pos, conf in all_peaks:
                idx = np.argmin(np.abs(window_positions - pos))
                axes[2].scatter(pos, op_smoothed[idx], color='red' if pos == poi_position else 'orange',
                                s=50, zorder=5)

            axes[2].axhline(y=self.peak_threshold, color='red', linestyle='--',
                            alpha=0.5, label=f'Threshold ({self.peak_threshold:.2f})')
            axes[2].set_xlabel('Window Position')
            axes[2].set_ylabel('OP Value')
            axes[2].set_title('Smoothed OP Predictions with Peak Detection')
            axes[2].set_ylim([0, 1.05])
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, FusionDataprocessori=150,
                        bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        if show_plot:
            plt.show()

        # Print summary statistics
        print(f"\n=== Summary Statistics ===")
        print(f"Windows analyzed: {len(window_positions)}")
        print(f"Max raw OP value: {np.max(op_values):.3f}")
        if op_smoothed is not None:
            print(f"Max smoothed OP value: {np.max(op_smoothed):.3f}")
        print(f"Peaks found: {len(all_peaks)}")
        if poi_position is not None:
            print(
                f"Selected POI position: {poi_position} (confidence: {poi_confidence:.3f})")


class MultiPOIPredictor:
    """
    Wrapper class to handle multiple POI predictors for different POI types.
    """

    def __init__(self, model_configs: Dict[str, str], device: str = None):
        """
        Initialize multiple POI predictors.

        Args:
            model_configs: Dictionary mapping POI names to model paths
                          e.g., {'POI1': 'model_poi1.pth', 'POI2': 'model_poi2.pth'}
            device: Device to use for all models
        """
        self.predictors = {}

        for poi_name, model_path in model_configs.items():
            print(f"\nLoading {poi_name} model...")
            self.predictors[poi_name] = V4RegPredictor(
                model_path=model_path,
                poi_type=poi_name,
                device=device
            )

    def predict_all(self,
                    data_path: str = None,
                    df: pd.DataFrame = None,
                    apply_smoothing: bool = True) -> Dict[str, Dict]:
        """
        Run predictions for all POI types.

        Returns:
            Dictionary with POI names as keys and prediction results as values
        """
        results = {}

        # Load data once
        if data_path is not None:
            df = pd.read_csv(data_path)
        elif df is None:
            raise ValueError("Either data_path or df must be provided")

        print("\n" + "="*60)
        print("MULTI-POI PREDICTION")
        print("="*60)

        for poi_name, predictor in self.predictors.items():
            print(f"\nProcessing {poi_name}...")
            results[poi_name] = predictor.predict(
                df=df,
                apply_smoothing=apply_smoothing
            )

        # Summary
        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        detected_pois = {}
        for poi_name, result in results.items():
            if result['poi_position'] is not None:
                detected_pois[poi_name] = {
                    'position': result['poi_position'],
                    'confidence': result['poi_confidence']
                }
                print(f"{poi_name}: Position {result['poi_position']:5d} "
                      f"(Confidence: {result['poi_confidence']:.3f})")
            else:
                print(f"{poi_name}: Not detected")

        return results


# Example usage
if __name__ == "__main__":
    # Single POI predictor
    predictor = V4RegPredictor(
        model_path='fusion/fusion_models/poi_model_xsmall_window_4.pth',  # POI1 model
        poi_type='POI5'
    )

    # Predict from file
    result = predictor.predict(
        data_path='M250505W3_IGG_NOSUC_20C_3rd.csv',
        apply_smoothing=True
    )

    # Access results
    if result['poi_position'] is not None:
        print(f"POI detected at position: {result['poi_position']}")
        print(f"Confidence: {result['poi_confidence']:.3f}")
    else:
        print("No POI detected")

    # Visualize
    predictor.visualize(result, save_path='poi1_prediction.png')
