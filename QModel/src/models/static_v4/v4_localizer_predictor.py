import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from scipy import signal
from typing import Union, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class V4LocalizerPredictor:
    """
    Standalone predictor for POI localization.
    Loads a trained model and scaler, then makes predictions on input windows.
    """

    def __init__(self,
                 model_path: str = "localizer/v4_localizer.h5",
                 scaler_path: str = "localizer/v4_localizer_scaler.joblib",
                 window_size: int = 128,
                 confidence_threshold: float = 0.7):
        """
        Initialize the predictor with saved model and scaler.

        Args:
            model_path: Path to the saved .h5 model file
            scaler_path: Path to the saved scaler .joblib file
            window_size: Window size used during training (default: 128)
            confidence_threshold: Threshold for high-confidence predictions (default: 0.7)
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold

        # Load model and scaler
        self._load_model(model_path)
        self._load_scaler(scaler_path)

        # Infer feature dimension from model input shape
        # Subtract advanced features
        self.base_feature_dim = self.model.input_shape[-1] - 8

        print(f"Predictor initialized successfully")
        print(f"Model input shape: {self.model.input_shape}")
        print(f"Base feature dimension: {self.base_feature_dim}")

    def _load_model(self, model_path: str):
        """Load the trained Keras model."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Custom objects for loading (if needed for custom layers/losses)
        custom_objects = {}

        try:
            self.model = tf.keras.models.load_model(
                model_path, custom_objects=custom_objects)
            print(f"Model loaded from: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _load_scaler(self, scaler_path: str):
        """Load the fitted scaler."""
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        try:
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from: {scaler_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load scaler: {e}")

    def create_advanced_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create advanced signal processing features.
        Must match the training pipeline exactly.

        Args:
            X: Input array of shape (batch_size, window_size, n_features)

        Returns:
            Enhanced array with additional features
        """
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
                    peak_indicator[peaks] = properties.get(
                        'prominences', np.ones(len(peaks)))

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

    def preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """
        Preprocess a single window for prediction.

        Args:
            window: Input array of shape (window_size, n_features) or (batch_size, window_size, n_features)

        Returns:
            Preprocessed window ready for prediction
        """
        # Handle both single window and batch input
        if window.ndim == 2:
            window = np.expand_dims(window, axis=0)

        batch_size, seq_len, n_features = window.shape

        # Validate window size
        if seq_len != self.window_size:
            raise ValueError(
                f"Expected window size {self.window_size}, got {seq_len}")

        # Apply scaler normalization
        window_reshaped = window.reshape(-1, n_features)
        window_normalized = self.scaler.transform(
            window_reshaped).astype(np.float32)
        window_normalized = window_normalized.reshape(
            batch_size, seq_len, n_features)

        # Create advanced features
        window_enhanced = self.create_advanced_features(window_normalized)

        return window_enhanced

    def predict(self,
                window: np.ndarray,
                return_confidence: bool = True,
                return_distribution: bool = False) -> Union[int, Tuple]:
        """
        Make a prediction on a single window.

        Args:
            window: Input array of shape (window_size, n_features)
            return_confidence: Whether to return confidence score
            return_distribution: Whether to return full probability distribution

        Returns:
            Predicted position (0-indexed within window)
            Optionally: (position, confidence) or (position, confidence, distribution)
        """
        # Preprocess
        X = self.preprocess_window(window)

        # Predict
        position_pred, confidence_pred = self.model.predict(X, verbose=0)

        # Get position
        position = int(np.argmax(position_pred[0]))
        confidence = float(confidence_pred[0, 0])

        # Prepare return values
        if return_distribution:
            return position, confidence, position_pred[0]
        elif return_confidence:
            return position, confidence
        else:
            return position

    def predict_batch(self,
                      windows: np.ndarray,
                      return_confidence: bool = True) -> Dict[str, np.ndarray]:
        """
        Make predictions on a batch of windows.

        Args:
            windows: Input array of shape (batch_size, window_size, n_features)
            return_confidence: Whether to return confidence scores

        Returns:
            Dictionary with 'positions', 'confidences', and 'high_confidence_mask'
        """
        # Preprocess
        X = self.preprocess_window(windows)

        # Predict
        position_pred, confidence_pred = self.model.predict(X, verbose=0)

        # Get positions and confidences
        positions = np.argmax(position_pred, axis=1)
        confidences = confidence_pred.squeeze()

        # High confidence mask
        high_conf_mask = confidences > self.confidence_threshold

        results = {
            'positions': positions,
            'confidences': confidences,
            'high_confidence_mask': high_conf_mask
        }

        return results

    def predict_with_uncertainty(self,
                                 window: np.ndarray,
                                 n_samples: int = 10) -> Tuple[int, float, float]:
        """
        Make prediction with uncertainty estimation using dropout sampling.

        Args:
            window: Input array of shape (window_size, n_features)
            n_samples: Number of forward passes for uncertainty estimation

        Returns:
            (predicted_position, mean_confidence, position_std)
        """
        # Preprocess once
        X = self.preprocess_window(window)

        # Multiple predictions with dropout enabled
        positions = []
        confidences = []

        for _ in range(n_samples):
            position_pred, confidence_pred = self.model(
                X, training=True)  # Enable dropout
            pos = int(np.argmax(position_pred[0]))
            conf = float(confidence_pred[0, 0])
            positions.append(pos)
            confidences.append(conf)

        # Calculate statistics
        positions = np.array(positions)
        confidences = np.array(confidences)

        # Most common position (mode)
        predicted_position = int(np.median(positions))  # or use mode
        mean_confidence = np.mean(confidences)
        position_std = np.std(positions)

        return predicted_position, mean_confidence, position_std

    def analyze_prediction(self,
                           window: np.ndarray,
                           true_position: Optional[int] = None) -> Dict:
        """
        Detailed analysis of a prediction.

        Args:
            window: Input array of shape (window_size, n_features)
            true_position: Optional ground truth position for error calculation

        Returns:
            Dictionary with detailed prediction analysis
        """
        # Get full prediction details
        position, confidence, distribution = self.predict(
            window, return_confidence=True, return_distribution=True
        )

        # Calculate entropy of distribution
        entropy = -np.sum(distribution * np.log(distribution + 1e-10))

        # Find top-k positions
        top_k = 5
        top_k_indices = np.argsort(distribution)[-top_k:][::-1]
        top_k_probs = distribution[top_k_indices]

        # Peak properties
        max_prob = np.max(distribution)
        peak_width = np.sum(distribution > max_prob *
                            0.5)  # Width at half maximum

        analysis = {
            'predicted_position': position,
            'confidence': confidence,
            'max_probability': max_prob,
            'entropy': entropy,
            'peak_width': peak_width,
            'top_k_positions': top_k_indices.tolist(),
            'top_k_probabilities': top_k_probs.tolist(),
            'is_high_confidence': confidence > self.confidence_threshold
        }

        # Add error if ground truth provided
        if true_position is not None:
            error = abs(position - true_position)
            analysis['true_position'] = true_position
            analysis['error'] = error
            analysis['is_correct'] = (error == 0)
            analysis['within_1'] = (error <= 1)
            analysis['within_3'] = (error <= 3)
            analysis['within_5'] = (error <= 5)

        return analysis

    def visualize_prediction(self,
                             window: np.ndarray,
                             true_position: Optional[int] = None,
                             feature_names: Optional[list] = None):
        """
        Visualize the prediction with matplotlib.
        Note: Requires matplotlib to be installed.

        Args:
            window: Input array of shape (window_size, n_features)
            true_position: Optional ground truth position
            feature_names: Optional list of feature names
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "matplotlib is required for visualization. Install with: pip install matplotlib")
            return

        # Get prediction
        position, confidence, distribution = self.predict(
            window, return_confidence=True, return_distribution=True
        )

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Plot 1: Input features
        ax1 = axes[0]
        for i in range(window.shape[1]):
            label = feature_names[i] if feature_names else f"Feature {i}"
            ax1.plot(window[:, i], label=label, alpha=0.7)
        ax1.axvline(x=position, color='red', linestyle='--',
                    label=f'Predicted: {position}')
        if true_position is not None:
            ax1.axvline(x=true_position, color='green',
                        linestyle='--', label=f'True: {true_position}')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Value')
        ax1.set_title(f'Input Window (Confidence: {confidence:.3f})')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Probability distribution
        ax2 = axes[1]
        ax2.bar(range(len(distribution)),
                distribution, color='blue', alpha=0.6)
        ax2.axvline(x=position, color='red', linestyle='--',
                    label=f'Predicted: {position}')
        if true_position is not None:
            ax2.axvline(x=true_position, color='green',
                        linestyle='--', label=f'True: {true_position}')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Probability')
        ax2.set_title('Position Probability Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Top-k positions
        ax3 = axes[2]
        analysis = self.analyze_prediction(window, true_position)
        top_k_pos = analysis['top_k_positions']
        top_k_prob = analysis['top_k_probabilities']

        colors = ['red' if p == position else 'blue' for p in top_k_pos]
        ax3.barh(range(len(top_k_pos)), top_k_prob, color=colors, alpha=0.7)
        ax3.set_yticks(range(len(top_k_pos)))
        ax3.set_yticklabels([f'Pos {p}' for p in top_k_pos])
        ax3.set_xlabel('Probability')
        ax3.set_title(f'Top-5 Positions (Entropy: {analysis["entropy"]:.3f})')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig


# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = V4LocalizerPredictor(
        model_path="localizer/v4_localizer.h5",
        scaler_path="localizer/v4_localizer_scaler.joblib",
        window_size=128,
        confidence_threshold=0.7
    )

    # Example: Create dummy data for testing
    # In real usage, this would be your actual feature data
    dummy_window = np.random.randn(128, 10).astype(
        np.float32)  # Adjust feature dim as needed

    # Single prediction
    position, confidence = predictor.predict(dummy_window)
    print(
        f"\nSingle prediction: Position={position}, Confidence={confidence:.3f}")

    # Batch prediction
    batch_windows = np.random.randn(5, 128, 10).astype(np.float32)
    results = predictor.predict_batch(batch_windows)
    print(f"\nBatch predictions:")
    print(f"Positions: {results['positions']}")
    print(f"Confidences: {results['confidences']}")
    print(f"High confidence mask: {results['high_confidence_mask']}")

    # Detailed analysis
    analysis = predictor.analyze_prediction(dummy_window, true_position=64)
    print(f"\nDetailed analysis:")
    for key, value in analysis.items():
        if isinstance(value, list):
            print(f"  {key}: {value[:3]}...")  # Show first 3 items
        else:
            print(f"  {key}: {value}")

    # Uncertainty estimation (optional)
    pos_uncertain, conf_uncertain, std_uncertain = predictor.predict_with_uncertainty(
        dummy_window, n_samples=10
    )
    print(f"\nUncertainty estimation:")
    print(f"  Position: {pos_uncertain} ± {std_uncertain:.2f}")
    print(f"  Mean confidence: {conf_uncertain:.3f}")
