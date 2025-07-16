import os
import logging
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StopPositionDetector:
    """
    TensorFlow-based system for detecting stop positions from sensor signals.

    Uses Dissipation, Relative_time, and Resonance_Frequency to predict stop positions
    from the 6th column of corresponding _poi.csv files.
    """

    def __init__(self, sequence_length: int = 50, model_type: str = "lstm"):
        """
        Initialize the detector.

        Args:
            sequence_length: Length of input sequences for time series prediction
            model_type: Type of model to use ("lstm", "gru", "cnn", "transformer")
        """
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = ['Dissipation',
                                'Relative_time', 'Resonance_Frequency']

    def load_content(self, data_dir: str, num_datasets: int = np.inf) -> List[Tuple[str, str]]:
        """Load dataset paths and corresponding POI files"""
        if not os.path.exists(data_dir):
            logger.error("Data directory does not exist: %s", data_dir)
            return []

        loaded = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if not f.endswith(".csv") or f.endswith(("_poi.csv", "_lower.csv")):
                    continue
                data_path = os.path.join(root, f)
                poi_path = data_path.replace(".csv", "_poi.csv")
                if os.path.exists(poi_path):
                    loaded.append((data_path, poi_path))

        random.shuffle(loaded)
        if num_datasets == np.inf:
            return loaded
        return loaded[:int(num_datasets)]

    def load_single_dataset(self, data_path: str, poi_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single dataset and its corresponding POI file.

        Returns:
            Tuple of (features, target) where features are the 3 sensor signals
            and target is the 6th column (index 5) from the POI file.
        """
        try:
            # Load main data
            data = pd.read_csv(data_path)

            # Ensure we have the required columns
            if not all(col in data.columns for col in self.feature_columns):
                logger.warning(f"Missing required columns in {data_path}")
                return None, None

            # Load POI data (no header)
            poi_data = pd.read_csv(poi_path, header=None)

            # Check if POI file has at least 6 columns
            if poi_data.shape[1] < 6:
                logger.warning(f"POI file {poi_path} has less than 6 columns")
                return None, None

            # Extract features and target
            features = data[self.feature_columns].values
            target = poi_data.iloc[:, 5].values  # 6th column (0-indexed)

            # Ensure both have the same length
            min_len = min(len(features), len(target))
            features = features[:min_len]
            target = target[:min_len]

            return features, target

        except Exception as e:
            logger.error(f"Error loading {data_path}: {str(e)}")
            return None, None

    def create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.

        Args:
            features: Input features of shape (n_samples, n_features)
            target: Target values of shape (n_samples,)

        Returns:
            Tuple of (X_seq, y_seq) where X_seq has shape (n_sequences, sequence_length, n_features)
        """
        X_seq, y_seq = [], []

        for i in range(len(features) - self.sequence_length):
            X_seq.append(features[i:i + self.sequence_length])
            y_seq.append(target[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def prepare_data(self, data_dir: str, num_datasets: int = np.inf) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare all datasets for training.

        Returns:
            Tuple of (X, y) ready for model training
        """
        dataset_paths = self.load_content(data_dir, num_datasets)

        if not dataset_paths:
            raise ValueError("No valid datasets found")

        all_X, all_y = [], []

        for data_path, poi_path in dataset_paths:
            features, target = self.load_single_dataset(data_path, poi_path)

            if features is None or target is None:
                continue

            # Create sequences
            X_seq, y_seq = self.create_sequences(features, target)

            if len(X_seq) > 0:
                all_X.append(X_seq)
                all_y.append(y_seq)

        if not all_X:
            raise ValueError("No valid sequences could be created")

        # Combine all datasets
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        logger.info(
            f"Prepared {len(X)} sequences from {len(dataset_paths)} datasets")
        logger.info(f"Input shape: {X.shape}, Target shape: {y.shape}")

        return X, y

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build the neural network model.

        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
        """
        model = keras.Sequential()

        if self.model_type == "lstm":
            model.add(layers.LSTM(64, return_sequences=True,
                      input_shape=input_shape))
            model.add(layers.Dropout(0.2))
            model.add(layers.LSTM(32, return_sequences=False))
            model.add(layers.Dropout(0.2))

        elif self.model_type == "gru":
            model.add(layers.GRU(64, return_sequences=True,
                      input_shape=input_shape))
            model.add(layers.Dropout(0.2))
            model.add(layers.GRU(32, return_sequences=False))
            model.add(layers.Dropout(0.2))

        elif self.model_type == "cnn":
            model.add(layers.Conv1D(
                64, 3, activation='relu', input_shape=input_shape))
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Conv1D(32, 3, activation='relu'))
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Flatten())
            model.add(layers.Dropout(0.2))

        elif self.model_type == "transformer":
            # Simple transformer-like architecture
            model.add(layers.Dense(
                64, activation='relu', input_shape=input_shape))
            model.add(layers.MultiHeadAttention(num_heads=4, key_dim=64))
            model.add(layers.GlobalAveragePooling1D())
            model.add(layers.Dropout(0.2))

        # Output layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1))  # Single output for stop position

        return model

    def train(self, data_dir: str, num_datasets: int = np.inf,
              validation_split: float = 0.2, epochs: int = 50,
              batch_size: int = 32, learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train the model on the prepared data.

        Args:
            data_dir: Directory containing the datasets
            num_datasets: Number of datasets to use
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer

        Returns:
            Dictionary containing training history and metrics
        """
        # Prepare data
        X, y = self.prepare_data(data_dir, num_datasets)

        # Normalize features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler_X.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)

        # Normalize target
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=validation_split, random_state=42
        )

        # Build model
        self.model = self.build_model(
            (self.sequence_length, len(self.feature_columns)))

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-6)
        ]

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        val_predictions = self.scaler_y.inverse_transform(val_predictions)
        y_val_original = self.scaler_y.inverse_transform(y_val.reshape(-1, 1))

        val_mse = mean_squared_error(y_val_original, val_predictions)
        val_mae = mean_absolute_error(y_val_original, val_predictions)

        logger.info(f"Validation MSE: {val_mse:.4f}")
        logger.info(f"Validation MAE: {val_mae:.4f}")

        return {
            'history': history.history,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'model_summary': self.model.summary()
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            features: Input features of shape (n_samples, n_features) or
                     (n_sequences, sequence_length, n_features)

        Returns:
            Predicted stop positions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Handle single sample prediction
        if features.ndim == 2 and features.shape[0] >= self.sequence_length:
            # Create sequences from the input
            X_seq, _ = self.create_sequences(
                features, np.zeros(features.shape[0]))
            if len(X_seq) == 0:
                raise ValueError(
                    f"Input too short. Need at least {self.sequence_length} samples.")
        elif features.ndim == 3:
            X_seq = features
        else:
            raise ValueError("Invalid input shape")

        # Scale features
        X_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
        X_scaled = self.scaler_X.transform(X_reshaped)
        X_scaled = X_scaled.reshape(X_seq.shape)

        # Make predictions
        predictions_scaled = self.model.predict(X_scaled)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)

        return predictions.flatten()

    def plot_training_history(self, history: Dict[str, List[float]]):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(history['mae'], label='Training MAE')
        ax2.plot(history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str):
        """Save the trained model and scalers."""
        if self.model is None:
            raise ValueError("No model to save")

        # Save model
        self.model.save(f"{filepath}_model.h5")

        # Save scalers
        import joblib
        joblib.dump(self.scaler_X, f"{filepath}_scaler_X.pkl")
        joblib.dump(self.scaler_y, f"{filepath}_scaler_y.pkl")

        logger.info(f"Model and scalers saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model and scalers."""
        # Load model
        self.model = keras.models.load_model(f"{filepath}_model.h5")

        # Load scalers
        import joblib
        self.scaler_X = joblib.load(f"{filepath}_scaler_X.pkl")
        self.scaler_y = joblib.load(f"{filepath}_scaler_y.pkl")

        logger.info(f"Model and scalers loaded from {filepath}")


# Example usage
def main():
    """Example usage of the StopPositionDetector."""

    # Initialize detector
    detector = StopPositionDetector(
        sequence_length=50,  # Use 50 time steps for prediction
        model_type="lstm"    # Use LSTM model
    )

    # Train the model
    data_dir = "content/static/train"  # Replace with your data directory

    try:
        results = detector.train(
            data_dir=data_dir,
            num_datasets=100,  # Use first 100 datasets
            epochs=50,
            batch_size=32,
            learning_rate=0.001
        )

        # Plot training history
        detector.plot_training_history(results['history'])

        # Save the trained model
        detector.save_model("stop_position_model")

        # Example prediction on new data
        # new_features = np.random.random((100, 3))  # Replace with actual data
        # predictions = detector.predict(new_features)
        # print(f"Predicted stop positions: {predictions}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")


if __name__ == "__main__":
    main()
