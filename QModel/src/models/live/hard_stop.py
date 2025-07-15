from keras import mixed_precision
import tensorflow_addons as tfa
import os
import logging
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import deque
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, callbacks, metrics, Model
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
# tf.config.optimizer.set_jit(True)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


@dataclass
class Config:
    """Configuration for the stop detection pipeline"""
    window_size: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.2
    positive_weight: float = 10.0  # Weight for positive class (stop position)


class DataLoader:
    """Data loading and preprocessing utilities"""

    @staticmethod
    def load_content(data_dir: str, num_datasets: int = np.inf) -> List[Tuple[str, str]]:
        """Load dataset paths and corresponding POI files"""
        if not os.path.exists(data_dir):
            logging.error("Data directory does not exist: %s", data_dir)
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

    @staticmethod
    def load_single_dataset(data_path: str, poi_path: str) -> Tuple[pd.DataFrame, int]:
        """Load a single dataset and its stop position"""
        try:
            # Load main data
            data = pd.read_csv(data_path)
            cols_to_keep = ['Dissipation',
                            'Relative_time', 'Resonance_Frequency']
            data = data[cols_to_keep]
            # Load POI data (6 indices, 6th is stop position)
            poi_data = pd.read_csv(poi_path, header=None)
            # 6th index (0-based: index 5)
            stop_position = int(poi_data.iloc[5, 0])

            return data, stop_position
        except Exception as e:
            logging.error(f"Error loading dataset {data_path}: {e}")
            return None, None


class FeatureEngineer:
    """Feature engineering for time series data"""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names = []

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw data"""
        features = data.copy()

        # Assume columns are: Relative_time, Dissipation, Resonance_Frequency
        cols = ['Relative_time', 'Dissipation', 'Resonance_Frequency']

        # Ensure we have the expected columns
        if not all(col in features.columns for col in cols):
            # If columns don't match, use first 3 columns
            cols = features.columns[:3].tolist()

        # Rate of change (derivatives)
        for col in cols:
            features[f'{col}_diff'] = features[col].diff().fillna(0)
            features[f'{col}_diff2'] = features[f'{col}_diff'].diff().fillna(0)

        # Moving averages
        for window in [5, 10, 20]:
            for col in cols:
                features[f'{col}_ma_{window}'] = features[col].rolling(
                    window=window).mean().fillna(features[col])
                features[f'{col}_std_{window}'] = features[col].rolling(
                    window=window).std().fillna(0)

        # Ratios and interactions
        if len(cols) >= 2:
            features['dissipation_rf_ratio'] = features[cols[1]] / \
                (features[cols[2]] + 1e-8)
            features['dissipation_time_interaction'] = features[cols[1]
                                                                ] * features[cols[0]]

        # Time-based features
        features['time_normalized'] = (features[cols[0]] - features[cols[0]].min()) / (
            features[cols[0]].max() - features[cols[0]].min() + 1e-8)

        # Store feature names
        self.feature_names = features.columns.tolist()

        return features

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit scaler and transform data"""
        features = self.create_features(data)
        scaled = self.scaler.fit_transform(features)
        self.fitted = True
        return scaled

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler"""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        features = self.create_features(data)
        return self.scaler.transform(features)

    @property
    def n_features(self) -> int:
        """Number of features after transformation"""
        return len(self.feature_names)


def create_stop_detection_model(input_shape: Tuple[int, int], config: Config) -> Model:
    """
    Faster Conv1D → LSTM model for stop detection.
    - Conv1D front‑end with pooling to shrink sequence length.
    - Single unidirectional LSTM (no recurrent_dropout) for GPU‑optimized CuDNN.
    - GlobalMaxPool for speed instead of attention.
    """
    inputs = layers.Input(
        shape=input_shape, name="sequence_input")  # (window_size, n_features)

    # 1) Conv1D feature extractor
    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1"
    )(inputs)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(
        x)             # halves time steps

    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv2"
    )(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(
        x)             # quarters original length

    # 2) Single LSTM layer (GPU‑fast, no recurrent_dropout)
    x = layers.LSTM(
        config.hidden_size,
        return_sequences=False,
        dropout=config.dropout,
        name="lstm"
    )(x)

    # 3) Dense head
    x = layers.Dense(
        config.hidden_size // 2,
        activation="relu",
        name="dense_1"
    )(x)
    x = layers.Dropout(config.dropout, name="dropout")(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="output"
    )(x)

    return Model(inputs=inputs, outputs=outputs, name="stop_detection_cnn_lstm")


class StopDetectionPipeline:
    """Complete pipeline for stop position detection"""

    def __init__(self, config: Config):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.model = None
        self.history = None

    def create_sequences(self, data: np.ndarray, stop_position: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences and labels"""
        sequences = []
        labels = []

        for i in range(self.config.window_size, len(data)):
            # Create sequence
            sequence = data[i - self.config.window_size:i]
            sequences.append(sequence)

            # Create label (1 if current position is stop, 0 otherwise)
            # Add some tolerance around the stop position
            tolerance = 10
            if abs(i - stop_position) <= tolerance:
                labels.append(1)
            else:
                labels.append(0)

        return np.array(sequences), np.array(labels)

    def prepare_data(self, data_dir: str, num_datasets: int = np.inf) -> Dict:
        """Prepare data for training"""
        print("Loading datasets...")
        dataset_paths = DataLoader.load_content(data_dir, num_datasets)

        all_sequences = []
        all_labels = []

        for data_path, poi_path in tqdm(dataset_paths, desc="Processing datasets"):
            data, stop_position = DataLoader.load_single_dataset(
                data_path, poi_path)
            if data is None:
                continue

            # Feature engineering
            if not self.feature_engineer.fitted:
                features = self.feature_engineer.fit_transform(data)
            else:
                features = self.feature_engineer.transform(data)

            # Create sequences
            sequences, labels = self.create_sequences(features, stop_position)
            all_sequences.append(sequences)
            all_labels.append(labels)

        # Concatenate all sequences
        X = np.concatenate(all_sequences, axis=0)
        y = np.concatenate(all_labels, axis=0)

        print(f"Total sequences: {len(X)}")
        print(f"Positive samples: {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
        print(f"Sequence shape: {X.shape}")

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_split,
            stratify=y,
            random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config.validation_split /
            (1 - self.config.test_split),
            stratify=y_temp,
            random_state=42
        )
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]

        # upsample positives to match negatives
        up_pos_idx = np.random.choice(pos_idx, size=len(neg_idx), replace=True)
        all_idx = np.concatenate([neg_idx, up_pos_idx])
        np.random.shuffle(all_idx)

        X_train = X_train[all_idx]
        y_train = y_train[all_idx]

        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }

    def train(self, data_splits: Dict):
        """Train the model"""
        X_train, y_train = data_splits['train']
        X_val, y_val = data_splits['val']

        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")

        # Create model
        # (window_size, n_features)
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = create_stop_detection_model(input_shape, self.config)

        # Compile model with weighted loss
        class_weights = {0: 1.0, 1: self.config.positive_weight}
        loss_fn = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=loss_fn,
            # loss='binary_crossentropy',
            metrics=['accuracy',
                     metrics.Precision(name='precision'),
                     metrics.Recall(name='recall')]
        )

        # Print model summary
        self.model.summary()

        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.patience,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        model_checkpoint = callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.num_epochs,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )

        # Plot training history
        self.plot_training_history()

    def plot_training_history(self):
        """Plot training and validation metrics"""
        if self.history is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'],
                        label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'],
                        label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'],
                        label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        axes[1, 0].plot(self.history.history['precision'],
                        label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'],
                        label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Recall
        axes[1, 1].plot(self.history.history['recall'],
                        label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'],
                        label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def evaluate(self, data_splits: Dict) -> Dict:
        """Evaluate the model"""
        X_test, y_test = data_splits['test']

        # Make predictions
        y_pred_proba = self.model.predict(
            X_test, batch_size=self.config.batch_size)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_pred_proba = y_pred_proba.flatten()

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )

        auc_score = roc_auc_score(y_test, y_pred_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report = classification_report(
            y_test, y_pred, target_names=['Normal', 'Stop'])

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Stop'],
                    yticklabels=['Normal', 'Stop'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Plot ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

        results = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'confusion_matrix': cm,
            'classification_report': report
        }

        print(f"Test Results:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC: {auc_score:.4f}")
        print(f"\nDetailed Classification Report:")
        print(report)

        return results

    def save_model(self, model_path: str = 'stop_detection_model.h5'):
        """Save the trained model"""
        if self.model is not None:
            # Save the full model
            self.model.save(model_path)

            # Also save just the weights as a backup
            weights_path = model_path.replace('.h5', '_weights.h5')
            self.model.save_weights(weights_path)

            # Save model config
            config_path = model_path.replace('.h5', '_config.json')
            with open(config_path, 'w') as f:
                import json
                model_config = {
                    # Remove batch dimension
                    'input_shape': self.model.input_shape[1:],
                    'config': {
                        'window_size': self.config.window_size,
                        'hidden_size': self.config.hidden_size,
                        'num_layers': self.config.num_layers,
                        'dropout': self.config.dropout
                    }
                }
                json.dump(model_config, f)

            print(f"Model saved to {model_path}")
            print(f"Weights saved to {weights_path}")
            print(f"Config saved to {config_path}")
        else:
            print("No model to save. Train the model first.")

    def load_model(self, model_path: str = 'stop_detection_model.h5'):
        """Load a pre-trained model"""
        try:
            # Try to load the full model first
            custom_objects = {
                'precision': metrics.Precision(),
                'recall': metrics.Recall()
            }

            self.model = keras.models.load_model(
                model_path, custom_objects=custom_objects)
            print(f"Model loaded from {model_path}")

        except Exception as e:
            print(f"Failed to load full model: {e}")
            print("Trying to load from weights...")

            # Try to load from weights and config
            try:
                weights_path = model_path.replace('.h5', '_weights.h5')
                config_path = model_path.replace('.h5', '_config.json')

                with open(config_path, 'r') as f:
                    import json
                    saved_config = json.load(f)

                # Recreate model
                temp_config = Config()
                temp_config.window_size = saved_config['config']['window_size']
                temp_config.hidden_size = saved_config['config']['hidden_size']
                temp_config.num_layers = saved_config['config']['num_layers']
                temp_config.dropout = saved_config['config']['dropout']

                self.model = create_stop_detection_model(
                    saved_config['input_shape'], temp_config)
                self.model.load_weights(weights_path)

                print(f"Model loaded from weights: {weights_path}")

            except Exception as e2:
                print(f"Failed to load from weights: {e2}")
                raise e2


class RealTimeStopDetector:
    """Real-time stop position detector"""

    def __init__(self, model_path: str, feature_engineer: FeatureEngineer, config: Config):
        self.config = config
        self.feature_engineer = feature_engineer
        self.buffer = deque(maxlen=config.window_size)

        # Load model with fallback to weights
        try:
            # Try to load the full model first
            custom_objects = {
                'precision': metrics.Precision(),
                'recall': metrics.Recall()
            }

            self.model = keras.models.load_model(
                model_path, custom_objects=custom_objects)
            print(f"Real-time detector loaded from {model_path}")

        except Exception as e:
            print(f"Failed to load full model: {e}")
            print("Trying to load from weights...")

            # Try to load from weights and config
            weights_path = model_path.replace('.h5', '_weights.h5')
            config_path = model_path.replace('.h5', '_config.json')

            with open(config_path, 'r') as f:
                import json
                saved_config = json.load(f)

            # Recreate model
            temp_config = Config()
            temp_config.window_size = saved_config['config']['window_size']
            temp_config.hidden_size = saved_config['config']['hidden_size']
            temp_config.num_layers = saved_config['config']['num_layers']
            temp_config.dropout = saved_config['config']['dropout']

            self.model = create_stop_detection_model(
                saved_config['input_shape'], temp_config)
            self.model.load_weights(weights_path)

            print(f"Real-time detector loaded from weights: {weights_path}")

        self.consecutive_predictions = deque(maxlen=5)
        self.prediction_history = []

    def update(self, new_measurement: pd.DataFrame) -> Tuple[float, bool]:
        """Update with new measurement and return stop probability"""
        # Transform the measurement
        features = self.feature_engineer.transform(new_measurement)

        # Add to buffer
        self.buffer.extend(features)

        if len(self.buffer) < self.config.window_size:
            return 0.0, False

        # Prepare input
        sequence = np.array(self.buffer).reshape(
            1, self.config.window_size, -1)

        # Predict
        probability = self.model.predict(sequence, verbose=0)[0, 0]

        # Store prediction history
        self.prediction_history.append(probability)

        # Decision logic with consecutive predictions
        self.consecutive_predictions.append(probability > 0.5)

        # Require 3 out of 5 consecutive predictions to be positive
        if len(self.consecutive_predictions) >= 3:
            stop_detected = sum(self.consecutive_predictions) >= 3
        else:
            stop_detected = False

        return float(probability), stop_detected

    def reset(self):
        """Reset the detector state"""
        self.buffer.clear()
        self.consecutive_predictions.clear()
        self.prediction_history.clear()

    def get_prediction_history(self) -> List[float]:
        """Get the history of predictions"""
        return self.prediction_history.copy()

    def plot_predictions(self, last_n: int = 100):
        """Plot recent predictions"""
        if not self.prediction_history:
            print("No predictions to plot")
            return

        recent_predictions = self.prediction_history[-last_n:]

        plt.figure(figsize=(12, 6))
        plt.plot(recent_predictions, label='Stop Probability')
        plt.axhline(y=0.5, color='r', linestyle='--',
                    label='Decision Threshold')
        plt.xlabel('Time Step')
        plt.ylabel('Stop Probability')
        plt.title('Real-time Stop Detection Predictions')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage and testing


def main():
    # Configuration
    config = Config(
        window_size=100,
        batch_size=32,
        learning_rate=0.001,
        num_epochs=10,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        patience=10,
        positive_weight=10.0
    )

    # Initialize pipeline
    pipeline = StopDetectionPipeline(config)

    # Prepare data
    data_dir = "content/dropbox_dump"  # Replace with your data directory
    data_splits = pipeline.prepare_data(data_dir, num_datasets=100)

    # Train model
    pipeline.train(data_splits)

    # Evaluate model
    results = pipeline.evaluate(data_splits)

    # Save model and feature engineer
    pipeline.save_model('stop_detection_model.h5')

    import pickle
    with open('feature_engineer.pkl', 'wb') as f:
        pickle.dump(pipeline.feature_engineer, f)

    print("Pipeline completed successfully!")

    # Example real-time usage
    print("\nTesting real-time detector...")
    detector = RealTimeStopDetector(
        'stop_detection_model.h5',
        pipeline.feature_engineer,
        config
    )

    # Simulate real-time data streaming
    X_test, y_test = data_splits['test']

    # Test on a sample sequence
    sample_idx = 0
    sample_sequence = X_test[sample_idx]

    detector.reset()

    # Simulate streaming each timestep
    for i in range(len(sample_sequence)):
        # Create a DataFrame with the features (simulate new measurement)
        timestep_data = pd.DataFrame([sample_sequence[i]],
                                     columns=pipeline.feature_engineer.feature_names)

        prob, stop_detected = detector.update(timestep_data)

        if stop_detected:
            print(f"Stop detected at timestep {i} with probability {prob:.3f}")

    # Plot the prediction history
    detector.plot_predictions()


if __name__ == "__main__":
    main()
