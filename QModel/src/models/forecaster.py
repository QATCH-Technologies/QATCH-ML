from typing import List, Dict, Tuple, Optional
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
from collections import deque
import numpy as np
import logging
import tensorflow as tf
from keras.layers import Input, Masking, Bidirectional, LSTM, BatchNormalization, Dropout, TimeDistributed, Dense, GRU
from keras.models import Model
from keras.optimizers import Adam
import logging
from typing import Optional, List, Tuple, Dict
from sklearn.model_selection import train_test_split
from forecaster_data_processor import DataProcessor
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict, deque
import random
# '2' to filter out warnings and info messages; use '3' to show only errors.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
TEST_SIZE = 0.2
RANDOM_STATE = 42

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logging.info(f"Hardware detected: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    logging.info("No GPU detected, running on CPU.")


# Assume that DataProcessor, train_test_split, TEST_SIZE, and RANDOM_STATE are defined elsewhere.

# Assume that DataProcessor, train_test_split, TEST_SIZE, and RANDOM_STATE are defined elsewhere.


class ForcasterTrainer:
    def __init__(self, classes: List, num_features: int) -> None:
        """
        Args:
            classes (List[str]): A list of class labels.
            num_features (int): The number of features in the processed data.
        """
        if not isinstance(classes, list):
            raise TypeError("classes must be a list")
        if not isinstance(num_features, int):
            raise TypeError("num_features must be an integer")

        self.classes: List = classes
        self.num_features: int = num_features
        self.model = self.build_model()  # Build GPU-enabled XGBoost model
        self.label_to_index: Dict[str, int] = {
            label: idx for idx, label in enumerate(classes)}
        self.content: Optional[List[Tuple[str, str]]] = None
        self.train_content: Optional[List[Tuple[str, str]]] = None
        self.test_content: Optional[List[Tuple[str, str]]] = None
        self.validation_content: Optional[List[Tuple[str, str]]] = None

        # Toggle flag for live plotting (default is off)
        self.live_plot_enabled: bool = False

    def toggle_live_plot(self, enabled: bool = True):
        """Toggle live plotting on or off."""
        self.live_plot_enabled = enabled

    def build_model(self):
        """
        Build an XGBoost classifier configured to use CUDA.
        For multi-class classification, we use 'multi:softprob' as the objective.
        The GPU acceleration is enabled by setting tree_method and predictor.
        """
        model = XGBClassifier(
            objective='multi:softprob',
            num_class=len(self.classes),
            n_estimators=50,      # Adjust the number of trees as needed
            learning_rate=0.1,
            eval_metric='mlogloss',
            tree_method='hist',      # Use GPU accelerated histogram algorithm
            device='cuda'   # Use GPU for prediction
        )
        return model

    def load_datasets(self, training_directory: str, validation_directory: str):
        """
        Loads and splits the training data.
        """
        self.content = DataProcessor.load_content(
            training_directory, num_datasets=150)
        self.train_content, self.test_content = train_test_split(
            self.content, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
        self.validation_content = DataProcessor.load_content(
            validation_directory, num_datasets=10)
        logging.info("Datasets loaded.")

    def get_slices(self, data_file: str, poi_file: str, slice_size: int = 100):
        data_df = pd.read_csv(data_file)
        start_idx = 0
        total_rows = len(data_df)
        slices = []
        while start_idx < total_rows:
            end_idx = min(start_idx + slice_size, total_rows)
            current_slice = data_df.iloc[start_idx:end_idx]
            processed_slice = DataProcessor.preprocess_data(
                current_slice, poi_file)
            start_idx = end_idx
            if processed_slice is None:
                continue
            slices.append(processed_slice)
        return slices

    def train_on_slice(self, data_slice):
        """
        Train the model on one slice of data with dynamic class weighting.
        Ensures that the feature matrix and label vector have matching numbers of rows.
        """
        y = data_slice['Fill']
        X = data_slice.drop(columns=['Fill'])

        # Map labels to indices if necessary
        y = y.map(lambda x: self.label_to_index.get(x, x))
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        # Check that the number of samples in features and labels match
        if X_np.shape[0] != y_np.shape[0]:
            logging.error(
                f"Mismatch between features and labels: X has {X_np.shape[0]} samples but y has {y_np.shape[0]} samples. Skipping this slice.")
            return None

        if X_np.shape[0] < 2:
            return None  # Not enough data

        # Compute dynamic class weights
        unique, counts = np.unique(y_np, return_counts=True)
        total_samples = sum(counts)
        class_weights = {cls: total_samples /
                         (len(unique) * count) for cls, count in zip(unique, counts)}
        sample_weights = np.array([class_weights[val] for val in y_np])

        try:
            if not hasattr(self, "initial_fit_done") or not self.initial_fit_done:
                # First time: perform a normal fit without incremental training
                self.model.fit(X_np, y_np, sample_weight=sample_weights)
                self.initial_fit_done = True
            else:
                # Subsequent times: perform incremental training using the current booster
                self.model.fit(X_np, y_np,
                               sample_weight=sample_weights,
                               xgb_model=self.model.get_booster())
        except Exception as e:
            logging.error(f"Error training on slice: {e}")
            return None

        # Evaluate on the current slice
        y_pred_proba = self.model.predict_proba(X_np)
        loss = log_loss(y_np, y_pred_proba)
        y_pred = self.model.predict(X_np)
        accuracy = accuracy_score(y_np, y_pred)

        return loss, accuracy

    def moving_average(self, data, window_size=20):
        """Compute the moving average of a list of numbers."""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def train(self, training_directory: str = r'content/training_data/full_fill',
              validation_directory: str = r'content/test_data'):
        self.load_datasets(training_directory=training_directory,
                           validation_directory=validation_directory)

        if self.live_plot_enabled:
            plt.ion()
            fig, (ax_loss, ax_acc) = plt.subplots(
                nrows=2, ncols=1, figsize=(12, 10))
            losses = []
            accuracies = []
            sma_window = 20

        # Use a buffer to store recent slices
        buffer_size = 10
        slice_buffer = deque(maxlen=buffer_size)

        for i, (data_file, poi_file) in enumerate(self.train_content):
            logging.info(
                f'Processing dataset {i + 1}/{len(self.train_content)}.')
            slices = self.get_slices(data_file, poi_file)
            if slices is None:
                continue

            for processed_slice in slices:
                slice_buffer.append(processed_slice)
                # Randomly sample slices from the buffer
                sampled_slices = random.sample(
                    slice_buffer, min(len(slice_buffer), 3))
                for sample in sampled_slices:
                    result = self.train_on_slice(sample)
                    if result is None:
                        continue
                    batch_loss, batch_accuracy = result

                    if self.live_plot_enabled:
                        losses.append(batch_loss)
                        accuracies.append(batch_accuracy)
                        smooth_losses = self.moving_average(losses, sma_window)
                        smooth_accuracies = self.moving_average(
                            accuracies, sma_window)

                        ax_loss.clear()
                        ax_loss.plot(losses, label='Training Loss (raw)')
                        if len(losses) >= sma_window:
                            ax_loss.plot(range(sma_window - 1, len(losses)), smooth_losses,
                                         label='Training Loss (smoothed)')
                        ax_loss.set_title("Training Loss")
                        ax_loss.set_xlabel("Batch Number")
                        ax_loss.set_ylabel("Loss")
                        ax_loss.legend()

                        ax_acc.clear()
                        ax_acc.plot(
                            accuracies, label='Training Accuracy (raw)')
                        if len(accuracies) >= sma_window:
                            ax_acc.plot(range(sma_window - 1, len(accuracies)), smooth_accuracies,
                                        label='Training Accuracy (smoothed)')
                        ax_acc.set_title("Training Accuracy")
                        ax_acc.set_xlabel("Batch Number")
                        ax_acc.set_ylabel("Accuracy")
                        ax_acc.legend()

                        plt.pause(0.001)

        logging.info("Training completed.")
        if self.live_plot_enabled:
            plt.ioff()
            plt.show()

    def test(self):
        """
        Evaluate the model on the validation data.
        """
        logging.info("Starting testing on validation data.")
        total_loss = 0.0
        total_accuracy = 0.0
        count = 0

        for data_file, poi_file in self.validation_content:
            slices = self.get_slices(data_file, poi_file)
            for processed_slice in slices:
                y = processed_slice['Fill']
                X = processed_slice.drop(columns=['Fill'])
                y = y.map(lambda x: self.label_to_index.get(x, x))
                X_np = X.to_numpy()
                y_np = y.to_numpy()
                if len(X_np) < 2:
                    continue

                y_pred_proba = self.model.predict_proba(X_np)
                loss = log_loss(y_np, y_pred_proba)
                y_pred = self.model.predict(X_np)
                accuracy = accuracy_score(y_np, y_pred)

                total_loss += loss
                total_accuracy += accuracy
                count += 1

        if count > 0:
            avg_loss = total_loss / count
            avg_accuracy = total_accuracy / count
            logging.info(
                f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")
            print(
                f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {avg_accuracy:.4f}")
        else:
            logging.warning("No valid slices found in the validation data.")

    def save_model(self, save_path: str):
        """
        Save the current model to the specified path.
        """
        try:
            # Save the model using XGBoost's built-in save_model method.
            self.model.save_model(save_path)
            logging.info(f"Model successfully saved to {save_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")


class Forecaster:
    def __init__(self):
        self.accumulated_buffer = None
        self.feature_vector = None
        self.model = None

    def extend_buffer(self, new_buffer: pd.DataFrame):
        if self.accumulated_buffer is None:
            self.accumulated_buffer = pd.DataFrame(columns=new_buffer.columns)
        self.accumulated_buffer = new_buffer
        self.feature_vector = DataProcessor.generate_features(
            self.accumulated_buffer, live=True)
        if 'Relative_time' in self.feature_vector.columns:
            self.feature_vector.drop(columns=['Relative_time'], inplace=True)

    def reset_buffer(self):
        self.accumulated_buffer = pd.DataFrame()

    def load_model(self, model_path: str):
        """
        Loads a pre-trained model from the specified path.

        Args:
            model_path (str): The file path from which the model will be loaded.
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            logging.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def predict_batch(self, batch_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on a batch of data.

        The method first extends the internal buffer with the new data,
        generates a feature vector using the DataProcessor, reshapes the data
        as required by the model, and then makes predictions.

        Args:
            batch_data (pd.DataFrame): A batch of new data to predict on.

        Returns:
            np.ndarray: Predicted class indices for the input sequence.
        """
        if self.model is None:
            raise ValueError(
                "Model not loaded. Please load a model before predicting.")

        # Update the internal buffer and regenerate the feature vector
        self.extend_buffer(batch_data)
        if self.feature_vector is None or self.feature_vector.empty:
            raise ValueError(
                "Feature vector is empty. Ensure that DataProcessor returns valid features.")

        # Convert the feature vector to a numpy array and reshape it for prediction.
        X = self.feature_vector.to_numpy()
        # Reshape to add the batch dimension (batch_size, sequence_length, num_features)
        X = X.reshape(1, X.shape[0], X.shape[1])

        # Use the model to predict and extract class predictions via argmax.
        predictions = self.model.predict(X)
        print(predictions)
        predicted_classes = np.argmax(predictions, axis=-1)
        return predicted_classes


TRAINING = True
TESTING = False

if __name__ == "__main__":
    if TRAINING:
        with tf.device('/GPU:0'):
            ft = ForcasterTrainer(num_features=16, classes=[0, 1, 2, 3, 4, 5])
            ft.toggle_live_plot(True)  # Enable live plotting
            # ft.tune()
            ft.train()
            ft.test()
            ft.save_model(save_path=r'QModel/SavedModels/tf_forecaster')

    if TESTING:
        # Initialize the Forecaster and load the saved model.
        f = Forecaster()
        try:
            f.load_model(model_path=r'QModel/SavedModels/tf_forecaster')
        except Exception as e:
            logging.error(f"Could not load model: {e}")
            exit(1)

        # Load the entire test data file.
        test_data_file = r'content/test_data/01037/W10_BUFFER_I10_3rd.csv'
        try:
            full_data = pd.read_csv(test_data_file)
        except Exception as e:
            logging.error(f"Error reading test data file: {e}")
            exit(1)

        full_data = full_data.iloc[::5]

        # Set up interactive plotting.
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 6))

        # Generate slices of increasing size (100 rows, 200 rows, etc.)
        for end_idx in range(100, len(full_data) + 1, 100):
            slice_df = full_data.iloc[:end_idx]
            # Reset the internal buffer so that each slice is processed independently.
            # f.reset_buffer()
            try:
                predicted_classes = f.predict_batch(slice_df)
            except Exception as e:
                logging.error(
                    f"Error during prediction for slice ending at row {end_idx}: {e}")
                continue

            # Get the feature vector generated by DataProcessor (assumed to include 'Dissipation').
            features = f.feature_vector
            if features is None or features.empty:
                logging.warning(
                    f"Feature vector is empty for slice ending at row {end_idx}. Skipping visualization.")
                continue
            if "Dissipation" not in features.columns:
                logging.warning(
                    f"Column 'Dissipation' not found in features for slice ending at row {end_idx}.")
                continue

            # Prepare data for plotting.
            x_values = list(range(len(features)))
            dissipation_values = features["Dissipation"].to_numpy()

            # Normalize dissipation using min-max normalization.
            min_val = dissipation_values.min()
            max_val = dissipation_values.max()
            if max_val - min_val == 0:
                norm_dissipation = np.zeros_like(dissipation_values)
            else:
                norm_dissipation = (dissipation_values -
                                    min_val) / (max_val - min_val)

            # Flatten predictions (assumes output shape is (1, sequence_length) after argmax)
            pred = predicted_classes.flatten()

            # Align lengths in case of mismatch.
            min_len = min(len(norm_dissipation), len(pred))
            x_values = x_values[:min_len]
            norm_dissipation = norm_dissipation[:min_len]
            pred = pred[:min_len]

            # Plot the current slice.
            ax.clear()
            ax.plot(x_values, norm_dissipation,
                    label="Normalized Dissipation", marker="o")
            ax.plot(x_values, pred, label="Predictions",
                    linestyle="--", marker="x")
            ax.set_title(
                f"Normalized Dissipation and Predictions (Slice size: {end_idx})")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.draw()
            plt.pause(0.01)  # Pause for 1 second to observe the plot

        # End interactive mode.
        plt.ioff()
        plt.show()
