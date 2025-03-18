import logging
from typing import Optional, List, Tuple
from sklearn.model_selection import train_test_split
from forecaster_data_processor import DataProcessor
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, BatchNormalization, Masking, Bidirectional, Input
from keras.models import Model

import matplotlib.pyplot as plt
import os
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
    logging.info("No GPU detected, runnin on CPU.")


class ForcasterTrainer:
    def __init__(self, classes, num_features):
        """
        Args:
            classes: A list of class labels.
            num_features: The number of features in the processed data (excluding the 'Fill' column).
        """
        self.classes = classes
        self.num_features = num_features
        self.model = self.build_model()
        self.label_to_index = {label: idx for idx, label in enumerate(classes)}
        self.content: Optional[List[Tuple[str, str]]] = None
        self.train_content: Optional[List[Tuple[str, str]]] = None
        self.test_content: Optional[List[Tuple[str, str]]] = None
        self.validation_content: Optional[List[Tuple[str, str]]] = None

        # Toggle flag for live plotting (default is off)
        self.live_plot_enabled = False

    def toggle_live_plot(self, enabled: bool = True):
        """Toggle live plotting on or off."""
        self.live_plot_enabled = enabled

    def build_model(self):
        inputs = Input(shape=(None, self.num_features))
        x = Masking()(inputs)
        # Bidirectional LSTM with recurrent dropout
        x = Bidirectional(
            LSTM(128, return_sequences=True, recurrent_dropout=0.2))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(
            LSTM(64, return_sequences=True, recurrent_dropout=0.2))(x)
        x = BatchNormalization()(x)
        outputs = TimeDistributed(
            Dense(len(self.classes), activation='softmax'))(x)
        model = Model(inputs, outputs)

        model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_datasets(self, training_directory: str, validation_directory: str):
        """
        Loads and splits the training data.
        """
        self.content = DataProcessor.load_content(
            training_directory, num_datasets=10)
        self.train_content, self.test_content = train_test_split(
            self.content, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
        self.validation_content = DataProcessor.load_content(
            validation_directory, num_datasets=10)
        logging.info("Datasets loaded.")

    def train_on_slice(self, data_slice: pd.DataFrame):
        """
        Trains the model on one slice of data.
        """
        y = data_slice['Fill']
        X = data_slice.drop(columns=['Fill'])
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        if len(X_np) < 2:
            return None  # Not enough data

        X_seq = X_np[:-1]
        y_seq = y_np[1:]
        X_seq = X_seq.reshape(1, X_seq.shape[0], X_seq.shape[1])
        y_seq = y_seq.reshape(1, y_seq.shape[0], 1)
        results = self.model.train_on_batch(X_seq, y_seq, reset_metrics=False)
        return results

    def get_slices(self, data_file: str, poi_file: str, slice_size: int = 100):
        data_df = pd.read_csv(data_file)

        start_idx = 0
        total_rows = len(data_df)
        slices = []
        while start_idx < total_rows:
            end_idx = min(start_idx + slice_size, total_rows)
            current_slice = data_df.iloc[0:end_idx]
            processed_slice = DataProcessor.preprocess_data(
                current_slice, poi_file)
            start_idx = end_idx
            if processed_slice is None:
                continue
            slices.append(processed_slice)
        return slices

    def train(self, training_directory: str = r'content/training_data/full_fill',
              validation_directory: str = r'content/test_data'):
        self.load_datasets(training_directory=training_directory,
                           validation_directory=validation_directory)
        logging.info(
            f'Beginning training on {len(self.train_content)} datasets.')

        if self.live_plot_enabled:
            plt.ion()  # Turn on interactive mode for live updates
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()  # Secondary y-axis for accuracy
            losses = []
            accuracies = []

        for i, (data_file, poi_file) in enumerate(self.train_content):
            slices = self.get_slices(data_file, poi_file)
            if slices is None:
                continue
            for processed_slice in slices:
                result = self.train_on_slice(processed_slice)
                if result is None:
                    continue

                if self.live_plot_enabled:
                    batch_loss = result[0]      # Training loss
                    batch_accuracy = result[1]  # Training accuracy

                    losses.append(batch_loss)
                    accuracies.append(batch_accuracy)
                    ax1.clear()
                    ax2.clear()
                    ax1.plot(losses, label='Training Loss', color='blue')
                    ax1.set_xlabel("Batch Number")
                    ax1.set_ylabel("Loss", color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')

                    # Plot training accuracy on ax2 (right y-axis)
                    ax2.plot(accuracies, label='Training Accuracy', color='red')
                    ax2.set_ylabel("Accuracy", color='red')
                    ax2.tick_params(axis='y', labelcolor='red')

                    ax1.set_title("Live Training Metrics")
                    # Combine legends from both axes
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 +
                               labels2, loc='upper right')

                    plt.draw()
                    plt.pause(0.001)

            logging.info(
                f'Beginning next dataset {i}/{len(self.train_content)}')

        logging.info("Training completed.")
        if self.live_plot_enabled:
            plt.ioff()
            plt.show()

    def tune(self):
        pass

    def test(self):
        logging.info("Starting testing on validation data.")
        total_loss = 0.0
        total_accuracy = 0.0
        count = 0
        for data_file, poi_file in self.validation_content:
            slices = self.get_slices(data_file, poi_file)
            for processed_slice in slices:
                y = processed_slice['Fill']
                X = processed_slice.drop(columns=['Fill'])
                X_np = X.to_numpy()
                y_np = y.to_numpy()
                if len(X_np) < 2:
                    continue
                X_seq = X_np[:-1]
                y_seq = y_np[1:]
                X_seq = X_seq.reshape(1, X_seq.shape[0], X_seq.shape[1])
                y_seq = y_seq.reshape(1, y_seq.shape[0], 1)
                result = self.model.test_on_batch(X_seq, y_seq)
                if result is not None:
                    batch_loss, batch_accuracy = result[0], result[1]
                    total_loss += batch_loss
                    total_accuracy += batch_accuracy
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


class Forecaster:
    def __init__(self):
        self.accumulated_buffer = None
        self.feature_vector = None
        self.model = None

    def extend_buffer(self, new_buffer: pd.DataFrame):
        if self.accumulated_buffer is None:
            self.accumulated_buffer = pd.DataFrame(columns=new_buffer.columns)
        self.accumulated_buffer = pd.concat(
            [self.accumulated_buffer, new_buffer], ignore_index=True)
        self.feature_vector = DataProcessor.generate_features(
            self.accumulated_buffer)

    def reset_buffer(self):
        self.accumulated_buffer = pd.DataFrame()


if __name__ == "__main__":
    with tf.device('/GPU:0'):
        ft = ForcasterTrainer(num_features=9, classes=[0, 1, 2, 3, 4, 5])
        ft.toggle_live_plot(True)  # Enable live plotting

        ft.train()
        ft.test()
