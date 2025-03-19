import numpy as np
import logging
import tensorflow as tf
from keras.layers import Input, Masking, Bidirectional, LSTM, BatchNormalization, Dropout, TimeDistributed, Dense
from keras.models import Model
from keras.optimizers import Adam
import logging
from typing import Optional, List, Tuple, Dict
from sklearn.model_selection import train_test_split
from forecaster_data_processor import DataProcessor
import pandas as pd
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
    logging.info("No GPU detected, running on CPU.")


class ForcasterTrainer:
    def __init__(self, classes: List[str], num_features: int) -> None:
        """
        Args:
            classes (List[str]): A list of class labels.
            num_features (int): The number of features in the processed data (excluding the 'Fill' column).
        """
        # Type checking for constructor arguments
        if not isinstance(classes, list) or not all(isinstance(label, str) for label in classes):
            raise TypeError("classes must be a list of strings")
        if not isinstance(num_features, int):
            raise TypeError("num_features must be an integer")

        self.classes: List[str] = classes
        self.num_features: int = num_features
        self.model: Model = self.build_model()  # initial default model
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
        # Original fixed hyperparameter model
        inputs = Input(shape=(None, self.num_features))
        x = Masking()(inputs)
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

    def build_model_with_hyperparams(self, units_lstm1, units_lstm2, dropout_rate, recurrent_dropout, learning_rate):
        """
        Build a model with specified hyperparameters.
        """
        inputs = Input(shape=(None, self.num_features))
        x = Masking()(inputs)
        x = Bidirectional(LSTM(units_lstm1, return_sequences=True,
                          recurrent_dropout=recurrent_dropout))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Bidirectional(LSTM(units_lstm2, return_sequences=True,
                          recurrent_dropout=recurrent_dropout))(x)
        x = BatchNormalization()(x)
        outputs = TimeDistributed(
            Dense(len(self.classes), activation='softmax'))(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
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

    def train_on_slice(self, data_slice):
        """
        Trains the model on one slice of data.
        """
        y = data_slice['Fill']
        X = data_slice.drop(columns=['Fill'])
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        if len(X_np) < 2:
            return None  # Not enough data

        # Create input/output sequences:
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

    def moving_average(self, data, window_size=20):
        """Compute the moving average of a list of numbers."""
        if len(data) < window_size:
            return data  # Not enough data to compute the moving average.
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def train(self, training_directory: str = r'content/training_data/full_fill',
              validation_directory: str = r'content/test_data'):
        self.load_datasets(training_directory=training_directory,
                           validation_directory=validation_directory)

        if self.live_plot_enabled:
            plt.ion()
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()
            losses = []
            accuracies = []
            sma_window = 20

        for i, (data_file, poi_file) in enumerate(self.train_content):
            logging.info(
                f'Beginning next dataset {i + 1}/{len(self.train_content)}.')
            slices = self.get_slices(data_file, poi_file)
            if slices is None:
                continue
            for processed_slice in slices:
                result = self.train_on_slice(processed_slice)
                if result is None:
                    continue

                if self.live_plot_enabled:
                    batch_loss = result[0]
                    batch_accuracy = result[1]
                    losses.append(batch_loss)
                    accuracies.append(batch_accuracy)
                    smooth_losses = self.moving_average(losses, sma_window)
                    smooth_accuracies = self.moving_average(
                        accuracies, sma_window)
                    ax1.clear()
                    ax2.clear()
                    ax1.plot(losses, label='Training Loss (raw)',
                             linestyle=':', color='blue')
                    if len(losses) >= sma_window:
                        ax1.plot(range(sma_window - 1, len(losses)), smooth_losses,
                                 label='Training Loss (smoothed)', color='blue')
                    ax1.set_xlabel("Batch Number")
                    ax1.set_ylabel("Loss", color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')
                    ax1.grid(True, linestyle='--', alpha=0.5)
                    ax2.plot(accuracies, label='Training Accuracy (raw)',
                             linestyle=':', color='red')
                    if len(accuracies) >= sma_window:
                        ax2.plot(range(sma_window - 1, len(accuracies)), smooth_accuracies,
                                 label='Training Accuracy (smoothed)', color='red')
                    ax2.set_ylabel("Accuracy", color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    ax1.set_title("Live Training Metrics")
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 +
                               labels2, loc='upper right')
                    plt.draw()
                    plt.pause(0.001)

        logging.info("Training completed.")
        if self.live_plot_enabled:
            plt.ioff()
            plt.show()

    def tune(self, training_directory: str = r'content/training_data/full_fill',
             validation_directory: str = r'content/test_data', num_epochs: int = 1,
             max_batches: int = 5):
        """
        Hyperparameter tuning using a basic grid search. For each combination in the grid,
        the model is built, trained on a limited number of batches, and then evaluated on the validation set.
        Live plotting shows the average validation loss for each hyperparameter combination as well as the best loss so far.
        """
        # Load datasets (both training and validation)
        self.load_datasets(training_directory, validation_directory)

        # Setup live plotting if enabled
        if self.live_plot_enabled:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel("Tuning Iteration")
            ax.set_ylabel("Average Validation Loss")
            tuning_iter = []
            val_losses_list = []

        # Define the hyperparameter grid
        from itertools import product
        hp_grid = {
            'units_lstm1': [64, 128],
            'units_lstm2': [32, 64],
            'dropout_rate': [0.3, 0.5],
            'recurrent_dropout': [0.2, 0.3],
            'learning_rate': [0.001, 0.0005]
        }

        best_val_loss = float('inf')
        best_hp = None
        all_results = []
        iteration = 0

        # Iterate over all combinations in the grid
        for units1, units2, dropout_rate, rec_dropout, lr in product(
            hp_grid['units_lstm1'], hp_grid['units_lstm2'],
            hp_grid['dropout_rate'], hp_grid['recurrent_dropout'],
            hp_grid['learning_rate']
        ):
            iteration += 1
            logging.info(f"Testing hyperparameters: LSTM1 Units={units1}, LSTM2 Units={units2}, "
                         f"Dropout={dropout_rate}, Recurrent Dropout={rec_dropout}, LR={lr}")
            # Build a new model with the current hyperparameters
            model = self.build_model_with_hyperparams(
                units1, units2, dropout_rate, rec_dropout, lr)

            # Train for a limited number of batches
            batch_count = 0
            for data_file, poi_file in self.train_content:
                slices = self.get_slices(data_file, poi_file)
                if slices is None:
                    continue
                for data_slice in slices:
                    # Prepare slice data (same logic as in train_on_slice)
                    y = data_slice['Fill']
                    X = data_slice.drop(columns=['Fill'])
                    X_np = X.to_numpy()
                    y_np = y.to_numpy()
                    if len(X_np) < 2:
                        continue
                    X_seq = X_np[:-1].reshape(1, -1, self.num_features)
                    y_seq = y_np[1:].reshape(1, -1, 1)
                    for epoch in range(num_epochs):
                        model.train_on_batch(X_seq, y_seq)
                    batch_count += 1
                    if batch_count >= max_batches:
                        break
                if batch_count >= max_batches:
                    break

            # Evaluate on the validation set
            val_losses = []
            for data_file, poi_file in self.validation_content:
                slices = self.get_slices(data_file, poi_file)
                if slices is None:
                    continue
                for data_slice in slices:
                    y = data_slice['Fill']
                    X = data_slice.drop(columns=['Fill'])
                    X_np = X.to_numpy()
                    y_np = y.to_numpy()
                    if len(X_np) < 2:
                        continue
                    X_seq = X_np[:-1].reshape(1, -1, self.num_features)
                    y_seq = y_np[1:].reshape(1, -1, 1)
                    loss, _ = model.evaluate(X_seq, y_seq, verbose=0)
                    val_losses.append(loss)
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            logging.info(f"Hyperparameters: LSTM1 Units={units1}, LSTM2 Units={units2}, Dropout={dropout_rate}, "
                         f"Recurrent Dropout={rec_dropout}, LR={lr} -> Avg Val Loss={avg_val_loss}")

            # Save results for this hyperparameter combination
            current_hp = {
                'units_lstm1': units1,
                'units_lstm2': units2,
                'dropout_rate': dropout_rate,
                'recurrent_dropout': rec_dropout,
                'learning_rate': lr,
                'avg_val_loss': avg_val_loss
            }
            all_results.append(current_hp)

            # Update best hyperparameters if current combination is better
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_hp = current_hp

            # Update live plot if enabled
            if self.live_plot_enabled:
                tuning_iter.append(iteration)
                val_losses_list.append(avg_val_loss)
                ax.clear()
                ax.plot(tuning_iter, val_losses_list,
                        'bo-', label="Avg Val Loss")
                ax.axhline(best_val_loss, color='green',
                           linestyle='--', label="Best Loss So Far")
                ax.set_xlabel("Tuning Iteration")
                ax.set_ylabel("Average Validation Loss")
                ax.legend()
                plt.draw()
                plt.pause(0.001)

        logging.info(f"Best hyperparameters: {best_hp}")
        # Optionally, update self.model with the best hyperparameters found
        self.model = self.build_model_with_hyperparams(
            best_hp['units_lstm1'],
            best_hp['units_lstm2'],
            best_hp['dropout_rate'],
            best_hp['recurrent_dropout'],
            best_hp['learning_rate']
        )

        if self.live_plot_enabled:
            plt.ioff()
            plt.show()
        self.hyperparams = best_hp
        self.tuning_results = all_results
        return best_hp, all_results

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

    def save_model(self, save_path: str):
        """
        Save the current model to the specified path.

        Args:
            save_path (str): The file path where the model should be saved.
        """
        try:
            self.model.save(save_path)
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
        ft.tune()
        ft.train()
        ft.test()
