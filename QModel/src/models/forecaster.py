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
    def __init__(self, classes: List, num_features: int) -> None:
        """
        Args:
            classes (List[str]): A list of class labels.
            num_features (int): The number of features in the processed data (excluding the 'Fill' column).
        """
        # Type checking for constructor arguments
        if not isinstance(classes, list):
            raise TypeError("classes must be a list")
        if not isinstance(num_features, int):
            raise TypeError("num_features must be an integer")

        self.classes: List = classes
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
        inputs = Input(shape=(None, self.num_features))
        x = Masking()(inputs)
        # Using a single bidirectional GRU layer with fewer units
        x = Bidirectional(GRU(64, return_sequences=True, dropout=0.2))(x)
        outputs = TimeDistributed(
            Dense(len(self.classes), activation='softmax'))(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def build_model_with_hyperparams(self, units, dropout_rate, recurrent_dropout, learning_rate):
        """
        Build a lightweight model with specified hyperparameters.
        """
        inputs = Input(shape=(None, self.num_features))
        x = Masking()(inputs)
        # Use a single bidirectional GRU layer instead of stacked LSTMs
        x = Bidirectional(
            GRU(units, return_sequences=True, dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout)
        )(x)
        outputs = TimeDistributed(
            Dense(len(self.classes), activation='softmax')
        )(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_datasets(self, training_directory: str, validation_directory: str):
        """
        Loads and splits the training data.
        """
        self.content = DataProcessor.load_content(
            training_directory, num_datasets=20)
        self.train_content, self.test_content = train_test_split(
            self.content, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
        self.validation_content = DataProcessor.load_content(
            validation_directory, num_datasets=20)
        logging.info("Datasets loaded.")

    def train_on_slice(self, data_slice):
        """
        Trains the model on one slice of data with class weighting.
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

        # Define class weights, increasing the importance of classes 1-5.
        # For example, class 0 has weight 1.0 and classes 1,2,3,4,5 have weight 2.0.
        class_weights = {0: 1.0, 1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0}

        # Create a sample weight array for each label in the sequence.
        # Flatten y_seq, then map each label to its weight.
        sample_weights = np.array(
            [class_weights.get(int(label), 1.0) for label in y_seq.flatten()])
        # Reshape to match the batch shape (batch_size, sequence_length).
        sample_weights = sample_weights.reshape(1, -1)

        # Pass the sample weights to train_on_batch.
        results = self.model.train_on_batch(
            X_seq, y_seq, sample_weight=sample_weights, reset_metrics=False)
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
            # Create two subplots: one for Dissipation, one for Loss/Accuracy
            fig, (ax_diss, ax_metrics) = plt.subplots(
                nrows=2, ncols=1, figsize=(12, 10))
            # Create a twin axis on the bottom plot for Accuracy
            ax_acc = ax_metrics.twinx()

            losses = []
            accuracies = []
            dissipation_values = []  # To store Dissipation values
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

                # Extract the dissipation value from the processed_slice.
                # Adjust this line as needed: here we assume it's a scalar or you may take a mean, etc.
                diss_value = processed_slice['Dissipation'].values
                dissipation_values = diss_value

                if self.live_plot_enabled:
                    batch_loss = result[0]
                    batch_accuracy = result[1]
                    losses.append(batch_loss)
                    accuracies.append(batch_accuracy)
                    smooth_losses = self.moving_average(losses, sma_window)
                    smooth_accuracies = self.moving_average(
                        accuracies, sma_window)

                    # Clear previous plots
                    ax_diss.clear()
                    ax_metrics.clear()
                    ax_acc.clear()

                    # --- Update the Dissipation plot (upper subplot) ---
                    ax_diss.plot(dissipation_values,
                                 label='Dissipation', color='green')
                    ax_diss.set_title("Dissipation Monitoring")
                    ax_diss.set_ylabel("Dissipation", color='green')
                    ax_diss.tick_params(axis='y', labelcolor='green')
                    ax_diss.grid(True, linestyle='--', alpha=0.5)

                    # --- Update the Loss/Accuracy plot (lower subplot) ---
                    ax_metrics.plot(
                        losses, label='Training Loss (raw)', linestyle=':', color='blue')
                    if len(losses) >= sma_window:
                        ax_metrics.plot(range(sma_window - 1, len(losses)), smooth_losses,
                                        label='Training Loss (smoothed)', color='blue')
                    ax_metrics.set_xlabel("Batch Number")
                    ax_metrics.set_ylabel("Loss", color='blue')
                    ax_metrics.tick_params(axis='y', labelcolor='blue')
                    ax_metrics.grid(True, linestyle='--', alpha=0.5)

                    ax_acc.plot(
                        accuracies, label='Training Accuracy (raw)', linestyle=':', color='red')
                    if len(accuracies) >= sma_window:
                        ax_acc.plot(range(sma_window - 1, len(accuracies)), smooth_accuracies,
                                    label='Training Accuracy (smoothed)', color='red')
                    ax_acc.set_ylabel("Accuracy", color='red')
                    ax_acc.tick_params(axis='y', labelcolor='red')

                    ax_metrics.set_title("Live Training Metrics")
                    # Combine legends from both axes for the lower subplot
                    lines1, labels1 = ax_metrics.get_legend_handles_labels()
                    lines2, labels2 = ax_acc.get_legend_handles_labels()
                    ax_metrics.legend(
                        lines1 + lines2, labels1 + labels2, loc='upper right')

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
        Hyperparameter tuning using a basic grid search for the lightweight GRU-based model.
        For each combination in the grid, the model is built, trained on a limited number of batches,
        and then evaluated on the validation set.
        Live plotting shows the average validation loss for each hyperparameter combination as well as
        the best loss so far.
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

        # Define the hyperparameter grid for the lightweight model
        from itertools import product
        hp_grid = {
            'units': [64, 128],
            'dropout_rate': [0.3, 0.5],
            'recurrent_dropout': [0.2, 0.3],
            'learning_rate': [0.001, 0.0005]
        }

        best_val_loss = float('inf')
        best_hp = None
        all_results = []
        iteration = 0

        # Iterate over all combinations in the grid
        for units, dropout_rate, rec_dropout, lr in product(
                hp_grid['units'], hp_grid['dropout_rate'],
                hp_grid['recurrent_dropout'], hp_grid['learning_rate']):
            iteration += 1
            logging.info(f"Testing hyperparameters: Units={units}, Dropout={dropout_rate}, "
                         f"Recurrent Dropout={rec_dropout}, LR={lr}")
            # Build a new model with the current hyperparameters
            model = self.build_model_with_hyperparams(
                units, dropout_rate, rec_dropout, lr)

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
            logging.info(f"Hyperparameters: Units={units}, Dropout={dropout_rate}, "
                         f"Recurrent Dropout={rec_dropout}, LR={lr} -> Avg Val Loss={avg_val_loss}")

            # Save results for this hyperparameter combination
            current_hp = {
                'units': units,
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
            best_hp['units'],
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
TESTING = True

if __name__ == "__main__":
    if TRAINING:
        with tf.device('/GPU:0'):
            ft = ForcasterTrainer(num_features=9, classes=[0, 1, 2, 3, 4, 5])
            ft.toggle_live_plot(True)  # Enable live plotting
            # ft.tune()
            ft.train()
            # ft.test()
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
