from typing import Tuple, List, Optional
import joblib
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import os
from v4_dp import DP
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from tensorflow import keras
import tensorflow as tf
import numpy as np
EPOCHS = 50           # Epochs per trial during tuning
FINAL_EPOCHS = 150    # Epochs for final training
MAX_TRIALS = 30       # Number of hyperparameter combinations


def visualize_offset_distribution(self, y_train, y_val, y_test):
    """
    Visualize the distribution of offsets in train/val/test sets

    Args:
        y_train: Training offsets
        y_val: Validation offsets  
        y_test: Test offsets
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    datasets = [('Training', y_train), ('Validation', y_val), ('Test', y_test)]

    for ax, (name, data) in zip(axes, datasets):
        ax.hist(data, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', label='Center')
        ax.axvline(x=-self.stride, color='green', linestyle=':',
                   label=f'±{self.stride} (stride)')
        ax.axvline(x=self.stride, color='green', linestyle=':')

        ax.set_xlabel('Offset from Window Center')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Set Offset Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f'Mean: {np.mean(data):.1f}\nStd: {np.std(data):.1f}\nRange: [{np.min(data):.1f}, {np.max(data):.1f}]'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('POI Offset Distribution Across Datasets', fontsize=14)
    plt.tight_layout()
    plt.show()


class POIDataLoader:
    """Load and preprocess POI data using v4_dp module with realistic window offsets"""

    def __init__(self, window_size: int = 128, poi_indices: List[int] = [0, 1, 3, 4, 5],
                 stride: int = 16, search_window_size: int = 256):
        """
        Initialize data loader

        Args:
            window_size: Size of the window around POI (should match your upstream system)
            poi_indices: Which POI indices to use (0-indexed, skipping 3rd)
            stride: Stride used in the upstream window selection system
            search_window_size: Size of the search region in the upstream system
        """
        self.window_size = window_size
        self.poi_indices = poi_indices
        self.stride = stride
        self.search_window_size = search_window_size
        self.scaler = StandardScaler()
        self.dp = DP()

    def simulate_upstream_window_selection(self, true_poi: int, data_length: int) -> int:
        """
        Simulate the upstream system that selects the best window using stride-based search.
        This simulates the window that would be passed to the localizer.

        Args:
            true_poi: True POI location
            data_length: Length of the data sequence

        Returns:
            window_start: Start position of the selected window
        """
        # Simulate that the upstream system searches in a region around the POI
        # with some estimation error
        estimation_error = np.random.randint(
            -self.search_window_size//4, self.search_window_size//4)
        estimated_region_center = true_poi + estimation_error

        # Define search region boundaries
        search_start = max(0, estimated_region_center -
                           self.search_window_size // 2)
        search_end = min(data_length - self.window_size,
                         search_start + self.search_window_size)

        # Generate all possible window positions with the given stride
        possible_starts = list(
            range(int(search_start), int(search_end), self.stride))

        if not possible_starts:
            # Fallback if no valid positions
            possible_starts = [
                max(0, min(true_poi - self.window_size // 2, data_length - self.window_size))]

        # Simulate selection of "best" window (in reality, this would be based on some score)
        # For training, we want to simulate realistic distributions of offsets
        # Windows closer to the POI are more likely to be selected
        weights = []
        for start in possible_starts:
            window_center = start + self.window_size // 2
            distance = abs(window_center - true_poi)
            # Use exponential decay for weight - closer windows more likely
            weight = np.exp(-distance / (self.window_size / 4))
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / weights.sum()

        # Ensure weights is 1-dimensional and properly normalized
        weights = weights.flatten()

        # Select a window position based on weights
        selected_start = np.random.choice(possible_starts, p=weights)

        return selected_start

    def extract_windows_from_pair(self, data_df: pd.DataFrame, poi_locations: np.ndarray,
                                  simulate_upstream: bool = True) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract windows from a single data/POI pair with realistic offsets

        Args:
            data_df: DataFrame with raw data
            poi_locations: Array of POI locations
            simulate_upstream: Whether to simulate the upstream window selection process

        Returns:
            windows: List of feature windows
            offsets: List of POI offsets from window center
        """
        # Generate features for the entire file
        features_df = self.dp.gen_features(data_df)

        # Select only desired POI indices
        selected_pois = poi_locations[self.poi_indices]

        windows = []
        offsets = []

        for poi_loc in selected_pois:
            # Ensure poi_loc is a scalar value
            if isinstance(poi_loc, np.ndarray):
                poi_loc = poi_loc.item() if poi_loc.size == 1 else poi_loc[0]
            poi_loc = int(poi_loc)

            if poi_loc < len(features_df):
                if simulate_upstream:
                    # Simulate the upstream system's window selection
                    window_start = self.simulate_upstream_window_selection(
                        poi_loc, len(features_df)
                    )
                else:
                    # For testing, use perfect centering
                    window_start = max(0, min(poi_loc - self.window_size // 2,
                                              len(features_df) - self.window_size))

                window_end = min(
                    window_start + self.window_size, len(features_df))
                window_start = int(window_start)
                window_end = int(window_end)

                # Extract window of features
                window = features_df.iloc[window_start:window_end].values

                # Pad if necessary (edge case at boundaries)
                if window.shape[0] < self.window_size:
                    padding = self.window_size - window.shape[0]
                    window = np.pad(
                        window, ((0, padding), (0, 0)), mode='edge')

                # Calculate true offset from window center
                window_center = window_start + self.window_size // 2
                true_offset = poi_loc - window_center

                # Only include windows where POI is actually within or near the window
                # (with some margin for edge cases)
                margin = self.window_size // 4
                if abs(true_offset) <= (self.window_size // 2 + margin):
                    windows.append(window)
                    offsets.append(true_offset)

        return windows, offsets

    def load_all_data(self, data_dir: str, simulate_upstream: bool = True,
                      augmentation_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all data pairs from directory using DP module

        Args:
            data_dir: Directory containing data pairs
            simulate_upstream: Whether to simulate the upstream window selection
            augmentation_factor: Number of different window selections per POI for augmentation

        Returns:
            X: All windows with features
            y: All offsets
        """
        all_windows = []
        all_offsets = []

        # Load content using DP module
        data_pairs = self.dp.load_content(data_dir, num_datasets=100)

        print(f"Found {len(data_pairs)} data pairs")

        for i, (d, p) in enumerate(data_pairs):
            data_df = pd.read_csv(d)
            poi_array = pd.read_csv(p, header=None).values

            # Generate multiple augmented versions with different window selections
            for aug in range(augmentation_factor if simulate_upstream else 1):
                # Extract windows from this pair
                windows, offsets = self.extract_windows_from_pair(
                    data_df, poi_array, simulate_upstream=simulate_upstream
                )

                all_windows.extend(windows)
                all_offsets.extend(offsets)

            print(
                f"Processed pair {i+1}/{len(data_pairs)}: extracted {len(windows) * augmentation_factor} windows")

        # Convert to numpy arrays
        X = np.array(all_windows)
        y = np.array(all_offsets, dtype=np.float32)  # Ensure float32 type

        # Normalize features
        X_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_flat = self.scaler.fit_transform(X_flat)
        X = X_flat.reshape(X_shape).astype(np.float32)  # Ensure float32 type

        print(f"\nTotal windows extracted: {len(X)}")
        print(f"Window shape: {X.shape}")
        print(f"Feature dimensions: {X.shape[-1]}")
        print(f"Offset statistics:")
        print(f"  Mean: {y.mean():.2f}")
        print(f"  Std: {y.std():.2f}")
        print(f"  Min: {y.min():.2f}")
        print(f"  Max: {y.max():.2f}")
        print(
            f"  Within ±{self.stride}: {np.mean(np.abs(y) <= self.stride):.2%}")

        return X, y

    def prepare_inference_window(self, data_df: pd.DataFrame, window_start: int) -> np.ndarray:
        """
        Prepare a single window for inference (from upstream system)

        Args:
            data_df: DataFrame with raw data
            window_start: Start position of the window (from upstream system)

        Returns:
            Normalized feature window ready for prediction
        """
        # Generate features
        features_df = self.dp.gen_features(data_df)

        # Extract window
        window_end = min(window_start + self.window_size, len(features_df))
        window = features_df.iloc[window_start:window_end].values

        # Pad if necessary
        if window.shape[0] < self.window_size:
            padding = self.window_size - window.shape[0]
            window = np.pad(window, ((0, padding), (0, 0)), mode='edge')

        # Normalize
        window = self.scaler.transform(window)

        return window[np.newaxis, ...]  # Add batch dimension


class POILocalizer:
    """Neural network model for POI localization with offset prediction"""

    def __init__(self, input_shape: Tuple[int, int], stride: int = 16):
        """
        Initialize localizer

        Args:
            input_shape: Shape of input windows (window_size, n_features)
            stride: Stride from upstream system (for architecture optimization)
        """
        self.input_shape = input_shape
        self.stride = stride
        self.model = None
        self.history = None

    def build_model(self, hp=None):
        """
        Build the neural network model optimized for offset prediction

        Args:
            hp: Keras Tuner hyperparameters object

        Returns:
            Compiled model
        """
        if hp is None:
            # Default hyperparameters optimized for offset prediction
            conv_filters = [128, 256, 512]
            kernel_sizes = [7, 5, 3]
            lstm_units = 256
            dense_units = [512, 256, 128]
            dropout_rate = 0.3
            learning_rate = 0.001
            use_lstm = True
            # use_attention = True  # Disabled for Sequential model compatibility
        else:
            # Hyperparameter search space
            conv_filters = [
                hp.Int(f'conv_{i}_filters', 32, 512, step=32)
                for i in range(hp.Int('n_conv_layers', 2, 4))
            ]
            # Pad with zeros if needed to maintain consistent structure
            while len(conv_filters) < 3:
                conv_filters.append(256)

            kernel_sizes = [
                hp.Choice(f'kernel_{i}_size', [3, 5, 7, 9])
                for i in range(len(conv_filters))
            ]
            # Pad with defaults if needed
            while len(kernel_sizes) < 3:
                kernel_sizes.append(5)

            lstm_units = hp.Int('lstm_units', 64, 512, step=32)

            n_dense_layers = hp.Int('n_dense_layers', 2, 4)
            dense_units = [
                hp.Int(f'dense_{i}_units', 64, 512, step=32)
                for i in range(n_dense_layers)
            ]

            dropout_rate = hp.Float('dropout', 0.1, 0.5, step=0.05)
            learning_rate = hp.Float(
                'learning_rate', 1e-5, 1e-2, sampling='LOG')
            use_lstm = hp.Boolean('use_lstm')
            batch_norm = hp.Boolean('use_batch_norm')
            pool_size = hp.Choice('pool_size', [2, 3])
            optimizer_type = hp.Choice(
                'optimizer', ['adam', 'adamw', 'rmsprop'])

        model = models.Sequential()

        # Input layer
        model.add(layers.Input(shape=self.input_shape))

        # Variable number of convolutional layers
        n_conv_layers = len(conv_filters) if hp else 3

        for i in range(n_conv_layers):
            if i < len(conv_filters) and i < len(kernel_sizes):
                model.add(layers.Conv1D(
                    conv_filters[i],
                    kernel_sizes[i],
                    activation='relu',
                    padding='same'
                ))

                # Optional batch normalization
                if hp is None or batch_norm:
                    model.add(layers.BatchNormalization())

                # Pooling for all but last conv layer
                if i < n_conv_layers - 1:
                    model.add(layers.MaxPooling1D(pool_size if hp else 2))

                # Dropout with reduced rate for conv layers
                model.add(layers.Dropout(dropout_rate * 0.5))

        # Note: Attention mechanism commented out for Sequential model compatibility
        # If you want to use attention, consider using Functional API instead
        # if use_attention if hp is None else use_attention:
        #     # Would require Functional API for proper self-attention implementation
        #     pass

        # Optional LSTM for sequential dependencies
        if use_lstm:
            model.add(layers.Bidirectional(layers.LSTM(
                lstm_units,
                return_sequences=False,
                dropout=dropout_rate
            )))
        else:
            model.add(layers.GlobalMaxPooling1D())

        # Dense layers for regression
        for i, units in enumerate(dense_units):
            model.add(layers.Dense(units, activation='relu'))
            if hp is None or batch_norm:
                model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))

        # Output layer - single value for offset prediction
        model.add(layers.Dense(1, activation='linear'))

        # Custom loss function that weights errors based on magnitude
        def weighted_huber_loss(y_true, y_pred):
            """Custom loss that penalizes larger errors more"""
            # Ensure proper types
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            error = y_true - y_pred
            # Use stride as the delta for Huber loss
            huber_delta = tf.cast(self.stride, tf.float32)

            # Standard Huber loss
            is_small_error = tf.abs(error) <= huber_delta
            small_error_loss = 0.5 * tf.square(error)
            large_error_loss = huber_delta * \
                (tf.abs(error) - 0.5 * huber_delta)

            loss = tf.where(is_small_error, small_error_loss, large_error_loss)

            # Add weight based on error magnitude
            weight = 1.0 + tf.abs(error) / \
                tf.cast(self.input_shape[0] / 2, tf.float32)
            weighted_loss = loss * weight

            return tf.reduce_mean(weighted_loss)

        # Select optimizer based on hyperparameter
        if hp:
            if optimizer_type == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_type == 'adamw':
                optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
            else:  # rmsprop
                optimizer = keras.optimizers.RMSprop(
                    learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=weighted_huber_loss,
            metrics=['mae', 'mse']
        )

        return model

    def create_tuner(self, X_val, y_val, max_trials=30, tuner_type='bayesian'):
        """
        Create Keras Tuner for hyperparameter optimization

        Args:
            X_val: Validation data
            y_val: Validation labels
            max_trials: Maximum number of hyperparameter combinations to try
            tuner_type: Type of tuner ('bayesian', 'random', or 'hyperband')

        Returns:
            Configured tuner
        """
        # Create appropriate tuner based on type
        if tuner_type == 'bayesian':
            tuner = kt.BayesianOptimization(
                self.build_model,
                objective=kt.Objective('val_mae', direction='min'),
                max_trials=max_trials,
                directory='poi_tuning',
                project_name='poi_localizer_offset_bayesian',
                overwrite=True,
                seed=42
            )
        elif tuner_type == 'random':
            tuner = kt.RandomSearch(
                self.build_model,
                objective=kt.Objective('val_mae', direction='min'),
                max_trials=max_trials,
                directory='poi_tuning',
                project_name='poi_localizer_offset_random',
                overwrite=True,
                seed=42
            )
        elif tuner_type == 'hyperband':
            tuner = kt.Hyperband(
                self.build_model,
                objective=kt.Objective('val_mae', direction='min'),
                max_epochs=100,
                factor=3,
                directory='poi_tuning',
                project_name='poi_localizer_offset_hyperband',
                overwrite=True,
                seed=42
            )
        else:
            raise ValueError(f"Unknown tuner type: {tuner_type}")

        return tuner

    def train_with_tuning(self, X_train, y_train, X_val, y_val,
                          epochs=50, final_epochs=150, batch_size=32,
                          max_trials=30, tuner_type='bayesian'):
        """
        Train model with hyperparameter tuning

        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of epochs per trial during tuning
            final_epochs: Number of epochs for final training with best params
            batch_size: Batch size
            max_trials: Maximum number of trials
            tuner_type: Type of tuner to use

        Returns:
            Best model and tuner
        """
        # Create tuner
        tuner = self.create_tuner(X_val, y_val, max_trials, tuner_type)

        # Define callbacks for tuning
        early_stop = callbacks.EarlyStopping(
            monitor='val_mae',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=0
        )

        # Run hyperparameter search
        print(
            f"Starting hyperparameter search using {tuner_type} optimization...")
        print(f"Running {max_trials} trials with {epochs} epochs each...")

        tuner.search(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]

        print("\n" + "="*50)
        print("BEST HYPERPARAMETERS FOUND")
        print("="*50)

        # Print hyperparameters in organized groups
        print("\nArchitecture:")
        for param, value in best_hp.values.items():
            if any(x in param for x in ['conv', 'lstm', 'dense', 'layers', 'kernel', 'pool']):
                print(f"  {param}: {value}")

        print("\nRegularization:")
        for param, value in best_hp.values.items():
            if any(x in param for x in ['dropout', 'batch_norm']):
                print(f"  {param}: {value}")

        print("\nOptimization:")
        for param, value in best_hp.values.items():
            if any(x in param for x in ['learning_rate', 'optimizer']):
                print(f"  {param}: {value}")

        print("\n" + "="*50)

        # Build and train best model with more epochs
        self.model = self.build_model(best_hp)

        print(
            f"\nTraining final model with best hyperparameters for {final_epochs} epochs...")
        print("Model architecture:")
        self.model.summary()

        self.history = self.model.fit(
            X_train, y_train,
            epochs=final_epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_mae',
                    patience=25,
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_mae',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    'best_poi_model_tuned.keras',
                    save_best_only=True,
                    monitor='val_mae',
                    verbose=1
                ),
                callbacks.TensorBoard(
                    log_dir='./logs_tuned',
                    histogram_freq=1,
                    write_graph=True,
                    update_freq='epoch'
                )
            ],
            verbose=1
        )

        # Save best hyperparameters
        import json
        with open('best_hyperparameters.json', 'w') as f:
            json.dump(best_hp.values, f, indent=2)
        print("\nBest hyperparameters saved to 'best_hyperparameters.json'")

        return self.model, tuner

    def train(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=32):
        """
        Train model without hyperparameter tuning

        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size

        Returns:
            Trained model
        """
        self.model = self.build_model()

        # Define callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_mae',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_mae',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'best_poi_model_offset.keras',
                save_best_only=True,
                monitor='val_mae',
                verbose=1
            )
        ]

        print("Training model with realistic offset distribution...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )

        return self.model

    def predict(self, X):
        """
        Predict POI offsets from window center

        Args:
            X: Input windows

        Returns:
            Predicted offsets
        """
        return self.model.predict(X, verbose=0)

    def predict_with_confidence(self, X, n_predictions=10):
        """
        Predict with confidence estimation using dropout at inference

        Args:
            X: Input windows
            n_predictions: Number of forward passes for uncertainty estimation

        Returns:
            mean_predictions: Mean predicted offsets
            std_predictions: Standard deviation of predictions (uncertainty)
        """
        predictions = []
        for _ in range(n_predictions):
            pred = self.model(X, training=True)  # Enable dropout
            predictions.append(pred.numpy())

        predictions = np.array(predictions)
        mean_predictions = np.mean(predictions, axis=0)
        std_predictions = np.std(predictions, axis=0)

        return mean_predictions, std_predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance

        Args:
            X_test: Test data
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X_test)
        errors = predictions.flatten() - y_test
        abs_errors = np.abs(errors)

        # Basic metrics
        mse = np.mean(errors ** 2)
        mae = np.mean(abs_errors)

        # Percentile statistics
        percentiles = [50, 75, 90, 95, 99]
        percentile_errors = {
            f'error_p{p}': np.percentile(abs_errors, p)
            for p in percentiles
        }

        # Accuracy within different thresholds (including stride-based)
        thresholds = [0.5, 1, 2, 3, 5, 8,
                      self.stride/2, self.stride, self.stride*2]
        accuracies = {
            f'accuracy_within_{t}': np.mean(abs_errors <= t)
            for t in thresholds
        }

        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            **percentile_errors,
            **accuracies
        }

    def plot_tuning_history(self, tuner, top_n=10):
        """
        Plot hyperparameter tuning history

        Args:
            tuner: Keras tuner object
            top_n: Number of top trials to display
        """
        import matplotlib.pyplot as plt

        # Get all trials
        trials = tuner.oracle.get_best_trials(num_trials=top_n)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Trial scores over time
        scores = [trial.score for trial in trials]
        trial_ids = list(range(1, len(scores) + 1))

        axes[0, 0].bar(trial_ids, scores, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('Validation MAE')
        axes[0, 0].set_title('Top Trial Scores')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Learning rate vs performance
        learning_rates = []
        val_maes = []
        for trial in trials:
            if 'learning_rate' in trial.hyperparameters.values:
                learning_rates.append(
                    trial.hyperparameters.values['learning_rate'])
                val_maes.append(trial.score)

        if learning_rates:
            axes[0, 1].scatter(learning_rates, val_maes, s=50, alpha=0.6)
            axes[0, 1].set_xscale('log')
            axes[0, 1].set_xlabel('Learning Rate')
            axes[0, 1].set_ylabel('Validation MAE')
            axes[0, 1].set_title('Learning Rate vs Performance')
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Dropout rate vs performance
        dropout_rates = []
        val_maes_dropout = []
        for trial in trials:
            if 'dropout' in trial.hyperparameters.values:
                dropout_rates.append(trial.hyperparameters.values['dropout'])
                val_maes_dropout.append(trial.score)

        if dropout_rates:
            axes[1, 0].scatter(dropout_rates, val_maes_dropout,
                               s=50, alpha=0.6, color='orange')
            axes[1, 0].set_xlabel('Dropout Rate')
            axes[1, 0].set_ylabel('Validation MAE')
            axes[1, 0].set_title('Dropout Rate vs Performance')
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Architecture complexity vs performance
        model_complexities = []
        val_maes_complex = []
        for trial in trials:
            hp = trial.hyperparameters.values
            # Calculate complexity as a rough measure
            complexity = 0
            for key, value in hp.items():
                if 'units' in key or 'filters' in key:
                    complexity += value
            if complexity > 0:
                model_complexities.append(complexity)
                val_maes_complex.append(trial.score)

        if model_complexities:
            axes[1, 1].scatter(
                model_complexities, val_maes_complex, s=50, alpha=0.6, color='green')
            axes[1, 1].set_xlabel('Model Complexity (Total Units/Filters)')
            axes[1, 1].set_ylabel('Validation MAE')
            axes[1, 1].set_title('Model Complexity vs Performance')
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Hyperparameter Tuning Analysis', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_training_history(self):
        """Plot training history with multiple metrics"""
        if self.history is None:
            print("No training history available")
            return

        import matplotlib.pyplot as plt

        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Loss
        axes[0].plot(epochs, history['loss'], 'b-', label='Training Loss')
        axes[0].plot(epochs, history['val_loss'],
                     'r-', label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss During Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: MAE
        axes[1].plot(epochs, history['mae'], 'b-', label='Training MAE')
        axes[1].plot(epochs, history['val_mae'], 'r-', label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Mean Absolute Error During Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: MSE
        axes[2].plot(epochs, history['mse'], 'b-', label='Training MSE')
        axes[2].plot(epochs, history['val_mse'], 'r-', label='Validation MSE')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('MSE')
        axes[2].set_title('Mean Squared Error During Training')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('Training History', fontsize=14)
        plt.tight_layout()
        plt.show()

        # Print final metrics
        print("\nFinal Training Metrics:")
        print(f"  Training Loss: {history['loss'][-1]:.4f}")
        print(f"  Validation Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Training MAE: {history['mae'][-1]:.4f}")
        print(f"  Validation MAE: {history['val_mae'][-1]:.4f}")
        print(f"  Training MSE: {history['mse'][-1]:.4f}")
        print(f"  Validation MSE: {history['val_mse'][-1]:.4f}")


class POISimulator:
    """Simulate and visualize POI localization predictions"""

    def __init__(self, model_path: str = 'final_poi_localizer_offset.keras',
                 config_path: str = 'poi_localizer_config.pkl'):
        """
        Initialize simulator with trained model and configuration

        Args:
            model_path: Path to saved model
            config_path: Path to saved configuration
        """
        # Load model and configuration
        try:
            self.model = keras.models.load_model(model_path, compile=False)
            print(f"✓ Model loaded from {model_path}")
        except:
            print(f"⚠ Could not load model from {model_path}")
            self.model = None

        try:
            config = joblib.load(config_path)
            self.scaler = config['scaler']
            self.window_size = config['window_size']
            self.stride = config['stride']
            self.poi_indices = config['poi_indices']
            self.search_window_size = config['search_window_size']
            print(f"✓ Configuration loaded from {config_path}")
        except:
            print(
                f"⚠ Could not load config from {config_path}, using defaults")
            self.window_size = 128
            self.stride = 16
            self.poi_indices = [0, 1, 3, 4, 5]
            self.search_window_size = 256
            self.scaler = None

        # Initialize DP module for feature generation
        try:
            from v4_dp import DP
            self.dp = DP()
            print("✓ DP module loaded")
        except:
            print("⚠ DP module not available, using mock features")
            self.dp = None

    def simulate_upstream_selection(self, true_poi: int, data_length: int,
                                    selection_mode: str = 'realistic') -> int:
        """
        Simulate upstream window selection around POI

        Args:
            true_poi: True POI location
            data_length: Length of data sequence
            selection_mode: 'realistic', 'perfect', or 'worst'

        Returns:
            window_start: Start position of selected window
        """
        if selection_mode == 'perfect':
            # Perfect centering
            window_start = true_poi - self.window_size // 2
        elif selection_mode == 'worst':
            # Worst case - POI at edge
            if np.random.random() > 0.5:
                window_start = true_poi - self.window_size + 10
            else:
                window_start = true_poi - 10
        else:  # realistic
            # Simulate realistic selection with some error
            estimation_error = np.random.randint(-self.search_window_size//4,
                                                 self.search_window_size//4)
            estimated_center = true_poi + estimation_error

            # Find best stride-aligned position near estimated center
            search_start = max(0, estimated_center -
                               self.search_window_size // 2)
            search_end = min(data_length - self.window_size,
                             search_start + self.search_window_size)

            possible_starts = list(
                range(int(search_start), int(search_end), self.stride))

            if not possible_starts:
                possible_starts = [max(0, min(true_poi - self.window_size // 2,
                                              data_length - self.window_size))]

            # Weight selection by proximity to POI
            weights = []
            for start in possible_starts:
                window_center = start + self.window_size // 2
                distance = abs(window_center - true_poi)
                weight = np.exp(-distance / (self.window_size / 4))
                weights.append(weight)

            weights = np.array(weights) / np.sum(weights)
            window_start = np.random.choice(possible_starts, p=weights)

        # Ensure within bounds
        window_start = max(
            0, min(window_start, data_length - self.window_size))
        return int(window_start)

    def generate_mock_features(self, length: int) -> np.ndarray:
        """Generate mock features if DP module not available"""
        # Create synthetic features with some structure
        t = np.linspace(0, 4*np.pi, length)
        features = np.column_stack([
            np.sin(t) + 0.1*np.random.randn(length),
            np.cos(t) + 0.1*np.random.randn(length),
            np.sin(2*t) + 0.1*np.random.randn(length),
            np.cos(2*t) + 0.1*np.random.randn(length),
            0.5*np.sin(3*t) + 0.1*np.random.randn(length),
            0.5*np.cos(3*t) + 0.1*np.random.randn(length),
            np.random.randn(length) * 0.5,
            np.random.randn(length) * 0.5,
        ])
        return features

    def prepare_window(self, features: np.ndarray, window_start: int) -> np.ndarray:
        """
        Prepare a window for model prediction

        Args:
            features: Full feature array
            window_start: Start position of window

        Returns:
            Normalized window ready for prediction
        """
        window_end = min(window_start + self.window_size, len(features))
        window = features[window_start:window_end]

        # Pad if necessary
        if window.shape[0] < self.window_size:
            padding = self.window_size - window.shape[0]
            window = np.pad(window, ((0, padding), (0, 0)), mode='edge')

        # Normalize if scaler available
        if self.scaler is not None:
            window = self.scaler.transform(window)
        else:
            # Simple standardization
            window = (window - np.mean(window, axis=0)) / \
                (np.std(window, axis=0) + 1e-8)

        return window[np.newaxis, ...]  # Add batch dimension

    def run_simulation(self, data_source=None, poi_source=None,
                       poi_index: int = 0, selection_mode: str = 'realistic',
                       verbose: bool = True) -> dict:
        """
        Run a single simulation

        Args:
            data_source: Path to data CSV or None for synthetic
            poi_source: Path to POI CSV or None for synthetic
            poi_index: Which POI to use (from poi_indices)
            selection_mode: Window selection mode
            verbose: Print details

        Returns:
            Dictionary with simulation results
        """
        # Load or generate data
        if data_source and poi_source and self.dp:
            # Load real data
            data_df = pd.read_csv(data_source)
            poi_array = pd.read_csv(poi_source, header=None).values
            features_df = self.dp.gen_features(data_df)
            features = features_df.values

            # Select POI
            if poi_index < len(self.poi_indices):
                true_poi = int(poi_array[self.poi_indices[poi_index]])
            else:
                true_poi = int(poi_array[self.poi_indices[0]])
        else:
            # Generate synthetic data
            data_length = np.random.randint(500, 1500)
            features = self.generate_mock_features(data_length)

            # Generate random POI location
            true_poi = np.random.randint(self.window_size,
                                         data_length - self.window_size)

        # Simulate upstream window selection
        window_start = self.simulate_upstream_selection(
            true_poi, len(features), selection_mode
        )
        window_center = window_start + self.window_size // 2
        true_offset = true_poi - window_center

        # Prepare window for prediction
        window = self.prepare_window(features, window_start)

        # Make prediction
        if self.model is not None:
            predicted_offset = float(
                self.model.predict(window, verbose=0)[0, 0])
        else:
            # Mock prediction for demonstration
            predicted_offset = true_offset + np.random.randn() * 5

        predicted_poi = window_center + predicted_offset
        prediction_error = abs(predicted_poi - true_poi)

        # Compile results
        results = {
            'features': features,
            'true_poi': true_poi,
            'window_start': window_start,
            'window_end': window_start + self.window_size,
            'window_center': window_center,
            'true_offset': true_offset,
            'predicted_offset': predicted_offset,
            'predicted_poi': predicted_poi,
            'prediction_error': prediction_error,
            'window': window[0],  # Remove batch dimension
            'selection_mode': selection_mode
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"SIMULATION RESULTS ({selection_mode} selection)")
            print(f"{'='*60}")
            print(f"Data length: {len(features)} samples")
            print(f"True POI location: {true_poi}")
            print(f"\nWindow Selection:")
            print(
                f"  Window: [{window_start}, {window_start + self.window_size}]")
            print(f"  Window center: {window_center}")
            print(f"  True offset from center: {true_offset:.1f}")
            print(f"\nPrediction:")
            print(f"  Predicted offset: {predicted_offset:.1f}")
            print(f"  Predicted POI location: {predicted_poi:.1f}")
            print(f"  Prediction error: {prediction_error:.1f} samples")
            print(
                f"  Within ±{self.stride}: {'✓' if prediction_error <= self.stride else '✗'}")

        return results

    def visualize_simulation(self, results: dict, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of simulation results

        Args:
            results: Dictionary from run_simulation
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

        # Extract results
        features = results['features']
        true_poi = results['true_poi']
        window_start = results['window_start']
        window_end = results['window_end']
        window_center = results['window_center']
        predicted_poi = results['predicted_poi']
        predicted_offset = results['predicted_offset']
        true_offset = results['true_offset']
        error = results['prediction_error']

        # Color scheme
        colors = {
            'window': 'lightblue',
            'true_poi': 'red',
            'predicted_poi': 'green',
            'center': 'orange',
            'search': 'yellow'
        }

        # 1. Full signal with window and POIs
        ax1 = fig.add_subplot(gs[0, :])

        # Plot signal (first feature dimension for visualization)
        x = np.arange(len(features))
        ax1.plot(x, features[:, 0], 'b-', alpha=0.5,
                 linewidth=0.8, label='Signal')

        # Show search region
        search_start = max(0, true_poi - self.search_window_size // 2)
        search_end = min(len(features), search_start + self.search_window_size)
        ax1.axvspan(search_start, search_end, alpha=0.1, color=colors['search'],
                    label='Search Region')

        # Show selected window
        window_rect = patches.Rectangle((window_start, ax1.get_ylim()[0]),
                                        self.window_size,
                                        ax1.get_ylim()[1] - ax1.get_ylim()[0],
                                        linewidth=2, edgecolor='blue',
                                        facecolor=colors['window'], alpha=0.3,
                                        label='Selected Window')
        ax1.add_patch(window_rect)

        # Mark positions
        ax1.axvline(x=true_poi, color=colors['true_poi'], linestyle='-',
                    linewidth=2, label=f'True POI ({true_poi})')
        ax1.axvline(x=predicted_poi, color=colors['predicted_poi'],
                    linestyle='--', linewidth=2,
                    label=f'Predicted POI ({predicted_poi:.1f})')
        ax1.axvline(x=window_center, color=colors['center'],
                    linestyle=':', linewidth=1.5, label='Window Center')

        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Feature Value')
        ax1.set_title(f'Full Signal View - Prediction Error: {error:.1f} samples',
                      fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', ncol=3)
        ax1.grid(True, alpha=0.3)

        # 2. Zoomed view of window region
        ax2 = fig.add_subplot(gs[1, 0])

        zoom_start = max(0, window_start - 50)
        zoom_end = min(len(features), window_end + 50)
        zoom_x = np.arange(zoom_start, zoom_end)

        ax2.plot(zoom_x, features[zoom_start:zoom_end, 0], 'b-', linewidth=1.5)
        ax2.axvspan(window_start, window_end,
                    alpha=0.3, color=colors['window'])
        ax2.axvline(
            x=true_poi, color=colors['true_poi'], linestyle='-', linewidth=2)
        ax2.axvline(x=predicted_poi, color=colors['predicted_poi'],
                    linestyle='--', linewidth=2)
        ax2.axvline(x=window_center, color=colors['center'],
                    linestyle=':', linewidth=1.5)

        # Add offset annotations
        ax2.annotate(f'True Offset:\n{true_offset:.1f}',
                     xy=(window_center, ax2.get_ylim()[0]),
                     xytext=(window_center - 20, ax2.get_ylim()[0] +
                             (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.2),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     fontsize=9, ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='red', alpha=0.8))

        ax2.annotate(f'Predicted Offset:\n{predicted_offset:.1f}',
                     xy=(window_center, ax2.get_ylim()[1]),
                     xytext=(window_center + 20, ax2.get_ylim()[1] -
                             (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.2),
                     arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                     fontsize=9, ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='green', alpha=0.8))

        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Feature Value')
        ax2.set_title('Zoomed Window View', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Window features heatmap
        ax3 = fig.add_subplot(gs[1, 1])

        window_features = results['window']
        im = ax3.imshow(window_features.T, aspect='auto', cmap='viridis',
                        extent=[0, self.window_size,
                                window_features.shape[1], 0])

        # Mark true and predicted positions within window
        true_pos_in_window = true_poi - window_start
        pred_pos_in_window = predicted_poi - window_start

        if 0 <= true_pos_in_window < self.window_size:
            ax3.axvline(x=true_pos_in_window, color='red', linestyle='-',
                        linewidth=2, alpha=0.7)
        if 0 <= pred_pos_in_window < self.window_size:
            ax3.axvline(x=pred_pos_in_window, color='lime', linestyle='--',
                        linewidth=2, alpha=0.7)
        ax3.axvline(x=self.window_size/2, color='orange', linestyle=':',
                    linewidth=1.5, alpha=0.7)

        ax3.set_xlabel('Window Position')
        ax3.set_ylabel('Feature Dimension')
        ax3.set_title('Window Feature Heatmap', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        # 4. Offset distribution plot
        ax4 = fig.add_subplot(gs[2, 0])

        # Create offset visualization
        offset_range = np.linspace(-self.window_size/2,
                                   self.window_size/2, 100)
        ax4.fill_between(offset_range, 0, 1, where=(np.abs(offset_range) <= self.stride),
                         alpha=0.3, color='green', label=f'Target Zone (±{self.stride})')

        ax4.axvline(x=true_offset, color='red', linestyle='-', linewidth=3,
                    label=f'True: {true_offset:.1f}')
        ax4.axvline(x=predicted_offset, color='green', linestyle='--', linewidth=3,
                    label=f'Predicted: {predicted_offset:.1f}')
        ax4.axvline(x=0, color='orange', linestyle=':',
                    linewidth=1.5, alpha=0.5)

        # Add error bar
        ax4.plot([true_offset, predicted_offset],
                 [0.5, 0.5], 'k-', linewidth=2)
        ax4.text((true_offset + predicted_offset)/2, 0.55, f'Error: {error:.1f}',
                 ha='center', fontsize=10, fontweight='bold')

        ax4.set_xlim(-self.window_size/2, self.window_size/2)
        ax4.set_ylim(0, 1)
        ax4.set_xlabel('Offset from Window Center')
        ax4.set_ylabel('Relative Position')
        ax4.set_title('Offset Prediction Visualization',
                      fontsize=11, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

        # 5. Performance metrics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        # Create metrics text
        metrics_text = f"""
        SIMULATION METRICS
        {'='*30}
        
        Selection Mode: {results['selection_mode'].upper()}
        
        Window Selection:
          • Window Size: {self.window_size}
          • Stride: {self.stride}
          • Window Range: [{window_start}, {window_end}]
          • Center Position: {window_center}
        
        POI Localization:
          • True POI: {true_poi}
          • True Offset: {true_offset:.2f}
          • Predicted Offset: {predicted_offset:.2f}
          • Predicted POI: {predicted_poi:.2f}
        
        Performance:
          • Absolute Error: {error:.2f} samples
          • Relative Error: {(error/self.window_size)*100:.1f}%
          • Within ±{self.stride}: {'✓ YES' if error <= self.stride else '✗ NO'}
          • Within ±{self.stride/2}: {'✓ YES' if error <= self.stride/2 else '✗ NO'}
        """

        # Color code based on performance
        bg_color = 'lightgreen' if error <= self.stride else 'lightyellow' if error <= self.stride*2 else 'lightcoral'

        ax5.text(0.1, 0.5, metrics_text, transform=ax5.transAxes,
                 fontsize=10, verticalalignment='center',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color,
                           edgecolor='black', alpha=0.7))

        # Overall title
        fig.suptitle(f'POI Localization Simulation - {results["selection_mode"].title()} Window Selection',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def run_multiple_simulations(self, n_simulations: int = 10,
                                 selection_modes: List[str] = ['realistic'],
                                 verbose: bool = False) -> pd.DataFrame:
        """
        Run multiple simulations and collect statistics

        Args:
            n_simulations: Number of simulations per mode
            selection_modes: List of selection modes to test
            verbose: Print individual results

        Returns:
            DataFrame with simulation results
        """
        results_list = []

        for mode in selection_modes:
            print(
                f"\nRunning {n_simulations} simulations with {mode} selection...")

            for i in range(n_simulations):
                results = self.run_simulation(
                    selection_mode=mode,
                    verbose=verbose
                )

                results_list.append({
                    'simulation': i,
                    'mode': mode,
                    'true_offset': results['true_offset'],
                    'predicted_offset': results['predicted_offset'],
                    'error': results['prediction_error'],
                    'within_stride': results['prediction_error'] <= self.stride,
                    'within_half_stride': results['prediction_error'] <= self.stride/2,
                })

        df = pd.DataFrame(results_list)

        # Print summary statistics
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)

        for mode in selection_modes:
            mode_df = df[df['mode'] == mode]
            print(f"\n{mode.upper()} Selection Mode:")
            print(
                f"  Mean Error: {mode_df['error'].mean():.2f} ± {mode_df['error'].std():.2f}")
            print(f"  Median Error: {mode_df['error'].median():.2f}")
            print(f"  Max Error: {mode_df['error'].max():.2f}")
            print(
                f"  Within ±{self.stride}: {mode_df['within_stride'].mean():.1%}")
            print(
                f"  Within ±{self.stride/2}: {mode_df['within_half_stride'].mean():.1%}")

        return df

    def plot_simulation_statistics(self, df: pd.DataFrame):
        """
        Plot statistics from multiple simulations

        Args:
            df: DataFrame from run_multiple_simulations
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        modes = df['mode'].unique()
        colors_map = {'realistic': 'blue', 'perfect': 'green', 'worst': 'red'}

        # 1. Error distribution by mode
        ax1 = axes[0, 0]
        for mode in modes:
            mode_data = df[df['mode'] == mode]['error']
            ax1.hist(mode_data, bins=20, alpha=0.6,
                     label=mode.title(), color=colors_map.get(mode, 'gray'))
        ax1.axvline(x=self.stride, color='black', linestyle='--',
                    label=f'Stride ({self.stride})')
        ax1.set_xlabel('Prediction Error (samples)')
        ax1.set_ylabel('Count')
        ax1.set_title('Error Distribution by Selection Mode')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Box plot of errors
        ax2 = axes[0, 1]
        data_to_plot = [df[df['mode'] == mode]
                        ['error'].values for mode in modes]
        bp = ax2.boxplot(data_to_plot, labels=[m.title() for m in modes])
        ax2.axhline(y=self.stride, color='red', linestyle='--',
                    label=f'Stride ({self.stride})')
        ax2.set_ylabel('Prediction Error (samples)')
        ax2.set_title('Error Distribution Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Accuracy within thresholds
        ax3 = axes[1, 0]
        thresholds = [self.stride/4, self.stride/2, self.stride, self.stride*2]
        x = np.arange(len(thresholds))
        width = 0.25

        for i, mode in enumerate(modes):
            mode_df = df[df['mode'] == mode]
            accuracies = [
                (mode_df['error'] <= t).mean() for t in thresholds
            ]
            ax3.bar(x + i*width, accuracies, width,
                    label=mode.title(),
                    color=colors_map.get(mode, 'gray'), alpha=0.7)

        ax3.set_xlabel('Error Threshold')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy Within Different Thresholds')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([f'±{t:.1f}' for t in thresholds])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)

        # 4. Scatter plot of true vs predicted offsets
        ax4 = axes[1, 1]
        for mode in modes:
            mode_df = df[df['mode'] == mode]
            ax4.scatter(mode_df['true_offset'], mode_df['predicted_offset'],
                        alpha=0.6, label=mode.title(),
                        color=colors_map.get(mode, 'gray'))

        # Add perfect prediction line
        lims = [df['true_offset'].min(), df['true_offset'].max()]
        ax4.plot(lims, lims, 'k-', alpha=0.5, label='Perfect Prediction')
        ax4.set_xlabel('True Offset')
        ax4.set_ylabel('Predicted Offset')
        ax4.set_title('True vs Predicted Offsets')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle('Multi-Simulation Statistics',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    """Main training pipeline with realistic offset training"""

    # Configuration
    WINDOW_SIZE = 128  # Match your upstream system
    STRIDE = 16  # Match your upstream system
    SEARCH_WINDOW_SIZE = 256  # Size of search region in upstream system
    DATA_DIR = './content/static/train'
    # Use POI 1,2,4,5,6 (0-indexed, skipping 3rd)
    POI_INDICES = [0, 1, 3, 4, 5]
    USE_HYPERPARAMETER_TUNING = True
    EPOCHS = 100
    BATCH_SIZE = 32
    MAX_TRIALS = 30
    AUGMENTATION_FACTOR = 3  # Number of different window positions per POI

    print("=== POI Localizer Training Pipeline with Realistic Offsets ===")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Stride: {STRIDE}")
    print(f"Search window size: {SEARCH_WINDOW_SIZE}")
    print(f"Augmentation factor: {AUGMENTATION_FACTOR}")
    print("Using v4_dp module for data loading and feature generation\n")

    # Load data with realistic offset distribution
    print("Loading data with realistic window offsets...")
    loader = POIDataLoader(
        window_size=WINDOW_SIZE,
        poi_indices=POI_INDICES,
        stride=STRIDE,
        search_window_size=SEARCH_WINDOW_SIZE
    )

    X, y = loader.load_all_data(
        DATA_DIR,
        simulate_upstream=True,
        augmentation_factor=AUGMENTATION_FACTOR
    )

    print(f"\nDataset Statistics:")
    print(f"  Total windows: {len(X)}")
    print(f"  Window shape: {X.shape}")
    print(f"  Offset range: [{y.min():.1f}, {y.max():.1f}]")
    print(f"  Offset std: {y.std():.2f}")
    print(f"  Mean absolute offset: {np.mean(np.abs(y)):.2f}\n")

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, shuffle=True
    )

    print("Data Split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples\n")

    # Create and train model
    localizer = POILocalizer(
        input_shape=(X.shape[1], X.shape[2]),
        stride=STRIDE
    )

    # Visualize offset distributions
    print("Visualizing offset distributions...")

    if USE_HYPERPARAMETER_TUNING:
        model, tuner = localizer.train_with_tuning(
            X_train, y_train, X_val, y_val,
            epochs=EPOCHS,
            final_epochs=FINAL_EPOCHS,
            batch_size=BATCH_SIZE,
            max_trials=MAX_TRIALS,
            tuner_type='bayesian'  # Can be 'bayesian', 'random', or 'hyperband'
        )

        # Display tuning results summary
        print("\n=== Hyperparameter Tuning Results ===")
        tuner.results_summary(num_trials=10)  # Show top 10 trials

    else:
        model = localizer.train(
            X_train, y_train, X_val, y_val,
            epochs=FINAL_EPOCHS, batch_size=BATCH_SIZE
        )

    # Evaluate on test set
    print("\n=== Test Set Performance ===")
    metrics = localizer.evaluate(X_test, y_test)

    print("\nRegression Metrics:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Mean Error: {metrics['mean_error']:.4f}")
    print(f"  Std Error: {metrics['std_error']:.4f}")

    print("\nPercentile Errors:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {metrics[f'error_p{p}']:.4f}")

    print("\nAccuracy within thresholds:")
    for t in [0.5, 1, 2, 3, 5, 8, STRIDE/2, STRIDE, STRIDE*2]:
        key = f'accuracy_within_{t}'
        if key in metrics:
            print(f"  Within ±{t:.1f}: {metrics[key]:.2%}")

    # Visualize training history
    print("\n=== Visualizing Training History ===")
    localizer.plot_training_history()

    # If hyperparameter tuning was used, visualize tuning results
    if USE_HYPERPARAMETER_TUNING:
        print("\n=== Visualizing Hyperparameter Tuning Results ===")
        localizer.plot_tuning_history(tuner, top_n=10)

    # Save final model with appropriate name
    if USE_HYPERPARAMETER_TUNING:
        model_name = 'final_poi_localizer_tuned.keras'
    else:
        model_name = 'final_poi_localizer_offset.keras'

    model.save(model_name)
    print(f"\nModel saved as '{model_name}'")

    # Save scaler and configuration for inference
    import joblib
    config_dict = {
        'scaler': loader.scaler,
        'window_size': WINDOW_SIZE,
        'stride': STRIDE,
        'poi_indices': POI_INDICES,
        'search_window_size': SEARCH_WINDOW_SIZE,
        'augmentation_factor': AUGMENTATION_FACTOR,
        'hyperparameter_tuned': USE_HYPERPARAMETER_TUNING
    }

    # If tuned, also save the best hyperparameters
    if USE_HYPERPARAMETER_TUNING:
        config_dict['best_hyperparameters'] = tuner.get_best_hyperparameters()[
            0].values

    joblib.dump(config_dict, 'poi_localizer_config.pkl')
    print("Configuration saved as 'poi_localizer_config.pkl'")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"The model is now trained to predict offsets from realistic window positions")
    print(f"that match your upstream stride-based window selection system.")
    print(f"\nKey Results:")
    print(f"  - Final Validation MAE: {metrics['mae']:.4f}")
    print(
        f"  - Accuracy within ±{STRIDE} samples: {metrics[f'accuracy_within_{STRIDE}']:.2%}")
    print(
        f"  - Model type: {'Hyperparameter-tuned' if USE_HYPERPARAMETER_TUNING else 'Standard'}")

    if USE_HYPERPARAMETER_TUNING:
        print(
            f"\nBest hyperparameters have been saved and can be reused for future training.")
        print(f"Check 'best_hyperparameters.json' for details.")

    """Main function to run simulations"""

    print("="*60)
    print("POI LOCALIZATION SIMULATION SYSTEM")
    print("="*60)

    # Initialize simulator
    simulator = POISimulator()

    # Run single simulation with visualization
    print("\n1. Running single simulation with visualization...")
    data, poi = DP.load_content('content/static/valid')
    results = simulator.run_simulation(data_source=data[0], poi_source=poi[0],
                                       selection_mode='realistic', verbose=True)
    simulator.visualize_simulation(results)

    # Run multiple simulations for statistics
    print("\n2. Running multiple simulations for statistics...")
    df = simulator.run_multiple_simulations(
        n_simulations=50,
        selection_modes=['realistic', 'perfect', 'worst'],
        verbose=False
    )

    # Plot statistics
    print("\n3. Plotting simulation statistics...")
    simulator.plot_simulation_statistics(df)

    # Interactive mode - run custom simulations
    print("\n" + "="*60)
    print("INTERACTIVE SIMULATION MODE")
    print("="*60)
    print("\nYou can now run custom simulations:")
    print("  - Change selection_mode: 'realistic', 'perfect', or 'worst'")
    print("  - Adjust POI index to test different POIs")
    print("  - Use real data if available")

    while True:
        user_input = input("\nRun another simulation? (y/n): ").strip().lower()
        if user_input != 'y':
            break

        mode = input(
            "Selection mode (realistic/perfect/worst) [realistic]: ").strip() or 'realistic'

        results = simulator.run_simulation(selection_mode=mode, verbose=True)
        simulator.visualize_simulation(results)

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
