from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.losses import CategoricalCrossentropy
from keras.layers import Dense, Dropout, LSTM, Reshape, BatchNormalization, Concatenate, GaussianNoise, Input, Add, LayerNormalization, Multiply
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, confusion_matrix
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from keras.regularizers import l2
from keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from scipy.stats import entropy, linregress
import shap
from concurrent.futures import ThreadPoolExecutor
from keras.callbacks import Callback
import matplotlib.ticker as ticker


class HybridBatchSizeCallback(Callback):
    def __init__(self, model, initial_batch_size, final_batch_size, switch_epoch):
        """
        Callback to change batch size during training.

        Parameters:
        - model: The Keras model being trained.
        - initial_batch_size: Batch size for the initial training phase.
        - final_batch_size: Batch size for the later training phase.
        - switch_epoch: Epoch at which to switch to the larger batch size.
        """
        super().__init__()
        self.model = model
        self.initial_batch_size = initial_batch_size
        self.final_batch_size = final_batch_size
        self.switch_epoch = switch_epoch

    def on_epoch_begin(self, epoch, logs=None):
        # Switch batch size at the defined epoch
        if epoch == self.switch_epoch:
            self.params['batch_size'] = self.final_batch_size
            print(f"Switching to batch size: {self.final_batch_size}")
        elif epoch == 0:
            print(f"Using initial batch size: {self.initial_batch_size}")


def extract_features(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=[
        "Date",
        "Time",
        "Ambient",
        "Temperature",
        "Peak Magnitude (RAW)",
    ],
        inplace=True, errors='ignore')

    features = {}

    # Columns to analyze
    target_columns = ["Relative_time", "Resonance_Frequency", "Dissipation"]
    for col in target_columns:
        if col in df.columns:
            column_features = {}
            column_data = df[col].dropna()

            # Basic statistics
            column_features[f'{col}_mean'] = column_data.mean()
            column_features[f'{col}_std'] = column_data.std()
            column_features[f'{col}_min'] = column_data.min()
            column_features[f'{col}_max'] = column_data.max()

            # Advanced statistics
            column_features[f'{col}_median'] = column_data.median()
            column_features[f'{col}_skew'] = column_data.skew()
            column_features[f'{col}_kurtosis'] = column_data.kurtosis()
            column_features[f'{col}_range'] = column_data.max() - \
                column_data.min()

            # Quantile statistics
            Q1 = column_data.quantile(0.25)
            Q3 = column_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((column_data < (Q1 - 1.5 * IQR)) |
                        (column_data > (Q3 + 1.5 * IQR))).sum()
            column_features[f'{col}_num_outliers'] = outliers

            # Entropy
            column_features[f'{col}_entropy'] = entropy(column_data.value_counts(
                normalize=True)) if column_data.nunique() > 1 else 0

            # Signal-to-noise ratio
            column_features[f'{col}_signal_to_noise'] = column_data.mean(
            ) / column_data.std() if column_data.std() != 0 else 0

            # Rolling statistics (time-series inspired)
            rolling_mean = column_data.rolling(
                window=50, min_periods=1).mean().mean()
            column_features[f'{col}_rolling_mean'] = rolling_mean

            rolling_std = column_data.rolling(
                window=50, min_periods=1).std().mean()
            column_features[f'{col}_rolling_std'] = rolling_std

            # Lag difference
            lag_diff = column_data.diff().abs().mean()
            column_features[f'{col}_lag_diff_mean'] = lag_diff

            # Trend analysis
            if len(column_data) > 1:
                trend = linregress(range(len(column_data)),
                                   column_data.values).slope
            else:
                trend = 0
            column_features[f'{col}_trend'] = trend

            # End-focused statistics
            tail_mean = column_data.tail(100).mean()
            head_mean = column_data.head(100).mean()
            column_features[f'{col}_mean_diff'] = tail_mean - head_mean

            tail_std = column_data.tail(100).std()
            head_std = column_data.head(100).std()
            column_features[f'{col}_std_diff'] = tail_std - head_std

            # Add column features to the overall feature dictionary
            features.update(column_features)

    return features


def load_and_prepare_data(dataset_paths):
    X = []
    y = []

    def process_path(label, path):
        local_X = []
        local_y = []
        for root, _, files in os.walk(path):
            files = [f for f in files if f.endswith(
                ".csv") and not f.endswith('_poi.csv')]
            for file in files:
                file_path = os.path.join(root, file)
                features = extract_features(file_path)
                local_X.append(features)
                local_y.append(label)
        return local_X, local_y

    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_path, label, path)
                   for label, path in dataset_paths.items()]
        for future in tqdm(futures, desc="Processing datasets"):
            result_X, result_y = future.result()
            X.extend(result_X)
            y.extend(result_y)

    X_df = pd.DataFrame(X)
    X_df.fillna(0, inplace=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, "label_encoder.pkl")

    return X_df, y_encoded


def build_advanced_model(input_dim,
                         num_classes,
                         dense_units,
                         dropout_rate,
                         learning_rate,
                         optimizer_name,
                         num_layers=2,
                         activation='swish',
                         weight_initializer='he_uniform',
                         use_feature_gating=True):
    """Improved advanced model with attention and SE blocks."""
    # Optimizer selection
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)

    # Input layer
    inputs = Input(shape=(input_dim,))

    # Wide component
    wide = Dense(num_classes, activation='linear',
                 kernel_regularizer=l2(0.01))(inputs)

    # Deep component with SE block
    x = Dense(dense_units, activation=activation,
              kernel_initializer=weight_initializer, kernel_regularizer=l2(0.01))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    squeeze = Dense(dense_units // 16, activation='relu')(x)
    excite = Dense(dense_units, activation='sigmoid')(squeeze)
    x = Multiply()([x, excite])

    # Add attention and residual blocks
    for _ in range(int(num_layers)):
        residual = x
        x = Dense(dense_units, activation=activation,
                  kernel_initializer=weight_initializer, kernel_regularizer=l2(0.01))(x)
        x = LayerNormalization()(x)
        if use_feature_gating:
            gate = Dense(dense_units, activation='sigmoid')(inputs)
            x = Multiply()([x, gate])
        x = Add()([x, residual])
        x = Dropout(dropout_rate)(x)

    # Attention mechanism
    attention = Dense(dense_units, activation='tanh')(x)
    attention = Dense(1, activation='softmax')(attention)
    x = Multiply()([x, attention])

    # Final dense layers
    x = Dense(dense_units // 2, activation='swish',
              kernel_initializer=weight_initializer, kernel_regularizer=l2(0.01))(x)
    x = LayerNormalization()(x)

    # Concatenate wide and deep outputs
    combined = Concatenate()([wide, x])
    outputs = Dense(num_classes, activation='softmax')(combined)

    # Model compilation
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(
        label_smoothing=0.1), metrics=['accuracy'])
    return model


class OptimizationMonitor:
    def __init__(self):
        self.losses = []
        self.accuracies = []
        self.best_loss = float('inf')
        self.best_accuracy = 0.0

        # Set up the figure and subplots
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 12))
        self.fig.suptitle('Hyperparameter Optimization Monitor',
                          fontsize=16, fontweight='bold')

        # Configure styles
        # print(plt.style.available)

        plt.style.use("seaborn-v0_8-darkgrid")
        plt.ion()  # Enable interactive mode

    def update(self, trial_number, current_loss, current_accuracy):
        # Update loss and accuracy lists
        self.losses.append(current_loss)
        self.accuracies.append(current_accuracy)
        self.best_loss = min(self.best_loss, current_loss)
        self.best_accuracy = max(self.best_accuracy, current_accuracy)

        # Update Loss Plot
        self.ax[0].clear()
        self.ax[0].plot(self.losses, label='Loss', marker='o',
                        color='orange', linewidth=2)
        self.ax[0].axhline(self.best_loss, color='red', linestyle='--', linewidth=1.5,
                           label=f'Best Loss: {self.best_loss:.4f}')
        self.ax[0].set_title('Loss Progress', fontsize=14, fontweight='bold')
        self.ax[0].set_xlabel('Trial Number', fontsize=12)
        self.ax[0].set_ylabel('Loss', fontsize=12)
        self.ax[0].legend(fontsize=10)
        self.ax[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        self.ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Update Accuracy Plot
        self.ax[1].clear()
        self.ax[1].plot(self.accuracies, label='Accuracy',
                        marker='s', color='green', linewidth=2)
        self.ax[1].axhline(self.best_accuracy, color='blue', linestyle='--', linewidth=1.5,
                           label=f'Best Accuracy: {self.best_accuracy:.4f}')
        self.ax[1].set_title('Accuracy Progress',
                             fontsize=14, fontweight='bold')
        self.ax[1].set_xlabel('Trial Number', fontsize=12)
        self.ax[1].set_ylabel('Accuracy', fontsize=12)
        self.ax[1].legend(fontsize=10)
        self.ax[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        self.ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Adjust layout and refresh
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the title
        plt.pause(0.01)

    def finalize(self):
        plt.ioff()  # Disable interactive mode
        plt.show()


def optimize_hyperparameters(X_train,
                             y_train_cat,
                             X_test,
                             y_test_cat,
                             input_dim,
                             num_classes):
    """Optimize model hyperparameters using Hyperopt."""

    monitor = OptimizationMonitor()

    def objective(params):
        model = build_advanced_model(
            input_dim=input_dim,
            num_classes=num_classes,
            dense_units=int(params['dense_units']),
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate'],
            optimizer_name=params['optimizer'],
            num_layers=int(params['num_layers']),
            activation=params['activation'],
            weight_initializer=params['weight_initializer']
        )

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        initial_batch_size = 32
        final_batch_size = 128
        switch_epoch = 50

        hybrid_batch_size = HybridBatchSizeCallback(
            model=model,
            initial_batch_size=initial_batch_size,
            final_batch_size=final_batch_size,
            switch_epoch=switch_epoch
        )

        # Train model
        model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=50,
            batch_size=initial_batch_size,
            callbacks=[early_stopping, lr_scheduler, hybrid_batch_size],
            verbose=0
        )

        loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
        trial_number = len(monitor.losses) + 1
        monitor.update(trial_number, loss, accuracy)
        return {'loss': loss, 'status': STATUS_OK}

    # Expanded Hyperparameter search space
    space = {
        'dense_units': hp.quniform('dense_units', 32, 1024, 16),
        'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.7),
        'learning_rate': hp.loguniform('learning_rate', -6, -2),
        'optimizer': hp.choice('optimizer', ['adam', 'rmsprop', 'sgd']),
        'num_layers': hp.quniform('num_layers', 1, 5, 1),
        'activation': hp.choice('activation', ['relu', 'tanh', 'selu', 'elu', 'leaky_relu']),
        'weight_initializer': hp.choice('weight_initializer', ['glorot_uniform', 'he_uniform', 'lecun_uniform', 'random_normal']),
    }

    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )

    monitor.finalize()

    return {
        'dense_units': int(best_params['dense_units']),
        'dropout_rate': best_params['dropout_rate'],
        'learning_rate': best_params['learning_rate'],
        'optimizer': ['adam', 'rmsprop', 'sgd'][best_params['optimizer']],
        'num_layers': int(best_params['num_layers']),
        'activation': ['relu', 'tanh', 'selu', 'elu', 'leaky_relu'][best_params['activation']],
        'weight_initializer': ['glorot_uniform', 'he_uniform', 'lecun_uniform', 'random_normal'][best_params['weight_initializer']],
    }


def train_advanced_model(X, y, feature_names=""):
    """Train the model with optimized hyperparameters."""
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, "csv_scaler.pkl")

    # One-hot encode labels
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    # Optimize hyperparameters
    best_params = optimize_hyperparameters(
        X_train, y_train_cat, X_test, y_test_cat,
        input_dim=X_train.shape[1],
        num_classes=y_train_cat.shape[1]
    )

    # Build model with optimized parameters
    model = build_advanced_model(
        input_dim=X_train.shape[1],
        num_classes=y_train_cat.shape[1],
        dense_units=best_params['dense_units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        optimizer_name=best_params['optimizer']
    )

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    initial_batch_size = 32
    final_batch_size = 128
    switch_epoch = 50
    hybrid_batch_size = HybridBatchSizeCallback(
        model=model,
        initial_batch_size=initial_batch_size,
        final_batch_size=final_batch_size,
        switch_epoch=switch_epoch
    )
    # Train model
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=200,
        batch_size=initial_batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping, lr_scheduler, hybrid_batch_size],
        verbose=1
    )

    # Save model
    model.save("csv_advanced_classifier_model.h5")

    # Evaluate model
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    class_names = [str(i) for i in range(len(np.unique(y)))]
    plot_confusion_matrix(cm, class_names)
    plot_feature_target_correlation(X, y, 10)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    return model


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_feature_target_correlation(X, y, top_n=10):
    """
    Plots and prints the correlation of features with the target variable.

    Parameters:
    - X: pd.DataFrame
        DataFrame containing the feature data.
    - y: pd.Series or pd.DataFrame
        Series or single-column DataFrame containing the target variable.
    - top_n: int, optional
        The number of most correlative features to print. If None, print all.

    Returns:
    - None
    """
    # Create a DataFrame by appending the target column to features
    X_with_target = X.copy()
    X_with_target['Target'] = y

    # Calculate the correlation matrix and extract correlation with the target
    correlation_matrix = X_with_target.corr()['Target'].drop('Target')

    # Sort correlations by absolute value (strongest first)
    sorted_correlation = correlation_matrix.abs().sort_values(ascending=False)

    # Print the most correlative features
    print("\nMost Correlative Features:\n")
    if top_n:
        print(sorted_correlation.head(top_n))
    else:
        print(sorted_correlation)

    # Plot the correlations
    plt.figure(figsize=(10, 8))
    sorted_correlation.plot(kind='barh', color='skyblue')
    plt.title("Feature Correlation with Target")
    plt.xlabel("Correlation")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


def predict(input_file_path, model_path="csv_advanced_classifier_model.h5", scaler_path="csv_scaler.pkl", label_encoder_path="label_encoder.pkl"):
    """
    Predict the class of a given input CSV file using the trained model.

    Parameters:
    - input_file_path: str, path to the input CSV file.
    - model_path: str, path to the trained model file.
    - scaler_path: str, path to the saved scaler file.
    - label_encoder_path: str, path to the saved label encoder file.

    Returns:
    - predicted_label: str, the predicted class label.
    - probabilities: dict, class probabilities for the input.
    """

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load the scaler and label encoder
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)

    # Extract features from the input file
    features = extract_features(input_file_path)

    # Ensure all required features are present in the correct order
    feature_df = pd.DataFrame([features])
    feature_df.fillna(0, inplace=True)  # Fill missing values with 0

    # Scale features
    scaled_features = scaler.transform(feature_df)

    # Make predictions
    # Single input, get first output
    probabilities = model.predict(scaled_features)[0]
    predicted_class_index = np.argmax(probabilities)

    # Decode the predicted class label
    predicted_label = label_encoder.inverse_transform(
        [predicted_class_index])[0]

    # Map probabilities to class labels
    class_probabilities = {
        label_encoder.inverse_transform([i])[0]: prob
        for i, prob in enumerate(probabilities)
    }

    return predicted_label, class_probabilities


if __name__ == "__main__":
    dataset_paths = {
        "full_fill": "content/dropbox_dump",
        "no_fill": "content/no_fill",
        "channel_1_partial": "content/channel_1",
        "channel_2_partial": "content/channel_2",
    }

    X, y = load_and_prepare_data(dataset_paths)
    train_advanced_model(X, y)

    test_file = r"content/dropbox_dump/00001/DD240125W1_C5_OLDBSA367_3rd.csv"
    predicted_class = predict(test_file)
    print(f"Predicted dataset type: {predicted_class}")
