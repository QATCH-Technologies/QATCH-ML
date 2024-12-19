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


# Step 1: Feature Extraction Function


def extract_features(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=[
        "Date",
        "Time",
        "Ambient",
        "Temperature",
        "Peak Magnitude (RAW)",
    ], inplace=True)

    features = {}

    # Basic statistics
    features['num_rows'] = df.shape[0]
    features['num_cols'] = df.shape[1]
    features['missing_values'] = df.isnull().sum().sum()
    features['mean'] = df.mean(numeric_only=True).mean()
    features['std'] = df.std(numeric_only=True).mean()
    features['min'] = df.min(numeric_only=True).min()
    features['max'] = df.max(numeric_only=True).max()

    # Fill types
    features['num_zeros'] = (df == 0).sum().sum()
    features['num_unique_cols'] = df.nunique().mean()

    if not df.empty:
        # Advanced statistics
        features['median'] = df.median(numeric_only=True).mean()
        features['skew'] = df.skew(numeric_only=True).mean()
        features['kurtosis'] = df.kurtosis(numeric_only=True).mean()
        features['range'] = df.max(
            numeric_only=True).max() - df.min(numeric_only=True).min()
        features['total_sum'] = df.sum(numeric_only=True).sum()
        features['total_variance'] = df.var(numeric_only=True).sum()
        features['unique_values_ratio'] = df.nunique().mean() / \
            features['num_rows']
        features['correlation_mean'] = df.corr(
            numeric_only=True).abs().mean().mean()

        # Quantile statistics
        Q1 = df.quantile(0.25, numeric_only=True)
        Q3 = df.quantile(0.75, numeric_only=True)
        Q1, Q3 = Q1.align(Q3, axis=0)  # Align indices
        IQR = Q3 - Q1
        df_aligned, IQR_aligned = df.align(
            IQR, axis=1, copy=False)  # Align DataFrame with IQR
        outliers = ((df_aligned < (Q1 - 1.5 * IQR_aligned)) |
                    (df_aligned > (Q3 + 1.5 * IQR_aligned))).sum().sum()
        features['num_outliers'] = outliers

        # Entropy
        entropy_values = df.select_dtypes(include=['number']).apply(
            lambda col: entropy(col.value_counts(
                normalize=True)) if col.nunique() > 1 else 0
        )
        features['entropy_mean'] = entropy_values.mean()

        # Signal-to-noise ratio
        features['signal_to_noise'] = (
            df.mean(numeric_only=True) / df.std(numeric_only=True)).mean()

        # Time-series inspired features (if data is sequential)
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        rolling = numeric_df.rolling(window=50, min_periods=1).mean()
        features['rolling_mean'] = rolling.mean().mean()
        rolling_std = numeric_df.rolling(window=50, min_periods=1).std()
        features['rolling_std'] = rolling_std.mean().mean()

        lag_diff = numeric_df.diff().abs().mean().mean()
        features['lag_diff_mean'] = lag_diff

        trends = [
            linregress(range(len(numeric_df)),
                       numeric_df[col].dropna().values).slope
            for col in numeric_df.columns
        ]
        features['avg_trend'] = sum(trends) / len(trends) if trends else 0

        # Pairwise correlation analysis
        corr_matrix = numeric_df.corr().abs()
        strong_corr = (corr_matrix > 0.8).sum().sum() - len(numeric_df.columns)
        features['strong_correlation_pairs'] = strong_corr

        # Polynomial features
        poly_sum = (numeric_df ** 2).sum().mean()
        features['poly_sum_mean'] = poly_sum

        # Missing data patterns
        missing_percentage = (df.isnull().sum() / len(df)).mean()
        features['missing_data_percentage'] = missing_percentage

        # Missing data patterns
        missing_streaks = df.isnull().astype(int).apply(
            lambda col: col.groupby(
                (col != col.shift()).cumsum()).cumsum().max()
        )
        features['max_missing_streak'] = missing_streaks.max()

        # End-focused statistics
        tail_mean = numeric_df.tail(100).mean()
        head_mean = numeric_df.head(100).mean()
        tail_mean, head_mean = tail_mean.align(
            head_mean, axis=0)  # Align indices
        features['mean_diff'] = tail_mean.mean() - head_mean.mean()

        tail_std = numeric_df.tail(100).std()
        head_std = numeric_df.head(100).std()
        tail_std, head_std = tail_std.align(head_std, axis=0)  # Align indices
        features['std_diff'] = tail_std.mean() - head_std.mean()
    else:
        features.update({key: 0 for key in [
            'median', 'skew', 'kurtosis', 'range', 'total_sum',
            'total_variance', 'unique_values_ratio', 'correlation_mean',
            'q25', 'q75', 'iqr', 'num_outliers', 'entropy_mean',
            'signal_to_noise', 'rolling_mean', 'rolling_std',
            'lag_diff_mean', 'avg_trend', 'strong_correlation_pairs',
            'poly_sum_mean', 'missing_data_percentage', 'max_missing_streak',
            'end_mean', 'end_std', 'end_min', 'end_max', 'mean_diff', 'std_diff'
        ]})

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


def build_advanced_model(input_dim, num_classes, dense_units, dropout_rate, learning_rate, optimizer_name, use_feature_gating=True):
    """Build an improved wide-and-deep model for tabular data."""

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

    # Deep component
    x = Dense(dense_units, activation='relu',
              kernel_regularizer=l2(0.01))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Residual block with gating
    for _ in range(2):  # Add two residual blocks
        residual = x
        x = Dense(dense_units, activation='relu',
                  kernel_regularizer=l2(0.01))(x)
        x = LayerNormalization()(x)
        if use_feature_gating:
            gate = Dense(dense_units, activation='sigmoid')(inputs)
            x = Multiply()([x, gate])
        x = Add()([x, residual])
        x = Dropout(dropout_rate)(x)

    # Final dense layers
    x = Dense(dense_units // 2, activation='elu',
              kernel_regularizer=l2(0.01))(x)
    x = LayerNormalization()(x)

    # Concatenate wide and deep outputs
    combined = Concatenate()([wide, x])
    outputs = Dense(num_classes, activation='softmax')(combined)

    # Model compilation
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(
        label_smoothing=0.1), metrics=['accuracy'])
    return model


def optimize_hyperparameters(X_train, y_train_cat, X_test, y_test_cat, input_dim, num_classes):
    """Optimize model hyperparameters using Hyperopt."""

    def objective(params):
        model = build_advanced_model(
            input_dim=input_dim,
            num_classes=num_classes,
            dense_units=int(params['dense_units']),
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate'],
            optimizer_name=params['optimizer']
        )

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        model.fit(X_train, y_train_cat,
                  validation_data=(X_test, y_test_cat),
                  epochs=50,
                  batch_size=32,
                  callbacks=[early_stopping, lr_scheduler],
                  verbose=0)

        loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
        return {'loss': -accuracy, 'status': STATUS_OK}

    # Hyperparameter search space
    space = {
        'dense_units': hp.quniform('dense_units', 64, 256, 1),
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
        'learning_rate': hp.loguniform('learning_rate', -4, -2),
        'optimizer': hp.choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    }

    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )

    return {
        'dense_units': int(best_params['dense_units']),
        'dropout_rate': best_params['dropout_rate'],
        'learning_rate': best_params['learning_rate'],
        'optimizer': ['adam', 'rmsprop', 'sgd'][best_params['optimizer']]
    }


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


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

    # Train model
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping, lr_scheduler],
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
    plot_feature_target_correlation(X, y)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    return model


def plot_feature_target_correlation(X, y):
    # Create a DataFrame with features and target
    X_with_target = X.copy()
    X_with_target['Target'] = y

    # Calculate correlation
    correlation_matrix = X_with_target.corr()['Target'].drop('Target')

    # Plot correlation with target
    plt.figure(figsize=(10, 8))
    correlation_matrix.sort_values().plot(kind='barh')
    plt.title("Feature Correlation with Target")
    plt.xlabel("Correlation")
    plt.ylabel("Features")
    plt.show()


def predict_dataset_type(file_path):
    model = tf.keras.models.load_model("csv_classifier_model.h5")
    scaler = joblib.load("csv_scaler.pkl")
    le = joblib.load("label_encoder.pkl")

    features = extract_features(file_path)
    features_df = pd.DataFrame([features])
    features_df.fillna(0, inplace=True)
    features_scaled = scaler.transform(features_df)

    prediction = model.predict(features_scaled)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]


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
    predicted_class = predict_dataset_type(test_file)
    print(f"Predicted dataset type: {predicted_class}")
