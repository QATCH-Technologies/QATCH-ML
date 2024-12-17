import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

# Step 1: Feature Extraction Function


def extract_features(file_path):
    df = pd.read_csv(file_path)
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

    # Additional features to describe entire dataset
    if not df.empty:
        features['median'] = df.median(numeric_only=True).mean()
        features['skew'] = df.skew(numeric_only=True).mean()
        features['kurtosis'] = df.kurtosis(numeric_only=True).mean()
        features['range'] = df.max(
            numeric_only=True).max() - df.min(numeric_only=True).min()
        features['total_sum'] = df.sum(numeric_only=True).sum()
        features['total_variance'] = df.var(numeric_only=True).sum()

        # Distribution and correlation features
        features['unique_values_ratio'] = df.nunique().mean() / \
            features['num_rows']
        features['correlation_mean'] = df.corr(
            numeric_only=True).abs().mean().mean()

        # End-focused statistics
        features['end_mean'] = df.tail(10).mean(numeric_only=True).mean()
        features['end_std'] = df.tail(10).std(numeric_only=True).mean()
        features['end_min'] = df.tail(10).min(numeric_only=True).min()
        features['end_max'] = df.tail(10).max(numeric_only=True).max()
        features['mean_diff'] = df.tail(10).mean(
            numeric_only=True).mean() - df.head(10).mean(numeric_only=True).mean()
        features['std_diff'] = df.tail(10).std(
            numeric_only=True).mean() - df.head(10).std(numeric_only=True).mean()
    else:
        features['median'] = 0
        features['skew'] = 0
        features['kurtosis'] = 0
        features['range'] = 0
        features['total_sum'] = 0
        features['total_variance'] = 0
        features['unique_values_ratio'] = 0
        features['correlation_mean'] = 0
        features['end_mean'] = 0
        features['end_std'] = 0
        features['end_min'] = 0
        features['end_max'] = 0
        features['mean_diff'] = 0
        features['std_diff'] = 0

    return features

# Step 2: Load Datasets and Create Training Data


def load_and_prepare_data(dataset_paths):
    X = []
    y = []

    for label, path in dataset_paths.items():
        for root, _, files in tqdm(os.walk(path), desc=f"Loading {label}"):
            files = [f for f in files if f.endswith(
                ".csv") and not f.endswith('_poi.csv')]
            for file in files:
                file_path = os.path.join(root, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(label)

    # Convert to DataFrame
    X_df = pd.DataFrame(X)
    X_df.fillna(0, inplace=True)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, "label_encoder.pkl")

    return X_df, y_encoded

# Step 3: Hyperparameter Tuning with Hyperopt


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    def objective(params):
        model = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        auc = roc_auc_score(y_test, model.predict_proba(
            X_test), multi_class='ovr')
        return {'loss': -auc, 'status': STATUS_OK}

    # Define the search space
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 500, 10),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    }

    trials = Trials()
    print("Starting hyperparameter optimization...")
    best_params = fmin(fn=objective, space=space,
                       algo=tpe.suggest, max_evals=50, trials=trials)

    print("Best parameters:", best_params)
    best_model = XGBClassifier(
        n_estimators=int(best_params['n_estimators']),
        max_depth=int(best_params['max_depth']),
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        random_state=42
    )
    best_model.fit(X_train, y_train)

    # Evaluate
    y_pred = best_model.predict(X_test)
    correct = (y_pred == y_test).sum()
    incorrect = (y_pred != y_test).sum()
    print(
        f"Correct predictions: {correct}, Incorrect predictions: {incorrect}")
    print("ROC AUC Score:", roc_auc_score(
        y_test, best_model.predict_proba(X_test), multi_class='ovr'))
    print(classification_report(y_test, y_pred))

    # Save model and scaler
    joblib.dump(best_model, "csv_classifier_model.pkl")
    joblib.dump(scaler, "csv_scaler.pkl")
    print("Model saved!")

    # Feature importance visualization
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(12, 8))
    plt.barh(feature_names, feature_importances)
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance in XGBoost Model")
    plt.show()

# Step 4: Predict Dataset Type


def predict_dataset_type(file_path):
    model = joblib.load("csv_classifier_model.pkl")
    scaler = joblib.load("csv_scaler.pkl")
    le = joblib.load("label_encoder.pkl")

    features = extract_features(file_path)
    features_df = pd.DataFrame([features])
    features_df.fillna(0, inplace=True)
    features_scaled = scaler.transform(features_df)

    prediction = model.predict(features_scaled)
    predicted_label = le.inverse_transform(prediction)
    return predicted_label[0]


if __name__ == "__main__":
    # Define dataset paths
    dataset_paths = {
        "full_fill": "content/dropbox_dump",
        "no_fill": "content/no_fill",
        "channel_1_partial": "content/channel_1",
        "channel_2_partial": "content/channel_2",
    }

    # Load and prepare data
    print("Loading and preparing data...")
    X, y = load_and_prepare_data(dataset_paths)

    # Train model
    print("Training model...")
    train_model(X, y)

    # Example prediction
    test_file = r"C:\Users\QATCH\dev\QATCH-ML\content\channel_2\00096\DD230321_2B_49.5CP_1_3rd.csv"
    predicted_class = predict_dataset_type(test_file)
    print(f"Predicted dataset type: {predicted_class}")
