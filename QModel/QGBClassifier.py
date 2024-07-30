import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tqdm as tqdm
from sklearn.metrics import accuracy_score, classification_report


def extract_features(file_path):
    # Load CSV file
    df = pd.read_csv(file_path)

    # Example features:
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    mean_values = df.mean().values
    std_values = df.std().values
    max_values = df.max().values
    min_values = df.min().values

    # Combine features into a single array
    features = np.concatenate(
        [
            np.array([num_rows, num_cols]),
            mean_values,
            std_values,
            max_values,
            min_values,
        ]
    )
    return features


def load_content(data_dir, label):
    features_list = []
    labels_list = []
    for root, dirs, files in os.walk(data_dir):
        for file_path in tqdm(files, desc="<<Loading Data>>"):
            if (
                file_path.endswith(".csv")
                and not file_path.endswith("_poi.csv")
                and not file_path.endswith("_lower.csv")
            ):
                features = extract_features(file_path)
                features_list.append(features)
                labels_list.append(label)
    return np.array(features_list), np.array(labels_list)


# Paths to files and their labels
X_good, y_good = load_content("content/good_runs", label=0)
X_bad, y_bad = load_content("content/bad_runs", label=1)

# Combine and split data
X = np.vstack([X_good, X_bad])
y = np.hstack([y_good, y_bad])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost classifier
model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
model.save_model("xgboost_model_file.json")
