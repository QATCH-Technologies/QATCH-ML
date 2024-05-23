import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import os


# Step 1: Data Preprocessing
def read_data(raw_data_file, poi_file):
    raw_data = pd.read_csv(raw_data_file)
    poi = pd.read_csv(poi_file)
    return raw_data, poi


# Step 1: Data Preprocessing
def load_data_from_directory(data_directory):
    all_raw_data = []
    all_poi_data = []
    for filename in os.listdir(data_directory):
        if filename.endswith("_poi.csv"):
            poi_file = os.path.join(data_directory, filename)
            raw_data_file = os.path.join(
                data_directory, filename.replace("_poi.csv", ".csv")
            )
            if os.path.isfile(raw_data_file):
                raw_data, poi = read_data(raw_data_file, poi_file)
                all_raw_data.append(raw_data)
                all_poi_data.append(poi)
    return all_raw_data, all_poi_data


def extract_features(raw_data):
    features = raw_data[
        ["Dissipation", "Relative_time", "Resonance_Frequency", "Peak Magnitude (RAW)"]
    ]
    return features


def normalize_features(features):
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features


# Step 2: Data Augmentation (if necessary)


# Step 3: Model Design
def build_model(input_shape):
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=input_shape),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Step 4: Training
def train_model(X_train, y_train, X_val, y_val):
    model = build_model(input_shape=X_train.shape[1:])
    history = model.fit(
        X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val)
    )
    return model, history


# Step 5: Evaluation
def evaluate_model(model, X_test, y_test):
    # Evaluate model on test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.round(y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    plt.title("Confusion Matrix")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes))


count = 0
# Example usage
data_directory = "content/training_data_with_points"
all_raw_data, all_poi_data = load_data_from_directory(data_directory)

# Process each dataset individually
for raw_data, poi in zip(all_raw_data, all_poi_data):
    if count > 10:
        break
    features = extract_features(raw_data)
    X = normalize_features(features)
    y = np.zeros(len(raw_data))  # Assuming no blips by default
    y[poi.values.flatten()] = 1

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model, history = train_model(X_train, y_train, X_val, y_val)
    count += 1
# Evaluate model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# evaluate_model(model, X_test, y_test)
# Predict blips in new datasets
# Predict probabilities
predicted_probabilities = model.predict(X_test)

# If you're doing binary classification and want to get predicted classes (0 or 1)
predicted_classes = (predicted_probabilities > 0.5).astype(int)
k = 0
for i in predicted_classes:
    print(i)
    if i == [1]:
        print(f"{X_test[k]}, idx {k}")

    k += 1

# Assuming your test data has timestamps associated with it
# You can use these timestamps to plot the actual predicted points
# timestamps = X_test["Timestamp"].values  # Extract timestamps from your test data

# Plotting the actual predicted points
plt.figure(figsize=(10, 6))
# plt.plot(
#     timestamps, predicted_probabilities, label="Predicted Probabilities", marker="o"
# )
plt.xlabel("Timestamps")
plt.ylabel("Predicted Probabilities")
plt.title("Predicted Probabilities Over Time")
plt.legend()
plt.grid(True)
plt.show()
