import tensorflow as tf
import pandas as pd
import numpy as np
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from simple_model import (
    MODEL_PATH,
    CONTENT_DIRECTORY,
    load_data_from_directory,
    extract_features,
    reshape_dataframe_tuples,
    normalize_features,
)
from utils import linebreak
from sklearn.model_selection import train_test_split


def plot_actual_vs_predicted(actual_values, predicted_values):
    # Find indices where either list has a value of 1

    actual_indices = [i for i, val in enumerate(actual_values) if val == 1]
    predicted_indices = get_results(predicted_values)
    print(predicted_indices)

    # Plot the actual and predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(
        actual_indices,
        [1] * len(actual_indices),
        color="blue",
        label="Actual",
        marker="o",
    )
    plt.scatter(
        predicted_indices,
        [1] * len(predicted_indices),
        color="red",
        label="Predicted",
        marker="x",
    )

    # Add labels with index values
    # for i, idx in enumerate(actual_indices):
    #     plt.annotate(
    #         str(idx), (idx, 1), textcoords="offset points", xytext=(0, 5), ha="center"
    #     )
    # for i, idx in enumerate(predicted_indices):
    #     plt.annotate(
    #         str(idx), (idx, 1), textcoords="offset points", xytext=(0, 5), ha="center"
    #     )

    plt.title("Actual vs. Predicted Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.yticks([])  # Hides y-axis ticks
    plt.show()


def unpack_result(data):
    result = []
    for sublist in data.tolist():
        for subsublist in sublist:
            for value in subsublist:
                result.append(value)

    return result


def get_results(pred):
    return sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:6]


def evaluate_model(model, X_test, y_test):
    y_pred_classes = []
    for x, y in zip(X_test, y_test):
        reshaped_x = np.reshape(np.asarray(x).astype(np.float32), (-1, 55340, 4))
        reshaped_y = np.reshape(np.asarray(y).astype(np.float32), (-1, 55340, 1))

        y_pred = model.predict(reshaped_x)
        unpacked_actual = unpack_result(reshaped_y)
        unpacked_pred = unpack_result(y_pred)
        plot_actual_vs_predicted(unpacked_actual, unpacked_pred)
        exit()
        y_pred_classes.append((reshaped_y, y_pred))


if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)
    data_frames = load_data_from_directory(CONTENT_DIRECTORY)
    features = extract_features(data_frames)
    features_norm = normalize_features(features)
    X_standard, y_standard = reshape_dataframe_tuples(features_norm)
    X_train, X_test, y_train, y_test = train_test_split(
        X_standard, y_standard, test_size=0.2, random_state=42
    )
    X_standard, y_standard = reshape_dataframe_tuples(features_norm)
    evaluate_model(model, X_test, y_test)
