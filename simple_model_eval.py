import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema, find_peaks
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


def find_local_maxima_with_indices(points):
    x = np.array(points)
    # for local maxima
    peaks, _ = find_peaks(x)

    return np.asarray(peaks)


def plot_actual_vs_predicted(actual_values, predicted_values):
    # Find indices where either list has a value of 1
    # Find the last occurrence of 1 in the binary data
    last_1_index = len(actual_values) - 1 - actual_values[::-1].index(1)

    # Truncate both datasets
    actual_values = actual_values[: last_1_index + 1]

    actual_indices = [i for i, val in enumerate(actual_values) if val == 1]

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
        [predicted_values],
        [1] * len(predicted_values),
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


def plot_pred_values(pred_values, actual_values):
    print(pred_values)
    linebreak()
    print(actual_values)
    # Find the last occurrence of 1 in the binary data
    last_1_index = len(actual_values) - 1 - actual_values[::-1].index(1)

    # Truncate both datasets
    values = pred_values[: last_1_index + 1]
    binary_data = actual_values[: last_1_index + 1]

    # Get the indices where binary_data is 1
    marked_indices = [i for i, val in enumerate(binary_data) if val == 1]

    # Plotting
    plt.plot(values)
    plt.scatter(
        marked_indices,
        [values[i] for i in marked_indices],
        color="red",
        label="Actual Indices",
    )
    plt.title("Prediction Distribution")
    plt.xlabel("Index")
    plt.ylabel("Percentage")
    plt.legend()
    plt.show()


def unpack_result(data):
    result = []
    for sublist in data.tolist():
        for subsublist in sublist:
            for value in subsublist:
                result.append(value)

    return result


def unpack_pred(data):
    result = []
    for sublist in data:
        for value in sublist:
            result.append(value)
    return np.array(result)


def get_peaks(data):
    max_peaks = 6
    min_distance = 0  # Minimum distance between peaks
    prominence_threshold = 0.6
    peaks, properties = find_peaks(data, prominence=(None, prominence_threshold))

    # Sort peaks by prominence
    sorted_peaks = sorted(
        peaks,
        key=lambda x: properties["prominences"][list(peaks).index(x)],
        reverse=True,
    )

    # Filter out peaks that are too close to each other
    significant_peaks = []
    last_peak_index = None
    while len(significant_peaks) < max_peaks:
        for peak_index in sorted_peaks:
            if last_peak_index is None or peak_index - last_peak_index >= min_distance:
                significant_peaks.append(peak_index)
                last_peak_index = peak_index

    return significant_peaks


def get_idxs(peaks, raw):
    indices = {
        value: [index for index, element in enumerate(raw) if element == value]
        for value in peaks
    }
    return indices


def plot_on_curve():
    # Replace 'file_path.csv' with the path to your CSV file
    file_path = "file_path.csv"

    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv("content/training_data_with_points/BSA200MGML_2_3rd.csv")

    # Extract the 'Dissipation' column into a list
    raw = np.array(data["Dissipation"])


def evaluate_model(model, X_test, y_test):
    y_pred_classes = []
    for x, y in zip(X_test, y_test):
        reshaped_x = np.reshape(np.asarray(x).astype(np.float32), (-1, 55340, 1))
        reshaped_y = np.reshape(np.asarray(y).astype(np.float32), (-1, 55340, 1))

        y_pred = model.predict(reshaped_x)[0]
        print(y_pred)
        unpacked_actual = unpack_result(reshaped_y)
        unpacked_predicted = unpack_pred(y_pred)
        plot_pred_values(unpacked_predicted, unpacked_actual)
        peaks = get_peaks(unpacked_predicted)
        for p in peaks:
            print(p)

        plot_actual_vs_predicted(unpacked_actual, peaks)
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
