"""
QATCH Model Training Script

This script contains functions and procedures for training a convolutional neural network model
to predict points of interest (POIs) in time series data.

Author: Paul MacNichol
Date: 05-28-2024

Requirements:
- pandas
- numpy
- scikit-learn
- tensorflow
- utils (custom module)

Usage:
- Ensure the necessary packages are installed.
- Place raw data CSV files and their corresponding POI CSV files in the specified directory.
- Adjust parameters such as file paths, features, and model architecture as needed.
- Run the script to train the model and save it for future use.
"""

import msvcrt
import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from scipy.signal import savgol_filter

from tensorflow.keras.layers import Dense
from utils import status, error, info, linebreak, loading, echo, plot_loss

""" Ensures pandas displays all rows of a dataframe on print(). """
pd.set_option("display.max_rows", None)

""" Path to the saved model. """
MODEL_PATH = "SavedModel/QATCH-simple-v1"

""" Path to the training data directory. """
CONTENT_DIRECTORY = "content/training_data_with_points"

""" Set of features to not include in training. """
DROPPED_FEATURES = [
    "Date",
    "Time",
    "Ambient",
    "Temperature",
    "POIs",
    "Relative_time",
    "Peak Magnitude (RAW)",
    "Resonance_Frequency",
    "Dissipation",
]

""" Set of features to include in training. """
TRAINING_FEATURES = ["der2"]

""" Target feature to train. """
TARGET_FEATURES = ["POIs"]


def compute_second_derivative(file_path, window_size=30, polyorder=3):
    """
    Compute the second derivative of a data array and apply smoothing.

    Parameters:
        data (array-like): Input data array.
        window_size (int): Window size for Savitzky-Golay filter. Should be odd.
        polyorder (int): Polynomial order for Savitzky-Golay filter.

    Returns:
        array: Smoothed second derivative of the input data.
    """
    data = pd.read_csv(file_path)

    # Extract the 'Dissipation' column into a list
    data = np.array(data["Dissipation"])

    # Compute the second derivative using central finite differences
    data[0] = data[-1] = 0
    dx = data[1] - data[0]
    d2_data = np.gradient(np.gradient(data, dx), dx)

    # Apply Savitzky-Golay smoothing
    smoothed_d2_data = savgol_filter(d2_data, window_size, polyorder)

    return smoothed_d2_data


def read_data(raw_data_file, poi_file):
    """Reads a data and poi CSV file and returns them as a combined as a pandas dataframe.
    The returned data frame has a an additional column titled 'POIs' of binary values.
     - 1 indicates a POI occured at the current row index.
     - 0 indicates a POI did not occur at the current row index.

    Args:
        raw_data_file (str): path to the raw data CSV file.
        poi_file (str): path to the POI CSV file.

    Returns:
        DataFrame: The combined DataFrame containing binary indicators for POIs.
    """
    raw_data = pd.read_csv(raw_data_file)
    der2 = compute_second_derivative(raw_data_file)
    pois_flags, pois_indexes = [], []
    with open(poi_file, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            pois_indexes.append(row)

    pois_indexes_as_ints = [int(item) for sublist in pois_indexes for item in sublist]
    for i in range(len(raw_data)):
        if i in pois_indexes_as_ints:
            pois_flags.append(1)
        else:
            pois_flags.append(0)
    raw_data["POIs"] = pois_flags
    raw_data["der2"] = der2
    return raw_data


def load_data_from_directory(data_directory):
    """Load data from a directory containing Point of Interest (POI) files.

    Args:
        data_directory (str): Path to the directory containing POI files.

    Returns:
        list: A list of DataFrames containing combined data from POI and raw files.
    """
    dfs = []

    for filename in os.listdir(data_directory):
        if filename.endswith("_poi.csv"):
            poi_file = os.path.join(data_directory, filename)
            raw_data_file = os.path.join(
                data_directory, filename.replace("_poi.csv", ".csv")
            )
            if os.path.isfile(raw_data_file):
                combined_df = read_data(raw_data_file, poi_file)
                dfs.append(combined_df)
    return dfs


def extract_features(data_frames):
    """
    Extracts features from a list of data frames.

    Args:
        data_frames (list): A list containing pandas DataFrames, each representing a dataset.

    Returns:
        list: A list of tuples containing the feature matrix (X) and target vector (y) for each DataFrame.
    """
    corr_data = []
    for df in data_frames:
        y = df[TARGET_FEATURES]
        X = df.drop(columns=DROPPED_FEATURES)
        corr_data.append((X, y))

    return corr_data


def normalize_features(features):
    """Normalize features using StandardScaler.

    Args:
        features (list of tuples): A list containing tuples where each tuple contains
            the raw features (X_raw) and corresponding labels (y_raw).

    Returns:
        list of tuples: A list of tuples where each tuple contains the normalized
            features (X_norm) and corresponding labels (y_raw).
    """
    standard_scaler = StandardScaler()
    features_norm = []
    for item in features:
        X_raw = item[0]
        y_raw = item[1]
        normalized_data = standard_scaler.fit_transform(X_raw)
        X_norm = pd.DataFrame(normalized_data, columns=X_raw.columns)
        features_norm.append((X_norm, y_raw))

    return features_norm


def reshape_dataframe_tuples(tuple_list):
    """Reshape DataFrame tuples to have consistent dimensions.

    Args:
        tuple_list (list of tuples): A list containing tuples where each tuple
            contains a DataFrame for features (X_df) and another DataFrame for
            labels (y_df).

    Returns:
        tuple: A tuple containing two lists. The first list contains DataFrames
            with standardized dimensions for features, and the second list
            contains DataFrames with standardized dimensions for labels.
    """
    max_rows_X = max(df[0].shape[0] for df in tuple_list)
    max_cols_X = max(df[0].shape[1] for df in tuple_list)

    standardized_X = []
    for X_df, _ in tuple_list:
        expanded_X = pd.DataFrame(index=range(max_rows_X), columns=range(max_cols_X))
        expanded_X.iloc[:, :] = 0  # Fill with zeros
        expanded_X.iloc[: X_df.shape[0], : X_df.shape[1]] = X_df.values
        standardized_X.append(expanded_X)

    max_rows_y = max(df[1].shape[0] for df in tuple_list)
    standardized_y = []

    for _, y_df in tuple_list:
        existing_rows = len(y_df)
        rows_to_add = max_rows_y - existing_rows

        additional_rows = pd.DataFrame({"POIs": [0] * rows_to_add})

        expanded_y = pd.concat([y_df, additional_rows], ignore_index=True)
        standardized_y.append(expanded_y)

    return standardized_X, standardized_y


def build_model():
    input_shape = (55340, 1)
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=input_shape),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def train_model(X_train, y_train, X_val, y_val):
    """
    Trains a neural network model using the provided training data and evaluates it on the validation data.

    Args:
        X_train (numpy.ndarray): Input features for training.
        y_train (numpy.ndarray): Target labels for training.
        X_val (numpy.ndarray): Input features for validation.
        y_val (numpy.ndarray): Target labels for validation.

    Returns:
        tuple: A tuple containing the trained model and the training history.
               - The trained model (tensorflow.python.keras.engine.sequential.Sequential): The trained neural network model.
               - The training history (dict): A dictionary containing training/validation loss and metrics per epoch.
    """
    model = build_model()
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)
    y_val = np.asarray(y_val).astype(np.float32)
    history = model.fit(
        X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val)
    )
    return model, history


if __name__ == "__main__":
    status("Ready")

    data_frames = load_data_from_directory(CONTENT_DIRECTORY)

    status(
        f"{len(os.listdir(CONTENT_DIRECTORY))} CSV files imported from directory '{CONTENT_DIRECTORY}'"
    )
    features = extract_features(data_frames)

    status(f"Extracted feature vectors\t{TRAINING_FEATURES}")
    status(f"Dropped feature vectors\t{DROPPED_FEATURES}")
    status(f"Target feature vector\t\t{TARGET_FEATURES}")
    features_norm = normalize_features(features)
    status("Normalized feature and target vectors")

    X_standard, y_standard = reshape_dataframe_tuples(features_norm)
    status(f"Reshape X-->{len(X_standard[0])} and y-->{len(y_standard[0])} features")
    status("Training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_standard, y_standard, test_size=0.2, random_state=42
    )
    model, history = train_model(X_train, y_train, X_test, y_test)
    status(f"{model} training complete with no errors.")
    plot_loss(history)
    model.save(MODEL_PATH)
    info(f"Model saved to path '{MODEL_PATH}'")
    print("Press any key to exit...")
    msvcrt.getch()
