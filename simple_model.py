import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from utils import status, error, info, linebreak, loading, echo, plot_loss

""" Ensures pandas displays all rows of a dataframe on print(). """
pd.set_option("display.max_rows", None)

""" Path to the saved model. """
MODEL_PATH = "SavedModel/QATCH-simple-v1"

""" Path to the training data directory. """
CONTENT_DIRECTORY = "content/training_data_with_points"

""" Set of features to not include in training. """
DROPPED_FEATURES = ["Date", "Time", "Ambient", "Temperature", "POIs"]

""" Set of features to include in training. """
TRAINING_FEATURES = [
    "Relative_time",
    "Peak Magnitude (RAW)",
    "Resonance_frequency",
    "Dissipation",
]

""" Target feature to train. """
TARGET_FEATURES = ["POIs"]


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
    return raw_data


def load_data_from_directory(data_directory):
    """Opens the training data directory.  This function controls execution flow of read_data().

    Args:
        data_directory (str): Path to the directory containing training data.

    Returns:
        list: Returns a list of combined DataFrames
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
    """This function is used for feature extraction.  The function extracts the target 'POIs'
    data from each DataFrame in the input (y) and drops the untrained features from each saving
    saving the remaining data as X.  The function returns a list of correlated tuples (X, y).

    Args:
        data_frames (list): List of DataFrames.

    Returns:
        list: Returns a list of tuples containing training and target features.
    """
    corr_data = []
    for df in data_frames:
        y = df[TARGET_FEATURES]
        X = df.drop(columns=DROPPED_FEATURES)
        corr_data.append((X, y))

    return corr_data


def normalize_features(features):
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
    model = Sequential(
        [
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="mean_absolute_error", metrics=["accuracy"])
    return model


def train_model(X_train, y_train, X_val, y_val):
    model = build_model()
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)
    y_val = np.asarray(y_val).astype(np.float32)

    history = model.fit(
        X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val)
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
    # print(X_train)
    # input("Press any key to continue...")
    # print(y_train)
    # input("Press any key to continue...")
    # print(X_test)
    # input("Press any key to continue...")
    # print(y_test)
    # input("Press any key to continue...")

    model, history = train_model(X_train, y_train, X_test, y_test)
    status(f"{model} training complete with no errors.")
    plot_loss(history)
    model.save(MODEL_PATH)
