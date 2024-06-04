import msvcrt
import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from scipy.signal import savgol_filter
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from sklearn.neighbors import LocalOutlierFactor
from utils import status, error, info, linebreak, loading, echo, plot_loss
import matplotlib.pyplot as plt

""" Ensures pandas displays all rows of a dataframe on print(). """
pd.set_option("display.max_rows", None)

""" Path to the saved model. """
MODEL_PATH = "SavedModel/QATCH-simple-v1"

""" Path to the training data directory. """
CONTENT_DIRECTORY = "content/training_data_with_points"

""" Set of features to not include in training. """
DROPPED_FEATURES = ["Date", "Time", "Ambient", "Temperature"]

""" Set of features to include in training. """
TRAINING_FEATURES = [
    "Dissipation",
    "Relative_time",
    "Peak Magnitude (RAW)",
    "Resonance_Frequency",
]

INPUT_SHAPE = (100, 4)

TRIM_OFFSET = 50

POI_FILE = "content/training_data_with_points/W0802_F5_DI6s_good_3rd_poi.csv"
DATA_FILE = "content/training_data_with_points/W0802_F5_DI6s_good_3rd.csv"


def find_fill_start_end(data):
    # Data preprocessing
    smoothed_data = savgol_filter(
        data["Dissipation"].values, window_length=11, polyorder=3
    )

    # Calculate the derivative
    gradient = np.gradient(smoothed_data)

    # Smooth the derivative if necessary
    smoothed_gradient = savgol_filter(gradient, window_length=11, polyorder=3)
    # Identify change points
    change_points = []
    threshold = 0.0000001  # Adjust as needed
    while len(change_points) == 0:
        for i in range(1, len(smoothed_gradient)):
            if (
                smoothed_gradient[i] > threshold
                and smoothed_gradient[i] > smoothed_gradient[i - 1]
            ):
                change_points.append(i)
        threshold = threshold / 2

    return change_points


def region_gen(data):
    start_time = find_fill_start_end(data)[0]
    end_time = find_fill_start_end(data)[-1]
    dissipation_raw = dissipation_trim = data["Dissipation"]
    dissipation_trim = dissipation_raw.values[start_time - TRIM_OFFSET : end_time]
    time = np.arange(len(dissipation_trim))
    lof_data = dissipation_trim.reshape(-1, 1)
    contamination = 0.05
    clusters = {}
    while len(clusters) < 6 and contamination < 0.5:
        lof = LocalOutlierFactor(contamination=contamination)
        anomalies = lof.fit_predict(lof_data)

        idxs = []
        for pt in time:
            if anomalies[pt] == -1:
                idxs.append(pt)
            else:
                idxs.append(0)

        data = np.asarray(idxs)
        data = np.reshape(data, (-1, 1))
        kmeans = KMeans(n_clusters=7)
        kmeans.fit(data)
        for i in range(len(kmeans.labels_)):
            if kmeans.labels_[i] > 0:
                if clusters.get(kmeans.labels_[i]) is None:
                    clusters[kmeans.labels_[i]] = [i]
                else:
                    clusters[kmeans.labels_[i]].append(i)
        contamination = contamination + 0.01
    regional_data = {}
    for k, v in clusters.items():
        cluster_start = v[0] + start_time
        if cluster_start + 50 > time[-1]:
            cluster_start = cluster_start - 25
        cluster_end = cluster_start + 50  # Make this a constant
        raw_cluster = dissipation_raw[cluster_start:cluster_end].values
        cluster_time = np.arange(cluster_start, cluster_end)
        regional_data[k - 1] = (raw_cluster, cluster_time)

    return regional_data


def region_gen_v2(data, poi):
    regions = {}
    data = data.drop(columns=DROPPED_FEATURES)
    id = 0
    for pt in poi.values:
        r = []
        l_range, r_range = 0, 0
        if pt[0] - TRIM_OFFSET < 0:
            l_range = 0
        else:
            l_range = pt[0] - TRIM_OFFSET
        if pt[0] + TRIM_OFFSET > len(data):
            r_range = len(data) - 1
        else:
            r_range = pt[0] + TRIM_OFFSET
        for i in range(l_range, r_range):
            r.append(data.iloc[i])

        regions[id] = r
        id += 1
    return regions


def read_data(raw_data_file, poi_file):
    raw_data = pd.read_csv(raw_data_file)
    poi_data = pd.read_csv(poi_file, header=None)
    return {"RAW": raw_data, "POI": poi_data}


def load_data_from_directory(data_directory):
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


def build_model():
    model = Sequential()
    model.add(
        Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(100, 4))
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="linear"))  # For regression
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


def convert(data):
    # Convert DataFrame to NumPy array
    ret = []
    for df in data:
        data_array = df.to_numpy()

        # Reshape data to the required shape (n_samples, 1, 4)
        data_array = data_array.reshape((len(data_array), 1, data_array.shape[1]))
        ret.append(data_array)
    return ret


def train_model(X_train, y_train, X_val, y_val):
    model = build_model()
    # Need to convert training and test data into useable format for tensorflow
    X_train = convert(X_train)
    X_val = convert(X_val)
    history = model.fit(
        X_train, y_train, epochs=32, batch_size=16, validation_data=(X_val, y_val)
    )
    model.summary()
    return model, history


if __name__ == "__main__":
    content = load_data_from_directory(CONTENT_DIRECTORY)
    merge = []
    r1, r2, r3, r4, r5, r6 = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for data_set in content:
        df = pd.DataFrame()
        regions = region_gen_v2(data_set["RAW"], data_set["POI"])
        for cluster in regions.keys():
            if cluster == 0:
                r1.append(
                    {"DATA": regions[cluster], "POI": data_set["POI"].values[cluster]}
                )
            elif cluster == 1:
                r2.append(
                    {"DATA": regions[cluster], "POI": data_set["POI"].values[cluster]}
                )
            elif cluster == 2:
                r3.append(
                    {"DATA": regions[cluster], "POI": data_set["POI"].values[cluster]}
                )
            elif cluster == 3:
                r4.append(
                    {"DATA": regions[cluster], "POI": data_set["POI"].values[cluster]}
                )
            elif cluster == 4:
                r5.append(
                    {"DATA": regions[cluster], "POI": data_set["POI"].values[cluster]}
                )
            else:
                r6.append(
                    {"DATA": regions[cluster], "POI": data_set["POI"].values[cluster]}
                )
    merge.append(r1)
    merge.append(r2)
    merge.append(r3)
    merge.append(r4)
    merge.append(r5)
    merge.append(r6)
    poi_models = []
    for region in merge:
        X = [item["DATA"] for item in region]
        y = [item["POI"] for item in region]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model, history = train_model(X_train, y_train, X_test, y_test)
        plot_loss(history)
