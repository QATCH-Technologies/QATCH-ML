import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    Input,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os

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


def plot_history(history, cluster_id):
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Training and Validation Loss for Cluster {cluster_id}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


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





def create_model(input_shape, num_filters=64, kernel_size=3, dropout_rate=0.2):
    inputs = Input(shape=input_shape)

    # Optional: Embedding Layer if needed
    # embedded = Embedding(input_dim, output_dim)(inputs)

    # Temporal Convolutional Neural Network (TCN)
    conv1 = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="causal",
    )(inputs)
    conv2 = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="causal",
    )(conv1)
    conv3 = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="causal",
    )(conv2)
    conv4 = Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="causal",
    )(conv3)
    pooling = GlobalMaxPooling1D()(conv4)
    dropout = Dropout(rate=dropout_rate)(pooling)

    # Dense Layers
    dense1 = Dense(64, activation="relu")(dropout)
    dense2 = Dense(32, activation="relu")(dense1)

    # Output Layer
    output = Dense(1)(dense2)  # Output layer for regression

    model = Model(inputs=inputs, outputs=output)
    return model


if __name__ == "__main__":
    # Initialize dictionaries to hold training, validation, and test data for each cluster
    X_train_sets = {}
    X_val_sets = {}
    X_test_sets = {}
    y_train_sets = {}
    y_val_sets = {}
    y_test_sets = {}
    region_data = region_gen_v2(data, pois)
    # Loop through each cluster to split data
    for cluster_id, records in region_data.items():
        # If there's only one record, we can't split; handle this by putting all into training
        if len(records) < 2:
            print(
                f"Cluster {cluster_id} has less than 2 records, not enough to split into train and test."
            )
            X_train_sets[cluster_id] = np.array([record["DATA"] for record in records])
            y_train_sets[cluster_id] = np.array([record["POI"] for record in records])
            X_val_sets[cluster_id] = np.array([])
            y_val_sets[cluster_id] = np.array([])
            X_test_sets[cluster_id] = np.array([])
            y_test_sets[cluster_id] = np.array([])
            continue

        # Extract the data and labels
        X = np.array([record["DATA"] for record in records])
        y = np.array([record["POI"] for record in records])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Further split the training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42
        )  # 0.25*0.8 = 0.2 of original data

        # Store the split data in the dictionaries
        X_train_sets[cluster_id] = X_train
        X_val_sets[cluster_id] = X_val
        X_test_sets[cluster_id] = X_test
        y_train_sets[cluster_id] = y_train
        y_val_sets[cluster_id] = y_val
        y_test_sets[cluster_id] = y_test

        # Dictionary to hold the models and their training histories
    models = {}
    histories = {}

    # Train a model for each cluster
    for cluster_id in region_data.keys():
        if (
            len(X_train_sets[cluster_id]) < 2
        ):  # Skip training if there's not enough data
            continue

        print(f"\nTraining model for Cluster {cluster_id}")

        # Get training, validation, and test data
        X_train = np.array(X_train_sets[cluster_id])
        y_train = np.array(y_train_sets[cluster_id])
        X_val = np.array(X_val_sets[cluster_id])
        y_val = np.array(y_val_sets[cluster_id])
        X_test = np.array(X_test_sets[cluster_id])
        y_test = np.array(y_test_sets[cluster_id])

        # Create a CNN model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_model(input_shape=input_shape)

        # Train the model and store the history
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=4,
            verbose=0,
        )

        # Evaluate the model
        if X_test.size > 0:  # If test data is available
            loss, mae = model.evaluate(X_test, y_test, verbose=0)
            print(
                f"Model for Cluster {cluster_id} - Test Loss: {loss:.4f}, Test MAE: {mae:.4f}"
            )

        # Store the trained model and its history
        models[cluster_id] = model
        histories[cluster_id] = history

    for cluster_id, history in histories.items():
        print(f"\nTraining and Validation History for Cluster {cluster_id}")
        plot_history(history, cluster_id)
