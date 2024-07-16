import math
import pandas as pd
from keras import models, layers, optimizers, regularizers
import numpy as np
import random
from sklearn import model_selection, preprocessing
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from QDataPipline import QDataPipeline
import os
from tensorflow import keras
from keras_tuner import HyperModel
from keras_tuner import RandomSearch

from tensorflow.python.client import device_lib


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


print(get_available_devices())

hidden_units = 150  # how many neurons in the hidden layer
activation = "relu"  # activation function for hidden layer
l2 = 0.01  # regularization - how much we penalize large parameter values
learning_rate = 0.01  # how big our steps are in gradient descent
epochs = 5  # how many epochs to train for
batch_size = 128  # how many samples to use for each gradient descent update

CONTENT_PATH = "content/training_data_with_points"
FEATURES = [
    "Relative_time",
    "Resonance_Frequency",
    "Dissipation",
    "Difference",
    "Cumulative",
    "Dissipation_super",
    "Difference_super",
    "Cumulative_super",
    "Resonance_Frequency_super",
    "Dissipation_gradient",
    "Difference_gradient",
    "Resonance_Frequency_gradient",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Resonance_Frequency_detrend",
    "Difference_detrend",
]
TARGETS = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6"]


class MyHyperModel(HyperModel):
    def build(self, hp):
        model = models.Sequential()

        # Tune the number of layers
        for i in range(hp.Int("num_layers", 1, 4)):
            model.add(
                keras.layers.Dense(
                    input_dim=len(FEATURES),
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                    activation="relu",
                )
            )

        # add the output layer
        model.add(layers.Dense(input_dim=len(FEATURES), units=1, activation="sigmoid"))

        # Tune the learning rate for the optimizer
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss="binary_crossentropy",
            metrics=[
                tfa.metrics.F1Score(num_classes=1, threshold=0.5),
            ],
        )
        return model


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def train_and_evaluate(model, x_train, y_train, x_test, y_test, n=20):
    train_accs = []
    test_accs = []
    with tqdm(total=n) as progress_bar:
        for _ in range(n):
            model.fit(
                x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=False
            )
            train_accs.append(
                model.evaluate(x_train, y_train, batch_size=32, verbose=False)[1]
            )
            test_accs.append(
                model.evaluate(x_test, y_test, batch_size=32, verbose=False)[1]
            )
            progress_bar.update()
    print("Avgerage Training Accuracy: %s" % np.average(train_accs))
    print("Avgerage Testing Accuracy: %s" % np.average(test_accs))
    return train_accs, test_accs


def evaluate_on_dir(test_dir, model, target):
    content = []

    for root, dirs, files in os.walk(test_dir):
        for file in files:
            content.append(os.path.join(root, file))
    for filename in tqdm(content, desc="<<Processing Files>>"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            if max(pd.read_csv(data_file)["Relative_time"].values) < 90:
                poi_file = filename.replace(".csv", "_poi.csv")
                actual_indices = pd.read_csv(poi_file, header=None).values
                qdp = QDataPipeline(data_file)
                qdp.preprocess(poi_file=None)
                qdp.__dataframe__ = qdp.__dataframe__.drop(columns=["Class", "Pooling"])
                has_nan = qdp.__dataframe__.isna().any().any()
                if not has_nan:
                    predictions = []
                    predictions = model.predict(qdp.__dataframe__)
                    print(predictions)
                    plt.figure()
                    plt.plot(
                        normalize(qdp.__dataframe__["Dissipation"]), label="Dissipation"
                    )
                    plt.plot(normalize(predictions), label=target)
                    plt.axvline(
                        x=actual_indices[0],
                        color="dodgerblue",
                        linestyle="--",
                        label="Actual",
                    )
                    for index in actual_indices:
                        plt.axvline(
                            x=index,
                            color="dodgerblue",
                            linestyle="--",
                        )
                    plt.legend()
                    plt.show()


def load_and_split_data(data_dir):
    count = 0
    data_df = pd.DataFrame()
    content = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            content.append(os.path.join(root, file))
    for filename in tqdm(content, desc="<<Processing Files>>"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            # if count > 100:
            #     break
            data_file = filename
            if max(pd.read_csv(data_file)["Relative_time"].values) < 90:
                poi_file = filename.replace(".csv", "_poi.csv")
                qdp = QDataPipeline(data_file)
                qdp.preprocess(poi_file=poi_file)

                has_nan = qdp.__dataframe__.isna().any().any()
                if not has_nan:
                    count += 1
                    data_df = pd.concat([data_df, qdp.get_dataframe()])
    y = data_df[TARGETS[4]]
    X = data_df.drop(columns=TARGETS)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return ((train_X, train_y), (test_X, test_y))


# grab the MNIST dataset (if this is your first time using this
# dataset then the 11MB download may take a minute)
print("[INFO] accessing data directory...")

((train_X, train_y), (test_X, test_y)) = load_and_split_data(CONTENT_PATH)

# tuner = RandomSearch(
#     MyHyperModel(),
#     objective="val_accuracy",
#     max_trials=5,  # Number of hyperparameter combinations to try
#     executions_per_trial=3,  # Number of models to be built and evaluated for each trial
#     directory="QModel/SavedModels",
#     project_name="pt5tuning",
# )

# tuner.search(train_X, train_y, epochs=10, validation_data=(test_X, test_y))
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# model = tuner.hypermodel.build(best_hps)
# model.summary()
# train the parameters

model = models.Sequential()

model.add(layers.Dense(input_dim=len(FEATURES), units=100, activation="relu"))
model.add(layers.Dense(input_dim=len(FEATURES), units=50, activation="relu"))
# add the output layer
model.add(layers.Dense(input_dim=len(FEATURES), units=1, activation="sigmoid"))
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        tfa.metrics.F1Score(num_classes=1, threshold=0.5),
    ],
)
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size)
evaluate_on_dir("content/VOYAGER_PROD_DATA", model, "Multi-Class")
# evaluate accuracy
train_acc = model.evaluate(train_X, train_y, batch_size=32)[1]
test_acc = model.evaluate(test_X, test_y, batch_size=32)[1]
print("Training accuracy: %s" % train_acc)
print("Testing accuracy: %s" % test_acc)

losses = history.history["loss"]
plt.plot(range(len(losses)), losses, "r")
plt.show()

# _, test_accs = train_and_evaluate(model, train_X, train_y, test_X, test_y)
# plt.hist(test_accs)
# plt.show()
