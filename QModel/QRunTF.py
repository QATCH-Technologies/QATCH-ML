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
from keras.callbacks import EarlyStopping
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from tensorflow.python.client import device_lib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from keras.layers.merge import concatenate
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.utils import to_categorical
from numpy import dstack, argmax
from keras.utils import plot_model


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


print(get_available_devices())

hidden_units = 150  # how many neurons in the hidden layer
activation = "relu"  # activation function for hidden layer
l2 = 0.01  # regularization - how much we penalize large parameter values
learning_rate = 0.01  # how big our steps are in gradient descent
epochs = 10  # how many epochs to train for
batch_size = 128  # how many samples to use for each gradient descent update

PATH = "content/training_data_with_points"

# pd.set_option("display.max_rows", None)
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
    "EMP",
]
S_TARGETS = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6"]
M_TARGET = "Class"


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


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
                    plt.figure()
                    plt.plot(
                        normalize(qdp.__dataframe__["Dissipation"]), label="Dissipation"
                    )

                    predictions = []
                    qdp.__dataframe__ = RobustScaler().fit_transform(qdp.__dataframe__)
                    predictions = model.predict(qdp.__dataframe__)

                    plt.plot(
                        normalize(predictions), linestyle="--", label="Predictions"
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


def tsne_view(X, y):
    # Perform t-SNE to reduce to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="viridis", edgecolor="k"
    )
    plt.title("t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(scatter, label="Target Value")
    plt.show()


def pca_view(X, y):
    print("[INFO] building PCA")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolor="k")
    plt.title("PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Target Value")
    plt.show()


def load_content(data_dir, size):
    content = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            content.append(os.path.join(root, file))
            if len(content) >= size:
                break
    train_content, test_content = train_test_split(
        content, test_size=0.2, random_state=42, shuffle=True
    )
    return train_content, test_content


def resample_df(data, target, droppable):
    print(f"[INFO] resampling df {target}")
    y = data[target].values
    X = data.drop(columns=droppable)

    over = SMOTE()
    under = RandomUnderSampler()
    steps = [("o", over), ("u", under)]
    pipeline = Pipeline(steps=steps)

    X, y = pipeline.fit_resample(X, y)

    resampled_df = pd.DataFrame(X, columns=data.drop(columns=droppable).columns)
    resampled_df[target] = y

    return resampled_df


def build_dataset(content, multi_class=False):
    data_df = pd.DataFrame()
    for filename in tqdm(content, desc="<<Processing Dataset>>"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            if max(pd.read_csv(data_file)["Relative_time"].values) < 90:
                poi_file = filename.replace(".csv", "_poi.csv")
                qdp = QDataPipeline(data_file, multi_class=True)
                qdp.preprocess(poi_file=poi_file)

                has_nan = qdp.__dataframe__.isna().any().any()
                if not has_nan:
                    data_df = pd.concat([data_df, qdp.get_dataframe()])
    if multi_class:
        return resample_df(data_df, M_TARGET, M_TARGET)
    else:
        single_dfs = []
        for target in S_TARGETS:
            single_dfs.append(resample_df(data_df, target, S_TARGETS))
        return single_dfs


def fit_model(trainX, trainy):
    # define model
    model = Sequential()
    model.add(Dense(25, input_dim=len(FEATURES), activation="relu"))
    model.add(Dense(7, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    # fit model
    model.fit(trainX, trainy, epochs=500, verbose=0)
    return model


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
            # flatten predictions to [rows, members x probabilities]
            stackX = stackX.reshape(
                (stackX.shape[0], stackX.shape[1] * stackX.shape[2])
            )
    return stackX


def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat


# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = "QModel/SavedModels/model_" + str(i + 1) + ".h5"
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print(">loaded %s" % filename)
    return all_models


# define stacked model from multiple member input models
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
    for layer in model.layers:
        # make not trainable
        layer.trainable = False
        # rename to avoid 'unique layer name' issue
        layer._name = "ensemble_" + str(i + 1) + "_" + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(10, activation="relu")(merge)
    output = Dense(3, activation="softmax")(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file="model_graph.png")
    # compile
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # encode output data
    inputy_enc = to_categorical(inputy)
    # fit model
    model.fit(X, inputy_enc, epochs=300, verbose=0)


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)


# grab the MNIST dataset (if this is your first time using this
# dataset then the 11MB download may take a minute)
print("[INFO] accessing data directory...")
train_content, test_content = load_content(PATH, size=200)
train_data = build_dataset(train_content, multi_class=True)
test_data = build_dataset(test_content, multi_class=True)

train_X = train_data[FEATURES]
train_y = to_categorical(train_data[M_TARGET])
test_X = train_data[FEATURES]
test_y = to_categorical(train_data[M_TARGET])
n_inputs = len(FEATURES)

# fit and save models
n_members = 5
for i in range(n_members):
    # fit model
    model = fit_model(train_X, train_y)
    # save model
    filename = "QModel/SavedModels/model_" + str(i + 1) + ".h5"
    model.save(filename)
    print(">Saved %s" % filename)


# load all models
n_members = 5
members = load_all_models(n_members)
print("Loaded %d models" % len(members))

# evaluate standalone models on test dataset
for model in members:
    testy_enc = to_categorical(test_y)
    _, acc = model.evaluate(test_X, testy_enc, verbose=0)
    print("Model Accuracy: %.3f" % acc)


# fit stacked model using the ensemble
model = fit_stacked_model(members, test_X, test_y)

# evaluate model on test set
yhat = stacked_prediction(members, model, test_X)
acc = accuracy_score(test_y, yhat)
print("Stacked Test Accuracy: %.3f" % acc)

print(train_X.shape, test_X.shape)
# load all models
n_members = 5
members = load_all_models(n_members)
print("Loaded %d models" % len(members))
# define ensemble model
stacked_model = define_stacked_model(members)
# fit stacked model on test dataset
fit_stacked_model(stacked_model, test_X, test_y)
# make predictions and evaluate
yhat = predict_stacked_model(stacked_model, test_X)
yhat = argmax(yhat, axis=1)
acc = accuracy_score(test_y, yhat)
print("Stacked Test Accuracy: %.3f" % acc)
