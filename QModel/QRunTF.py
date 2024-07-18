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
from imblearn.over_sampling import  RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from tensorflow.python.client import device_lib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


print(get_available_devices())

hidden_units = 150  # how many neurons in the hidden layer
activation = "relu"  # activation function for hidden layer
l2 = 0.01  # regularization - how much we penalize large parameter values
learning_rate = 0.01  # how big our steps are in gradient descent
epochs = 10000000  # how many epochs to train for
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
CLASS = "Class"

class MyHyperModel(HyperModel):
    def build(self, hp):
        
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        first_layer_units = hp.Int('first_layer_units', min_value=32, max_value=512, step=32)

        model = models.Sequential()
        model.add(keras.layers.Dense(units=first_layer_units, activation='relu', input_shape=len(FEATURES)))
        model.add(layers.Dense(input_dim=len(FEATURES), units=1, activation="sigmoid"))

        # Tune the learning rate for the optimizer
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=hp.Choice('loss', values=['binary_crossentropy', 'hinge']),
             metrics=[
                hp.Choice('metrics', values=[
                    'accuracy', 
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.AUC(name='roc'),  
                    tf.keras.metrics.Precision(name='precision'),  
                    tf.keras.metrics.Recall(name='recall'),  
                    tf.keras.metrics.TrueNegatives(name='tnr')  
                ])
            ]
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
                    plt.figure()
                    plt.plot(
                        normalize(qdp.__dataframe__["Dissipation"]), label="Dissipation"
                    )

                    predictions = []
                    qdp.__dataframe__ = RobustScaler().fit_transform(qdp.__dataframe__)
                    predictions = model.predict(qdp.__dataframe__)

                    plt.plot(normalize(predictions), linestyle='--', label="Predictions")
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
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title('t-SNE')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar(scatter, label='Target Value')
    plt.show()

def pca_view(X, y):
    print('[INFO] building PCA')
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title('PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Target Value')
    plt.show()



def resample(data, target):
    print(f'[INFO] resampling {target}')
    y = data[target].values
    X = data.drop(columns=TARGETS)
    

    under = RandomUnderSampler(sampling_strategy='majority')
    steps = [('under_sample', under)]
    pipeline = Pipeline(steps=steps)
    
    X, y = pipeline.fit_resample(X, y)

    X = RobustScaler().fit_transform(X)

    tsne_view(X, y)
    pca_view(X, y)
    return X, y

def load_and_split_data(data_dir):
    count = 0
    data_df = pd.DataFrame()
    content = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            content.append(os.path.join(root, file))
    for filename in tqdm(content, desc="<<Processing Files>>"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            # if count > 25:
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
    
    
    y = data_df[TARGETS[4]].values
    X = data_df.drop(columns=TARGETS)

    X, y = resample(data_df, TARGETS[4])
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return ((train_X, train_y), (test_X, test_y))


# grab the MNIST dataset (if this is your first time using this
# dataset then the 11MB download may take a minute)
print("[INFO] accessing data directory...")

((train_X, train_y), (test_X, test_y)) = load_and_split_data(CONTENT_PATH)
n_inputs = len(FEATURES)

visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)


# define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)

model.compile(optimizer='adam', loss='mse')

overfitCallback = EarlyStopping(monitor='loss', min_delta=0.001, patience=20)
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, callbacks=[overfitCallback], validation_data=(test_X, test_y))

# Plot the training and validation accuracy and loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot training & validation loss values
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(['Train', 'Validation'], loc='upper left')

plt.show()
evaluate_on_dir("content/VOYAGER_PROD_DATA", model, "Target 5")
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
