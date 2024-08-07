# -*- coding: utf-8 -*-
"""QATCH separate models good or bad.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1emNu7VAtOs4Qk-GNWuXL34EnJafRa73n
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import csv
import os
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import sklearn
import sklearn.model_selection
np.set_printoptions(precision=3, suppress=True) # Make numpy values easier to read.

HOW_MANY_DATA_POINTS_PER_ROW = 1000

def returnTrainingData(data_path):
  filename = os.path.split(data_path)[1]
  good = filename.lower().strip().find("good") >= 0

  with open(data_path) as f:
        csv_headers = next(f)

        if "Ambient" in csv_headers:
            csv_cols = (2,4,6,7)
        else:
            csv_cols = (2,3,5,6)

        data  = np.loadtxt(data_path, delimiter = ',', skiprows = 1, usecols = csv_cols)
        relative_time = data[:,0]
        temperature = data[:,1]
        resonance_frequency = data[:,2]
        dissipation = data[:,3]

        # raw data
        xs = relative_time
        ys = dissipation

        # t_0p5 = 0       if (xs[-1] < 0.5)    else next(x for x,t in enumerate(xs) if t > 0.5)
        # t_1p0 = 100     if (len(xs) < 500)   else next(x for x,t in enumerate(xs) if t > 1.0)

        try:
          t_0p5 = next(x for x,t in enumerate(xs) if t > 0.5)
        except:
          t_0p5 = 0
        try:
          t_1p0 = next(x for x,t in enumerate(xs) if t > 1.0)
        except:
          t_1p0 = 100

        #t_1p0, done = QtWidgets.QInputDialog.getDouble(None, 'Input Dialog', 'Confirm rough start index:', value=t_1p0)

        # new maths for resonance and dissipation (scaled)
        avg = np.average(resonance_frequency[t_0p5:t_1p0])
        ys = ys * avg / 2
        #ys_fit = ys_fit * avg / 2
        ys = ys - np.amin(ys)
        #ys_fit = ys_fit - np.amin(ys_fit)
        ys_freq = avg - resonance_frequency
        #ys_freq_fit = savgol_filter(ys_freq, smooth_factor, 1)
        ys_diff = ys_freq - ys
        #ys_diff_fit = savgol_filter(ys_diff, smooth_factor, 1)

        t_start = 0
        t_stop = -1

        if False: # only work within 2nd derivative start-stop window
          try:
            env_size = np.amax(ys[t_0p5:t_1p0]) - np.amin(ys[t_0p5:t_1p0])
            env_beg_up = np.average(ys[t_0p5:t_1p0]) + (3*env_size)
            env_beg_dn = np.average(ys[t_0p5:t_1p0]) - (3*env_size)
            env_size = np.amax(ys[-t_1p0:]) - np.amin(ys[-t_1p0:])
            env_end_up = np.average(ys[-t_1p0:]) + (3*env_size)
            env_end_dn = np.average(ys[-t_1p0:]) - (3*env_size)
            t_start = next(x for x,y in enumerate(ys[t_1p0:]) if y > env_beg_up or y < env_beg_dn)
            t_stop = min(len(ys), (len(ys) + t_1p0) - next(x for x,y in enumerate(ys[::-1]) if y > env_end_up or y < env_end_dn))
          except:
            pass

        xs = xs[t_start:t_stop]
        ys = ys[t_start:t_stop]
        ys_freq = ys_freq[t_start:t_stop]
        ys_diff = ys_diff[t_start:t_stop]

        lin_xs = np.linspace(xs[0], xs[-1], HOW_MANY_DATA_POINTS_PER_ROW)
        lin_ys = np.interp(lin_xs, xs, ys)
        lin_ys_freq = np.interp(lin_xs, xs, ys_freq)
        lin_ys_diff = np.interp(lin_xs, xs, ys_diff)

        return int(good), lin_xs, lin_ys, lin_ys_freq, lin_ys_diff #relative_time, resonance_frequency, dissipation, difference_curve

training_time = []
training_diss = []
training_freq = []
training_diff = []
training_good = []

print("Print Good Graph")
good_count = 0
bad_count = 0
path_root = "content/training_data/"
training_files = os.listdir(path_root)
for x in range(len(training_files)):
  data_path = os.path.join(path_root, training_files[x])
  if not os.path.isfile(data_path): continue #skip folders

  filename = os.path.split(data_path)[1]
  good = filename.lower().strip().find("good") >= 0
  bad = filename.lower().strip().find("bad") >= 0
  if good == bad: continue #skip files not marked good/bad

  good, time, diss, freq, diff = returnTrainingData(data_path)

  if True:
    plt.figure()
    plt.title(f"{training_files[x]}: good = {good}")
    plt.plot(time, freq, 'g-')
    #plt.figure()
    plt.plot(time, diss, 'r-')
    #plt.figure()
    plt.plot(time, diff, 'b-')
    #break # only print 1

  training_good.append(good)
  training_time.append(time)
  training_diss.append(diss)
  training_freq.append(freq)
  training_diff.append(diff)

print(training_good)

# qatch specific
time_features = training_time
diss_features = training_diss
freq_features = training_freq
diff_features = training_diff
data_labels = training_good

#format as numpy
time_features = np.array(time_features)
diss_features = np.array(diss_features)
freq_features = np.array(freq_features)
diff_features = np.array(diff_features)
data_labels = np.array(data_labels).astype(int)

print(len(diss_features[0]), "x", len(diss_features))
print(diss_features)
print(len(data_labels))
print(data_labels)

#normalize for optimal train
time_normalize = layers.Normalization()
time_normalize.adapt(time_features)
diss_normalize = layers.Normalization()
diss_normalize.adapt(diss_features)
freq_normalize = layers.Normalization()
freq_normalize.adapt(freq_features)
diff_normalize = layers.Normalization()
diff_normalize.adapt(diff_features)

#print how many bad vs good we've imported
(unique, counts) = np.unique(training_good, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)

TEST_SIZE = 0.1
DROPOUT_RATIO = 0.5
LEARNING_RATE = 0.003
EPOCHS = 1000
BATCH_SIZE = 100
VERBOSE = 1

#split train / validation sets
time_train, time_val, diss_train, diss_val, freq_train, freq_val, diff_train, diff_val, data_train, data_val = sklearn.model_selection.train_test_split(
    time_features,diss_features,freq_features,diff_features,data_labels,test_size=TEST_SIZE)

time_model = tf.keras.Sequential([
    time_normalize,
    layers.Dense(64, activation='relu'),
    layers.Dropout(DROPOUT_RATIO), #quick experiments showed overfitting... so introduced dropout to prevent
    layers.Dense(64, activation='relu'),
    #layers.Dense(32, activation='relu'),
    #layers.Dense(16, activation='relu'),
    layers.Dense(1)
    ])
    #BinaryCrossentropy / MeanSquaredError / tf.keras.losses.BinaryCrossentropy()
time_model.compile(loss = 'mean_absolute_error', #from_logits=True
                  optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE))
history = time_model.fit(x=time_train, y=data_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(time_val, data_val), verbose=VERBOSE)

diss_model = tf.keras.Sequential([
    diss_normalize,
    layers.Dense(64, activation='relu'),
    layers.Dropout(DROPOUT_RATIO), #quick experiments showed overfitting... so introduced dropout to prevent
    layers.Dense(64, activation='relu'),
    #layers.Dense(32, activation='relu'),
    #layers.Dense(16, activation='relu'),
    layers.Dense(1)
    ])
    #BinaryCrossentropy / MeanSquaredError / tf.keras.losses.BinaryCrossentropy()
diss_model.compile(loss = 'mean_absolute_error', #from_logits=True
                  optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE))
history = diss_model.fit(x=diss_train, y=data_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(diss_val, data_val), verbose=VERBOSE)

freq_model = tf.keras.Sequential([
    freq_normalize,
    layers.Dense(64, activation='relu'),
    layers.Dropout(DROPOUT_RATIO), #quick experiments showed overfitting... so introduced dropout to prevent
    layers.Dense(64, activation='relu'),
    #layers.Dense(32, activation='relu'),
    #layers.Dense(16, activation='relu'),
    layers.Dense(1)
    ])
    #BinaryCrossentropy / MeanSquaredError / tf.keras.losses.BinaryCrossentropy()
freq_model.compile(loss = 'mean_absolute_error', #from_logits=True
                  optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE))
history = freq_model.fit(x=freq_train, y=data_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(freq_val, data_val), verbose=VERBOSE)

diff_model = tf.keras.Sequential([
    diff_normalize,
    layers.Dense(64, activation='relu'),
    layers.Dropout(DROPOUT_RATIO), #quick experiments showed overfitting... so introduced dropout to prevent
    layers.Dense(64, activation='relu'),
    #layers.Dense(32, activation='relu'),
    #layers.Dense(16, activation='relu'),
    layers.Dense(1)
    ])
    #BinaryCrossentropy / MeanSquaredError / tf.keras.losses.BinaryCrossentropy()
diff_model.compile(loss = 'mean_absolute_error', #from_logits=True
                  optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE))
history = diff_model.fit(x=diff_train, y=data_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(diff_val, data_val), verbose=VERBOSE)

def data_model(data):
  data_time = data[0]
  data_diss = data[1]
  data_freq = data[2]
  data_diff = data[3]
  predict_time = 0 #max(0, min(1, time_model([data_time]).numpy()[0][0]))
  predict_diss = max(0, min(1, diss_model([data_diss]).numpy()[0][0]))
  predict_freq = max(0, min(1, freq_model([data_freq]).numpy()[0][0]))
  predict_diff = max(0, min(1, diff_model([data_diff]).numpy()[0][0]))
  predictors_count = 3 #ignore time
  predict_data = (predict_time + predict_diss + predict_freq + predict_diff) / predictors_count
  return max(0, min(1, np.round(predict_data).astype(int)))

correct = []
for i in range(0,len(data_train)):
  val = data_model([time_train[i],diss_train[i],freq_train[i],diff_train[i]])
  correct.insert(i,val == data_train[i])

(unique, counts) = np.unique(correct, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print("Train")
print(frequencies)

correct = []
for i in range(0,len(data_val)):
  val = data_model([time_val[i],diss_val[i],freq_val[i],diff_val[i]])
  correct.insert(i,val == data_val[i])

(unique, counts) = np.unique(correct, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print("Validation")
print(frequencies)

history_dict = history.history
history_dict.keys()

#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

correct = []
# path_root = "content/training_data/"
validation_files = os.listdir(path_root)
for x in range(len(validation_files)):
  data_path = os.path.join(path_root, validation_files[x])
  if not os.path.isfile(data_path): continue #skip folders

  filename = os.path.split(data_path)[1]
  good = filename.lower().strip().find("good") >= 0
  bad = filename.lower().strip().find("bad") >= 0
  if good == bad: continue #skip files not marked good/bad

  good, good_xs, good_ys, good_ys_freq, good_ys_diff = returnTrainingData(data_path)

  val = data_model([good_xs, good_ys, good_ys_freq, good_ys_diff])
  print(val)

  if val != good:
    plt.figure()
    plt.title(f"{validation_files[x]}: good = {val}")
    plt.plot(good_xs, good_ys, 'r-')
    plt.plot(good_xs, good_ys_freq, 'g-')
    plt.plot(good_xs, good_ys_diff, 'b-')

  correct.insert(i,val == good)

(unique, counts) = np.unique(correct, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print("Results")
print(frequencies)

#save and reload the model - uncomment following if you want to save the model and reload it
MODEL_PATH = 'SavedModel/Model #2 (91 of 93)/{0}'
time_model.save(MODEL_PATH.format("time_model"))
diss_model.save(MODEL_PATH.format("diss_model"))
freq_model.save(MODEL_PATH.format("freq_model"))
diff_model.save(MODEL_PATH.format("diff_model"))

time_model = tf.keras.models.load_model(MODEL_PATH.format("time_model"))
diss_model = tf.keras.models.load_model(MODEL_PATH.format("diss_model"))
freq_model = tf.keras.models.load_model(MODEL_PATH.format("freq_model"))
diff_model = tf.keras.models.load_model(MODEL_PATH.format("diff_model"))

time_model.summary()
diss_model.summary()
freq_model.summary()
diff_model.summary()

import shutil
import os
MODEL_PATH = 'SavedModel/Model #2 (91 of 93)/{0}'
output_filename = os.path.split(MODEL_PATH)[0]
dir_name = output_filename
shutil.make_archive(output_filename, 'zip', dir_name)