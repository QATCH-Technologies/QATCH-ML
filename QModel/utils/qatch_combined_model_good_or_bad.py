# -*- coding: utf-8 -*-
"""QATCH combined model good or bad.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nIIjOQlfGfehenC4fULtRZxksHdGITeG

Sample ML model that learns to detect "good" data trends from "bad" ones
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
plt.ion()
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

"""For the sake of experimentation: Create a fake CSV with two types of tabular data.  "Good graphs" (Sin curves + noise) and "Bad graphs" (just noise).  Eventually replace this with your actual files in CSV format."""

HOW_MANY_DATA_POINTS_PER_ROW = 1000
def returnFakeRow(good=True):
  x = np.linspace(0, np.random.normal(15, 5, 1), HOW_MANY_DATA_POINTS_PER_ROW)
  noise1 = np.random.normal(9, 2, 1)
  noise2 = np.random.normal(0, .6, x.shape)
  return noise1 * np.sin(x) + noise2 if good else noise1 + noise2
def returnTrainingData(data_path):
  filename = os.path.split(data_path)[1]
  good = filename.lower().strip().startswith("good")

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

        t_0p5 = 0       if (xs[-1] < 0.5)    else next(x for x,t in enumerate(xs) if t > 0.5)
        t_1p0 = 100     if (len(xs) < 500)   else next(x for x,t in enumerate(xs) if t > 1.0)

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
  good = filename.lower().strip().startswith("good")
  bad = filename.lower().strip().startswith("bad")
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

# print("Print a Bad Graph")
# for x in range(5):
#   time, diss, freq, diff = returnTrainingData(False, x)
#   plt.figure()
#   plt.plot(time, freq, 'g-')
#   #plt.figure()
#   plt.plot(time, diss, 'r-')
#   #plt.figure()
#   plt.plot(time, diff, 'b-')
#   #break # only print 1
#   training_time.append(time)
#   training_diss.append(diss)
#   training_freq.append(freq)
#   training_diff.append(diff)
#   training_good.append(False)

CSV_PATH = 'content/dummyData.csv'
HOW_MANY_ROWS_FOR_EACH_GOOD_AND_BAD = 5000
#create 1,000 samples.  500 good, 500 bag and save to dummyData.csv
def createCSV():
  f = open(CSV_PATH, 'w')
  writer = csv.writer(f)

  #write header row
  row = np.insert(x,0,-1) #reserving header -1 for good or bad column
  writer.writerow(row)

  #Write Good rows
  for i in range(0,HOW_MANY_ROWS_FOR_EACH_GOOD_AND_BAD):
    row = returnFakeRow(True)
    label = 1 #1 = Good data
    row = np.insert(row,0,label) #add label as first column
    writer.writerow(row)
  #Write Bad rows
  for i in range(0,HOW_MANY_ROWS_FOR_EACH_GOOD_AND_BAD):
    row = returnFakeRow(False)
    label = 0 #0 = Bad data
    row = np.insert(row,0,label) #add label as first column
    writer.writerow(row)
  f.close()


createCSV()

"""Import CSV data"""

csv_import = pd.read_csv(CSV_PATH)

#print first 5 rows
csv_import.head()

csv_features = csv_import.copy()
csv_labels = csv_features.pop('-1.0') #first column is good (1.0) vs bad (0.0).  Therefore label

#format as numpy
csv_features = np.array(csv_features)
csv_labels = np.array(csv_labels).astype(int)

# qatch specific
csv_features = [training_diss, training_freq, training_diff]
csv_labels = training_good

#format as numpy
csv_features = np.array(csv_features)
csv_labels = np.array(csv_labels).astype(int)

print(len(csv_features[0]), "x", len(csv_features))
print(csv_features)
print(len(csv_labels))
print(csv_labels)

#print how many bad vs good we've imported
(unique, counts) = np.unique(csv_labels, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)

"""Time for ML training"""

#normalize for optimal train
normalize = layers.Normalization(input_shape=[3,], axis=None)
normalize.adapt(csv_features)

if True:
  # the first branch operates on the first input
  inputA = layers.Input(shape=(HOW_MANY_DATA_POINTS_PER_ROW,))
  inputB = layers.Input(shape=(HOW_MANY_DATA_POINTS_PER_ROW,))
  inputC = layers.Input(shape=(HOW_MANY_DATA_POINTS_PER_ROW,))

  x = layers.Dense(64, activation="relu")(inputA)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(64, activation='relu')(x)
  x = layers.Dense(1)(x)
  x = tf.keras.Model(inputs=inputA, outputs=x)
  # the second branch opreates on the second input
  y = layers.Dense(64, activation="relu")(inputB)
  y = layers.Dropout(0.2)(y)
  y = layers.Dense(64, activation='relu')(y)
  y = layers.Dense(1)(y)
  y = tf.keras.Model(inputs=inputB, outputs=y)
  # the third branch operates on the third inpout
  z = layers.Dense(64, activation="relu")(inputC)
  z = layers.Dropout(0.2)(z)
  z = layers.Dense(64, activation='relu')(z)
  z = layers.Dense(1)(z)
  z = tf.keras.Model(inputs=inputC, outputs=z)
  # combine the output of the two branches
  combined = layers.concatenate([x.output, y.output, z.output])
  # apply a FC layer and then a regression prediction on the
  # combined outputs
  out = layers.Dense(2, activation="relu")(combined)
  out = layers.Dense(1, activation="linear")(out)
  # our model will accept the inputs of the three branches and
  # then output a single value
  csv_model = tf.keras.Model(inputs=[x.input, y.input, z.input], outputs=out)
  csv_model.compile(loss = 'mean_absolute_error', #from_logits=True
                    optimizer = tf.optimizers.Adam(learning_rate=0.003))
elif True:
  csv_model = tf.keras.Sequential([
    normalize,
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2), #quick experiments showed overfitting... so introduced dropout to prevent
    layers.Dense(64, activation='relu'),
    #layers.Dense(32, activation='relu'),
    #layers.Dense(16, activation='relu'),
    layers.Dense(1)
    ])
    #BinaryCrossentropy / MeanSquaredError / tf.keras.losses.BinaryCrossentropy()
  csv_model.compile(loss = 'mean_absolute_error', #from_logits=True
                    optimizer = tf.optimizers.Adam(learning_rate=0.003))
else:
  csv_model = tf.keras.Sequential([
    normalize,
    layers.Dense(64, activation='relu'), #a very simple model seems to work... so why complicate...
    #layers.Dropout(0.2), #quick experiments showed overfitting... so introduced dropout to prevent
    #layers.Dense(64, activation='relu'),
    #layers.Dense(16, activation='relu'),
    layers.Dense(1,activation='sigmoid')
  ])
  #BinaryCrossentropy / MeanSquaredError / tf.keras.losses.BinaryCrossentropy()
  csv_model.compile(loss = tf.keras.losses.BinaryCrossentropy(), #from_logits=True
                        optimizer = tf.optimizers.Adam(),metrics=['accuracy'])

#split train / validation sets
x_train, x_val, freq_train, freq_val, diff_train, diff_val, y_train, y_val = sklearn.model_selection.train_test_split(csv_features[0],csv_features[1],csv_features[2],csv_labels,test_size=0.1)
print(x_train)

print("Print a Val Graph")
print(y_val[201])
plt.plot(x, x_val[201], '-')

print("Print another Val Graph")
print(len(y_val))
print(y_val[2])
plt.plot(x_val[2], x_val[2], '-')

print(x_train, freq_train, diff_train)
cum_train = (x_train, freq_train, diff_train)
cum_val = (x_val, freq_val, diff_val)

history = csv_model.fit(x=[x_train, freq_train, diff_train], y=y_train, epochs=100, batch_size=100,validation_data=([x_val, freq_val, diff_val], y_val),
                    verbose=1)

#save and reload the model - uncomment following if you want to save the model and reload it
MODEL_PATH = 'SavedModel/test_model'
csv_model.save(MODEL_PATH)
csv_model = tf.keras.models.load_model(MODEL_PATH)
csv_model.summary()

#tryRow = returnFakeRow(False)
#print(tryRow)
#csv_model(x_train)
#csv_model.evaluate(x_train, y_train)
csv_model.evaluate([x_val,freq_val,diff_val], y_val)

#probability_model = tf.keras.Sequential([csv_model, tf.keras.layers.Softmax()])
#predictions = probability_model.predict([x_train[2]])
print(len(x_train), len(x_train[0]))
print(len(freq_train), len(freq_train[0]))
print(len(diff_train), len(diff_train[0]))
correct = []
for i in range(0,len(x_train)):
  predictions = csv_model([x_train[i],freq_train[i],diff_train[i]])
  val = np.round(predictions.numpy()[0][0]).astype(int)
  correct.insert(i,val == y_train[i])

(unique, counts) = np.unique(correct, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print("Train")
print(frequencies)

correct = []
for i in range(0,len(x_val)):
  predictions = csv_model([x_val[i],freq_val[i],diff_val[i]])
  val = np.round(predictions.numpy()[0][0]).astype(int)
  correct.insert(i,val == y_val[i])

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
path_root = "content/training_data/"
validation_files = os.listdir(path_root)
for x in range(len(validation_files)):
  data_path = os.path.join(path_root, validation_files[x])
  if not os.path.isfile(data_path): continue #skip folders

  filename = os.path.split(data_path)[1]
  good = filename.lower().strip().startswith("gooddata")
  bad = filename.lower().strip().startswith("baddata")
  if good == bad: continue #skip files not marked good/bad

  good, good_xs, good_ys, good_ys_freq, good_ys_diff = returnTrainingData(data_path)

  predictions = csv_model([good_ys])
  val = np.round(predictions.numpy()[0][0]).astype(int)
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

while True:
  good_graph = returnFakeRow(False)
  predictions = csv_model([good_graph])
  val = np.round(predictions.numpy()[0][0]).astype(int)
  if val == 1:
    break
print(val)
plt.plot(x, good_graph, '-')

plt.waitforbuttonpress()