# -*- coding: utf-8 -*-
"""QATCH combined model predict POIs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J8TFIM9l2xZhdZuV_4vQHKNsSP_Jvwm0
"""

# Commented out IPython magic to ensure Python compatibility.
# %autosave 60

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
plt.ion()
import numpy as np
import csv
import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
# from keras.layers.merge import Maximum, Minimum
import sklearn
import sklearn.model_selection
np.set_printoptions(precision=3, suppress=True) # Make numpy values easier to read.
from random import random, randint

HOW_MANY_DATA_POINTS_PER_ROW = 1000
PRECISION = 100

def returnTrainingData(data_path, poi_path):
  filename = os.path.split(data_path)[1]
  good = True # "good" in filename.lower()

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

        # import POIs, convert indices to timestamps
        pois = np.loadtxt(poi_path).astype(int)
        #print(pois, xs[pois])
        # pois = PRECISION * xs[pois]
        pois = (pois / len(xs) * HOW_MANY_DATA_POINTS_PER_ROW).astype(int)
        #pois = 1000 * pois / len(xs)
        #print(pois)
        #pois = (len(xs)*pois / 1000).astype(int)
        #print(pois, len(xs), xs[-1])

        t_0p5 = 0       #if (xs[-1] < 0.5)    else next(x for x,t in enumerate(xs) if t > 0.5)
        t_1p0 = 100     #if (len(xs) < 500)   else next(x for x,t in enumerate(xs) if t > 1.0)

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

        if False:
          plt.figure()
          plt.title(f"{filename[:-len('.csv')]}: good = {good}")
          plt.plot(xs, ys, 'r-')
          for y in range(len(pois)):
            temp = next(x for x,t in enumerate(xs) if t > pois[y]/PRECISION)
            plt.plot(pois[y]/PRECISION, ys[temp], 'gx')

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

        if False:
          for y in range(len(pois)):
            temp = next(x for x,t in enumerate(lin_xs) if t > pois[y]/PRECISION)
            plt.plot(pois[y]/PRECISION, lin_ys[temp], 'go')

        return int(good), lin_xs, lin_ys, lin_ys_freq, lin_ys_diff, pois #relative_time, resonance_frequency, dissipation, difference_curve

training_name = []
training_time = []
training_diss = []
training_freq = []
training_diff = []
training_good = []
training_poi1 = []
training_poi2 = []
training_poi3 = []
training_poi4 = []
training_poi5 = []
training_poi6 = []

print("Print Good Graph")
good_count = 0
bad_count = 0
path_root = "content/training_data_with_points/"
training_files = os.listdir(path_root)
# show 10 random plots from training data
plot_count = 10
plots_to_show = []
plot_names = []
while len(plots_to_show) < min(len(training_files), plot_count):
  x = randint(0, len(training_files)-1)
  if x % 2 == 1: x -= 1 # if x refers to 'poi' file, point to 'data' instead
  if x not in plots_to_show:
    plots_to_show.append(x)
for x in range(len(training_files)):
  data_path = os.path.join(path_root, training_files[x])
  if not os.path.isfile(data_path): continue #skip folders
  if "_poi.csv" in data_path: continue #skip POI files
  if data_path.endswith('.xml'): continue #skip XML files
  if "_lower.csv" in data_path: continue
  if "_upper.csv" in data_path: continue
  poi_path = data_path.replace(".csv", "_poi.csv") #assumes POI has same filename root

  filename = os.path.split(data_path)[1]
  good = True # "good" in filename.lower()
  bad = "bad" in filename.lower()
  if good == bad: continue #skip files not marked good/bad

  good, time, diss, freq, diff, pois = returnTrainingData(data_path, poi_path)
  (poi1, poi2, poi3, poi4, poi5, poi6) = pois

  classify = False
  if x in plots_to_show or classify:
    if x in plots_to_show:
      plot_names.append(training_files[x][:-len("_poi.csv")])
    if classify:
      if x == 0:
        plt.figure()
        with open("bad_files.txt", "r") as cfile:
          bad_files = cfile.readlines()
      else:
        if training_files[x] + '\n' in bad_files:
          print("Skipping bad file:", training_files[x])
          continue
        plt.cla()
    else:
      plt.figure() # append to existing figure, not a new one
    plt.title(f"{training_files[x][:-len('_poi.csv')]}: good = {good}")
    plt.plot(time[pois[0]:pois[-1]], diss[pois[0]:pois[-1]], 'r-')
    num_pts = len(time)
    #idxs = tuple(int(num_pts*item/1000) for item in pois)
    #print(pois)
    #print(idxs)
    for y in range(len(pois)):
      #print(time[idxs[x]])
      # temp = next(x for x,t in enumerate(time) if t > pois[y]/PRECISION)
      plt.plot(time[pois[y]], diss[pois[y]], 'bx')

    if classify:
      good = input(f"{x}/{len(training_files)}: {training_files[x]} [y/n]: ").lower().strip() != 'n'
      if not good:
        with open("bad_files.txt", "a") as cfile:
          cfile.write(training_files[x] + "\n")

  training_name.append(training_files[x][:-len("_poi.csv")])
  training_good.append(good)
  training_time.append(time)
  training_diss.append(diss)
  training_freq.append(freq)
  training_diff.append(diff)
  training_poi1.append(poi1)
  training_poi2.append(poi2)
  training_poi3.append(poi3)
  training_poi4.append(poi4)
  training_poi5.append(poi5)
  training_poi6.append(poi6)

print(training_good)

print("Press any key to close . . .")
input()

init = []
blip1 = []
blip2 = []
blip3 = []
for x in range(len(training_poi1)):
  pois = (0, training_poi1[x], training_poi2[x], training_poi3[x], training_poi4[x], training_poi5[x], training_poi6[x])
  total_size = pois[6] - pois[1]
  _init_pct = (pois[2] - pois[1]) / total_size * 100
  blip1_pct = (pois[4] - pois[1]) / total_size * 100
  blip2_pct = (pois[5] - pois[1]) / total_size * 100
  blip3_pct = (pois[6] - pois[1]) / total_size * 100
  init.append(_init_pct)
  blip1.append(blip1_pct)
  blip2.append(blip2_pct)
  blip3.append(blip3_pct)
#print(pois)
print(np.average(init), np.average(blip1), np.average(blip2), np.average(blip3))
blip1

# qatch specific
time_features = training_time
diss_features = training_diss
freq_features = training_freq
diff_features = training_diff
data_label0 = training_good
data_label1 = training_poi1
data_label2 = training_poi2
data_label3 = training_poi3
data_label4 = training_poi4
data_label5 = training_poi5
data_label6 = training_poi6

#format as numpy
time_features = np.array(time_features)
diss_features = np.array(diss_features)
freq_features = np.array(freq_features)
diff_features = np.array(diff_features)
data_label0 = np.array(data_label0).astype(int)
data_label1 = np.array(data_label1).astype(int)
data_label2 = np.array(data_label2).astype(int)
data_label3 = np.array(data_label3).astype(int)
data_label4 = np.array(data_label4).astype(int)
data_label5 = np.array(data_label5).astype(int)
data_label6 = np.array(data_label6).astype(int)
data_labels = (data_label0, data_label1, data_label2, data_label3, data_label4, data_label5, data_label6)

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

"""Time for ML Training"""

NUM_DIMS = 1024
DROPOUTS = 0.1

# Define model layers.
input_layer0 = layers.Input(name="time", shape=(HOW_MANY_DATA_POINTS_PER_ROW,))
first_layer0 = time_normalize(input_layer0)
input_layer1 = layers.Input(name="diss", shape=(HOW_MANY_DATA_POINTS_PER_ROW,))
first_layer1 = diss_normalize(input_layer1)
input_layer2 = layers.Input(name="freq", shape=(HOW_MANY_DATA_POINTS_PER_ROW,))
first_layer2 = freq_normalize(input_layer2)
input_layer3 = layers.Input(name="diff", shape=(HOW_MANY_DATA_POINTS_PER_ROW,))
first_layer3 = diff_normalize(input_layer3)

second_layer0_0 = layers.Dense(NUM_DIMS, activation='relu')(first_layer0)
third_layer0_0 = layers.Dropout(DROPOUTS)(second_layer0_0)
fourth_layer0_0 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer0_0)
second_layer0_1 = layers.Dense(NUM_DIMS, activation='relu')(first_layer1)
third_layer0_1 = layers.Dropout(DROPOUTS)(second_layer0_1)
fourth_layer0_1 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer0_1)
second_layer0_2 = layers.Dense(NUM_DIMS, activation='relu')(first_layer2)
third_layer0_2 = layers.Dropout(DROPOUTS)(second_layer0_2)
fourth_layer0_2 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer0_2)
second_layer0_3 = layers.Dense(NUM_DIMS, activation='relu')(first_layer3)
third_layer0_3 = layers.Dropout(DROPOUTS)(second_layer0_3)
fourth_layer0_3 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer0_3)
combined_output0 = layers.concatenate([fourth_layer0_0, fourth_layer0_1, fourth_layer0_2, fourth_layer0_3])
# Y0 output will be fed from the fourth layer
y0_output = layers.Dense(1, name='good')(combined_output0)
#constrain_output0 = y0_output #Maximum()([0, y0_output])

second_layer1_0 = layers.Dense(NUM_DIMS, activation='relu')(first_layer0)
third_layer1_0 = layers.Dropout(DROPOUTS)(second_layer1_0)
fourth_layer1_0 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer1_0)
second_layer1_1 = layers.Dense(NUM_DIMS, activation='relu')(first_layer1)
third_layer1_1 = layers.Dropout(DROPOUTS)(second_layer1_1)
fourth_layer1_1 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer1_1)
second_layer1_2 = layers.Dense(NUM_DIMS, activation='relu')(first_layer2)
third_layer1_2 = layers.Dropout(DROPOUTS)(second_layer1_2)
fourth_layer1_2 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer1_2)
second_layer1_3 = layers.Dense(NUM_DIMS, activation='relu')(first_layer3)
third_layer1_3 = layers.Dropout(DROPOUTS)(second_layer1_3)
fourth_layer1_3 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer1_3)
combined_output1 = layers.concatenate([fourth_layer1_0, fourth_layer1_1, fourth_layer1_2, fourth_layer1_3])
# Y1 output will be fed from the fourth layer
y1_output = layers.Dense(1, name='start')(combined_output1)
#constrain_output1 = Maximum(name='start')([constrain_output0, y1_output])

second_layer2_0 = layers.Dense(NUM_DIMS, activation='relu')(first_layer0)
third_layer2_0 = layers.Dropout(DROPOUTS)(second_layer2_0)
fourth_layer2_0 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer2_0)
second_layer2_1 = layers.Dense(NUM_DIMS, activation='relu')(first_layer1)
third_layer2_1 = layers.Dropout(DROPOUTS)(second_layer2_1)
fourth_layer2_1 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer2_1)
second_layer2_2 = layers.Dense(NUM_DIMS, activation='relu')(first_layer2)
third_layer2_2 = layers.Dropout(DROPOUTS)(second_layer2_2)
fourth_layer2_2 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer2_2)
second_layer2_3 = layers.Dense(NUM_DIMS, activation='relu')(first_layer3)
third_layer2_3 = layers.Dropout(DROPOUTS)(second_layer2_3)
fourth_layer2_3 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer2_3)
combined_output2 = layers.concatenate([fourth_layer2_0, fourth_layer2_1, fourth_layer2_2, fourth_layer2_3])
# Y2 output will be fed from the fourth layer
y2_output = layers.Dense(1, name='stop')(combined_output2)
#constrain_output2 = Maximum(name='stop')([constrain_output1, y2_output])

second_layer3_0 = layers.Dense(NUM_DIMS, activation='relu')(first_layer0)
third_layer3_0 = layers.Dropout(DROPOUTS)(second_layer3_0)
fourth_layer3_0 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer3_0)
second_layer3_1 = layers.Dense(NUM_DIMS, activation='relu')(first_layer1)
third_layer3_1 = layers.Dropout(DROPOUTS)(second_layer3_1)
fourth_layer3_1 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer3_1)
second_layer3_2 = layers.Dense(NUM_DIMS, activation='relu')(first_layer2)
third_layer3_2 = layers.Dropout(DROPOUTS)(second_layer3_2)
fourth_layer3_2 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer3_2)
second_layer3_3 = layers.Dense(NUM_DIMS, activation='relu')(first_layer3)
third_layer3_3 = layers.Dropout(DROPOUTS)(second_layer3_3)
fourth_layer3_3 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer3_3)
combined_output3 = layers.concatenate([fourth_layer3_0, fourth_layer3_1, fourth_layer3_2, fourth_layer3_3])
# Y3 output will be fed from the fourth layer
y3_output = layers.Dense(1, name='post')(combined_output3)
#constrain_output3 = Maximum(name='post')([constrain_output2, y3_output])

second_layer4_0 = layers.Dense(NUM_DIMS, activation='relu')(first_layer0)
third_layer4_0 = layers.Dropout(DROPOUTS)(second_layer4_0)
fourth_layer4_0 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer4_0)
second_layer4_1 = layers.Dense(NUM_DIMS, activation='relu')(first_layer1)
third_layer4_1 = layers.Dropout(DROPOUTS)(second_layer4_1)
fourth_layer4_1 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer4_1)
second_layer4_2 = layers.Dense(NUM_DIMS, activation='relu')(first_layer2)
third_layer4_2 = layers.Dropout(DROPOUTS)(second_layer4_2)
fourth_layer4_2 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer4_2)
second_layer4_3 = layers.Dense(NUM_DIMS, activation='relu')(first_layer3)
third_layer4_3 = layers.Dropout(DROPOUTS)(second_layer4_3)
fourth_layer4_3 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer4_3)
combined_output4 = layers.concatenate([fourth_layer4_0, fourth_layer4_1, fourth_layer4_2, fourth_layer4_3])
# Y4 output will be fed from the fourth layer
y4_output = layers.Dense(1, name='blip1')(combined_output4)
#constrain_output4 = Maximum(name='blip1')([constrain_output3, y4_output])

second_layer5_0 = layers.Dense(NUM_DIMS, activation='relu')(first_layer0)
third_layer5_0 = layers.Dropout(DROPOUTS)(second_layer5_0)
fourth_layer5_0 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer5_0)
second_layer5_1 = layers.Dense(NUM_DIMS, activation='relu')(first_layer1)
third_layer5_1 = layers.Dropout(DROPOUTS)(second_layer5_1)
fourth_layer5_1 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer5_1)
second_layer5_2 = layers.Dense(NUM_DIMS, activation='relu')(first_layer2)
third_layer5_2 = layers.Dropout(DROPOUTS)(second_layer5_2)
fourth_layer5_2 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer5_2)
second_layer5_3 = layers.Dense(NUM_DIMS, activation='relu')(first_layer3)
third_layer5_3 = layers.Dropout(DROPOUTS)(second_layer5_3)
fourth_layer5_3 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer5_3)
combined_output5 = layers.concatenate([fourth_layer5_0, fourth_layer5_1, fourth_layer5_2, fourth_layer5_3])
# Y5 output will be fed from the fourth layer
y5_output = layers.Dense(1, name='blip2')(combined_output5)
#constrain_output5 = Maximum(name='blip2')([constrain_output4, y5_output])

second_layer6_0 = layers.Dense(NUM_DIMS, activation='relu')(first_layer0)
third_layer6_0 = layers.Dropout(DROPOUTS)(second_layer6_0)
fourth_layer6_0 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer6_0)
second_layer6_1 = layers.Dense(NUM_DIMS, activation='relu')(first_layer1)
third_layer6_1 = layers.Dropout(DROPOUTS)(second_layer6_1)
fourth_layer6_1 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer6_1)
second_layer6_2 = layers.Dense(NUM_DIMS, activation='relu')(first_layer2)
third_layer6_2 = layers.Dropout(DROPOUTS)(second_layer6_2)
fourth_layer6_2 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer6_2)
second_layer6_3 = layers.Dense(NUM_DIMS, activation='relu')(first_layer3)
third_layer6_3 = layers.Dropout(DROPOUTS)(second_layer6_3)
fourth_layer6_3 = layers.Dense(NUM_DIMS/2, activation='relu')(third_layer6_3)
combined_output6 = layers.concatenate([fourth_layer6_0, fourth_layer6_1, fourth_layer6_2, fourth_layer6_3])
# Y6 output will be fed from the fourth layer
y6_output = layers.Dense(1, name='blip3')(combined_output6)
#constrain_output6 = Maximum(name='blip3')([constrain_output5, y6_output])
#constrain_output6 = Minimum()([constrain_output6_0, HOW_MANY_DATA_POINTS_PER_ROW*PRECISION])

# Define the model with the input layer
# and a list of output layers
csv_model = tf.keras.Model(inputs=[input_layer0, input_layer1, input_layer2, input_layer3],
                          outputs=[y0_output, y1_output, y2_output, y3_output, y4_output, y5_output, y6_output])

# Specify the optimizer, and compile the model with loss functions for both outputs
optimizer = tf.optimizers.Adam(learning_rate=0.01)

csv_model.compile(optimizer=optimizer,
                  loss={'good': tf.keras.losses.BinaryCrossentropy(),
                        'start': tf.keras.losses.MeanAbsoluteError(),
                        'stop': tf.keras.losses.MeanAbsoluteError(),
                        'post': tf.keras.losses.MeanAbsoluteError(),
                        'blip1': tf.keras.losses.MeanAbsoluteError(),
                        'blip2': tf.keras.losses.MeanAbsoluteError(),
                        'blip3': tf.keras.losses.MeanAbsoluteError()},
                  metrics={'good': tf.keras.metrics.BinaryCrossentropy(),
                           'start': tf.keras.metrics.MeanAbsoluteError(),
                           'stop': tf.keras.metrics.MeanAbsoluteError(),
                           'post': tf.keras.metrics.MeanAbsoluteError(),
                           'blip1': tf.keras.metrics.MeanAbsoluteError(),
                           'blip2': tf.keras.metrics.MeanAbsoluteError(),
                           'blip3': tf.keras.metrics.MeanAbsoluteError()})

#split train / validation sets
if True:
  x_train0, x_val0, x_train1, x_val1, x_train2, x_val2, x_train3, x_val3, y_train0, y_val0, y_train1, y_val1, y_train2, y_val2, y_train3, y_val3, y_train4, y_val4, y_train5, y_val5, y_train6, y_val6 = sklearn.model_selection.train_test_split(time_features,diss_features,freq_features,diff_features,
                                                                                                                                                                                                                                                  data_label0,data_label1,data_label2,data_label3,data_label4,data_label5,data_label6,
                                                                                                                                                                                                                                                  test_size=0.2)
else:
  x_train0 = x_val0 = time_features
  x_train1 = x_val1 = diss_features
  x_train2 = x_val2 = freq_features
  x_train3 = x_val3 = diff_features
  y_train0 = y_val0 = data_label0
  y_train1 = y_val1 = data_label1
  y_train2 = y_val2 = data_label2
  y_train3 = y_val3 = data_label3
  y_train4 = y_val4 = data_label4
  y_train5 = y_val5 = data_label5
  y_train6 = y_val6 = data_label6

history = csv_model.fit([x_train0,x_train1,x_train2,x_train3], [y_train0,y_train1,y_train2,y_train3,y_train4,y_train5,y_train6],
                        validation_data=([x_val0,x_val1,x_val2,x_val3], [y_val0,y_val1,y_val2,y_val3,y_val4,y_val5,y_val6]),
                        epochs=25, batch_size=(len(x_train0) if len(x_train0) < 100 else 100), verbose=1)

#save and reload the model - uncomment following if you want to save the model and reload it
#MODEL_PATH = 'SavedModel.tf'
#csv_model.save(MODEL_PATH)
#reloaded_model = tf.keras.models.load_model(MODEL_PATH)

csv_model.evaluate([x_val0,x_val1,x_val2,x_val3], [y_val0,y_val1,y_val2,y_val3,y_val4,y_val5,y_val6])

history_dict = history.history
history_dict.keys()

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

print("Press any key to close . . .")
input()

correct = []
for input_diss in training_diss:
  i = len(correct)

  predictions = csv_model([training_time[i], input_diss, training_freq[i], training_diff[i]])
  val0 = np.round(predictions[0].numpy()[0][0]).astype(int)
  val1 = np.round(predictions[1].numpy()[0][0]).astype(int)
  val2 = np.round(predictions[2].numpy()[0][0]).astype(int)
  val3 = np.round(predictions[3].numpy()[0][0]).astype(int)
  val4 = np.round(predictions[4].numpy()[0][0]).astype(int)
  val5 = np.round(predictions[5].numpy()[0][0]).astype(int)
  val6 = np.round(predictions[6].numpy()[0][0]).astype(int)
  vals = (val0, val1, val2, val3, val4, val5, val6)
  pois = (training_poi1[i], training_poi2[i], training_poi3[i], training_poi4[i], training_poi5[i], training_poi6[i])

  valid = True
  #valid = (val0 < val1 < val2 < val3 < val4 < val5 < val6)
  valid = valid and val0 == good

  diffs = []
  for x in range(len(pois)):
    diff = int(abs(vals[x+1] - pois[x]))
    diffs.append(diff)
    if diff > PRECISION:
      valid = False

  correct.append(valid)

  print(training_name[i], valid)
  pois = tuple(int(item) for item in pois)
  print(pois, vals, diffs)

  #num_pts = len(training_time[i])
  #idxs_predict = tuple(max(0, min(num_pts-1, int(num_pts*item/1000))) for item in vals)
  #idxs_actual = tuple(max(0, min(num_pts-1, int(num_pts*item/1000))) for item in pois)
  #print("ids: ", idxs, num_pts)

  if training_name[i] in plot_names:
    plt.figure()
    plt.title(f"{training_name[i]}: good = {valid}")
    plt.plot(training_time[i], training_diss[i], 'r-')
    for y in range(len(pois)):
      # # try:
      # #   temp1 = next(x for x,t in enumerate(training_time[i]) if t > vals[y+1]/PRECISION)
      # # except:
      # #   temp1 = 0
      # # temp2 = next(x for x,t in enumerate(training_time[i]) if t > pois[y]/PRECISION)
      # plt.plot(vals[y+1]/PRECISION, training_diff[i][temp1], 'bx')
      # plt.plot(pois[y]/PRECISION, training_diff[i][temp2], 'gx')
      idx = max(0, min(HOW_MANY_DATA_POINTS_PER_ROW-1, vals[y+1]))
      plt.plot(training_time[i][idx], training_diss[i][idx], 'bx')
      plt.plot(training_time[i][pois[y]], training_diss[i][pois[y]], 'gx')

(unique, counts) = np.unique(correct, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print("Results")
print(frequencies)

print("Press any key to close . . .")
input()