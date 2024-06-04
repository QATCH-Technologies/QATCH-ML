import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pandas as pd
import tensorflow as tf
import sklearn
import sklearn.model_selection

from tensorflow.keras import layers
from random import random, randint

np.set_printoptions(precision=3, suppress=True)  # Make numpy values easier to read.
plt.ion()

PPR = 1000
PRECISION = 100
PATH_TO_TRAINING_DATA = "content/training_data_with_points/"
NUM_RAND_PLOTS_TO_DISP = 10


def status(message):
    print("(status)", message)


def error(message):
    print("(err)", message)


def echo(message):
    print("(echo)", message)


def info(message):
    print("(info)", message)


def plot_loss(history):
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, "r", label="Training Loss")
    plt.plot(epochs, val_loss, "b", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def get_training_data(data_path, poi_path):
    """_summary_

    Args:
        data_path (str): Path to data input file.
        poi_path (str): Path to Point of Interest (POI) file.

    Returns:
        tuple[int, Any, NDArray[float64], NDArray[float64], NDArray[float64], NDArray[Any]]: _description_
    """
    filename = os.path.split(data_path)[1]
    good = True

    with open(data_path) as f:
        # Read data file and load time, temperature, resonance, and dissipation data.
        csv_headers = next(f)

        csv_cols = (2, 4, 6, 7) if "Ambient" in csv_headers else (2, 3, 5, 6)

        data = np.loadtxt(data_path, delimiter=",", skiprows=1, usecols=csv_cols)
        relative_time = data[:, 0]
        temperature = data[:, 1]
        resonance_frequency = data[:, 2]
        dissipation = data[:, 3]

        # raw data
        xs = relative_time
        ys = dissipation

        # import POIs, convert indices to timestamps
        pois = np.loadtxt(poi_path).astype(int)
        # print(pois, xs[pois])
        # pois = PRECISION * xs[pois]
        pois = (pois / len(xs) * PPR).astype(int)
        # pois = 1000 * pois / len(xs)
        # print(pois)
        # pois = (len(xs)*pois / 1000).astype(int)
        # print(pois, len(xs), xs[-1])

        t_0p5 = (
            0  # if (xs[-1] < 0.5)    else next(x for x,t in enumerate(xs) if t > 0.5)
        )
        t_1p0 = (
            100  # if (len(xs) < 500)   else next(x for x,t in enumerate(xs) if t > 1.0)
        )

        # t_1p0, done = QtWidgets.QInputDialog.getDouble(None, 'Input Dialog', 'Confirm rough start index:', value=t_1p0)

        # new maths for resonance and dissipation (scaled)
        avg = np.average(resonance_frequency[t_0p5:t_1p0])
        ys = ys * avg / 2
        # ys_fit = ys_fit * avg / 2
        ys = ys - np.amin(ys)
        # ys_fit = ys_fit - np.amin(ys_fit)
        ys_freq = avg - resonance_frequency
        # ys_freq_fit = savgol_filter(ys_freq, smooth_factor, 1)
        ys_diff = ys_freq - ys
        # ys_diff_fit = savgol_filter(ys_diff, smooth_factor, 1)

        t_start = 0
        t_stop = -1

        xs = xs[t_start:t_stop]
        ys = ys[t_start:t_stop]
        ys_freq = ys_freq[t_start:t_stop]
        ys_diff = ys_diff[t_start:t_stop]

        lin_xs = np.linspace(xs[0], xs[-1], PPR)
        lin_ys = np.interp(lin_xs, xs, ys)
        lin_ys_freq = np.interp(lin_xs, xs, ys_freq)
        lin_ys_diff = np.interp(lin_xs, xs, ys_diff)

        return (
            int(good),
            lin_xs,
            lin_ys,
            lin_ys_freq,
            lin_ys_diff,
            pois,
        )  # relative_time, resonance_frequency, dissipation, difference_curve


def display_plots(plot_idx, plots_to_show, plot_names):
    """Utilitiy function to display a selection of plotted, valid data from the training set.


    Args:
        plot_idx (int): the index of the plot to display from plots_to_show.
        plots_to_show (list): a list of MatPlotLib plots to display.
        plot_names (list): the name of each plot to display.
    """
    if plot_idx in plots_to_show:
        plot_names.append(training_files[i][: -len("_poi.csv")])
        plt.figure()  # append to existing figure, not a new one
        plt.title(f"{training_files[i][:-len('_poi.csv')]}: good = {good}")
        plt.plot(time[pois[0] : pois[-1]], diss[pois[0] : pois[-1]], "r-")
        for y in range(len(pois)):
            plt.plot(time[pois[y]], diss[pois[y]], "bx")


if __name__ == "__main__":
    # Initialize training input data arrays and POI arrays.
    (
        training_time,
        training_name,
        training_diss,
        training_freq,
        training_diff,
        training_good,
        training_poi1,
        training_poi2,
        training_poi3,
        training_poi4,
        training_poi5,
        training_poi6,
    ) = [], [], [], [], [], [], [], [], [], [], [], []

    # The number of good/bad training samples.
    good_count, bad_count = 0, 0

    training_files = os.listdir(PATH_TO_TRAINING_DATA)

    # List of sample plots to display and the names of each respective plot.
    plots_to_show, plot_names = [], []

    # Collect a random sample of random plots to display.
    # Number of Training Files < Number of plots displayed < NUM_RAND_PLOTS_TO_DISP.
    while len(plots_to_show) < min(len(training_files), NUM_RAND_PLOTS_TO_DISP):
        rand_plt_idx = randint(0, len(training_files) - 1)
        rand_plt_idx -= 1 if rand_plt_idx % 2 == 1 else 0
        plots_to_show.append(
            rand_plt_idx
        ) if rand_plt_idx not in plots_to_show else None

    status("Beginning training data I/O.")

    for i in range(len(training_files)):
        data_path = os.path.join(PATH_TO_TRAINING_DATA, training_files[i])

        # Ignore training files with suffixes of:
        # '_poi.csv', '_lower.csv', '_upper.csv', or ending in '.xml'
        if (
            (not os.path.isfile(data_path))
            or ("_poi.csv" in data_path)
            or (data_path.endswith(".xml"))
            or ("_lower.csv" in data_path)
            or ("_upper.csv" in data_path)
        ):
            continue

        # Update any .csv files suffixes with '_poi.csv' modifier.
        poi_path = data_path.replace(".csv", "_poi.csv")

        filename = os.path.split(data_path)[1]
        if "good" in filename.lower() != "bad" in filename.lower():
            continue

        # Get training data and set of POIs.
        good, time, diss, freq, diff, pois = get_training_data(data_path, poi_path)
        (poi1, poi2, poi3, poi4, poi5, poi6) = pois
        # display_plots(i, plots_to_show, plot_names)

        # Add the i-th file's data to the training set.
        training_name.append(training_files[i][: -len("_poi.csv")])
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

    status("Training data initializied.")
    echo(input(">>> Press any key to proceed..."))

    # Arrays for the list of initialization points, followed by 3 arrays for each of the 3
    # possible inflection points in the data.
    initialization, inflection_1, inflection_2, infelction_3 = [], [], [], []

    for i in range(len(training_poi1)):
        pois = (
            0,
            training_poi1[i],
            training_poi2[i],
            training_poi3[i],
            training_poi4[i],
            training_poi5[i],
            training_poi6[i],
        )
        total_size = pois[6] - pois[1]
        _init_pct = (pois[2] - pois[1]) / total_size * 100
        blip1_pct = (pois[4] - pois[1]) / total_size * 100
        blip2_pct = (pois[5] - pois[1]) / total_size * 100
        blip3_pct = (pois[6] - pois[1]) / total_size * 100
        initialization.append(_init_pct)
        inflection_1.append(blip1_pct)
        inflection_2.append(blip2_pct)
        infelction_3.append(blip3_pct)

    info(
        f"Init Avg: {np.average(initialization)} Inf_1 Avg: {np.average(inflection_1)} Inf_2 Avg: {np.average(inflection_2)} Inf_3 Avg: {np.average(infelction_3)}"
    )

    # Reformat training data as type numpy.array for model.
    time_features = np.array(training_time)
    diss_features = np.array(training_diss)
    freq_features = np.array(training_freq)
    diff_features = np.array(training_diff)
    data_label0 = np.array(training_good).astype(int)
    data_label1 = np.array(training_poi1).astype(int)
    data_label2 = np.array(training_poi2).astype(int)
    data_label3 = np.array(training_poi3).astype(int)
    data_label4 = np.array(training_poi4).astype(int)
    data_label5 = np.array(training_poi5).astype(int)
    data_label6 = np.array(training_poi6).astype(int)

    # Data labels for model.
    data_labels = (
        data_label0,
        data_label1,
        data_label2,
        data_label3,
        data_label4,
        data_label5,
        data_label6,
    )

    # Normalize for optimal training
    time_normalize = layers.Normalization()
    time_normalize.adapt(time_features)

    diss_normalize = layers.Normalization()
    diss_normalize.adapt(diss_features)

    freq_normalize = layers.Normalization()
    freq_normalize.adapt(freq_features)

    diff_normalize = layers.Normalization()
    diff_normalize.adapt(diff_features)

    # print how many bad vs good we've imported
    (unique, counts) = np.unique(training_good, return_counts=True)
    frequencies = np.asarray((unique, counts)).T

    info(f"{frequencies} unique counts frequency.")

    # MACHINE LEARNING PORTION

    NUM_DIMS = 128
    DROPOUTS = 0.1
    POOL_SIZE = 5

    # Define model layers.
    input_layer0 = layers.Input(name="time", shape=(PPR,))
    first_layer0 = time_normalize(input_layer0)
    second_layer0_0 = layers.Dense(NUM_DIMS, activation="relu")(first_layer0)
    third_layer0_0 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer0_0, axis=-1)
    )
    fourth_layer0_0 = layers.Dropout(DROPOUTS)(third_layer0_0)

    input_layer1 = layers.Input(name="diss", shape=(PPR,))
    first_layer1 = diss_normalize(input_layer1)
    second_layer0_1 = layers.Dense(NUM_DIMS, activation="relu")(first_layer1)
    third_layer0_1 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer0_1, axis=-1)
    )
    fourth_layer0_1 = layers.Dropout(DROPOUTS)(third_layer0_1)

    input_layer2 = layers.Input(name="freq", shape=(PPR,))
    first_layer2 = freq_normalize(input_layer2)
    second_layer0_2 = layers.Dense(NUM_DIMS, activation="relu")(first_layer2)
    third_layer0_2 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer0_2, axis=-1)
    )
    fourth_layer0_2 = layers.Dropout(DROPOUTS)(third_layer0_2)

    input_layer3 = layers.Input(name="diff", shape=(PPR,))
    first_layer3 = diff_normalize(input_layer3)
    second_layer0_3 = layers.Dense(NUM_DIMS, activation="relu")(first_layer3)
    third_layer0_3 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer0_3, axis=-1)
    )
    fourth_layer0_3 = layers.Dropout(DROPOUTS)(third_layer0_3)

    combined_output0 = layers.concatenate(
        [fourth_layer0_0, fourth_layer0_1, fourth_layer0_2, fourth_layer0_3]
    )
    # Y0 output will be fed from the fourth layer
    y0_output = layers.Dense(1, name="good")(combined_output0)
    # constrain_output0 = y0_output #Maximum()([0, y0_output])

    second_layer1_0 = layers.Dense(NUM_DIMS, activation="relu")(first_layer0)
    third_layer1_0 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer1_0, axis=-1)
    )
    fourth_layer1_0 = layers.Dropout(DROPOUTS)(third_layer1_0)

    second_layer1_1 = layers.Dense(NUM_DIMS, activation="relu")(first_layer1)
    third_layer1_1 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer1_1, axis=-1)
    )
    fourth_layer1_1 = layers.Dropout(DROPOUTS)(third_layer1_1)

    second_layer1_2 = layers.Dense(NUM_DIMS, activation="relu")(first_layer2)
    third_layer1_2 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer1_2, axis=-1)
    )
    fourth_layer1_2 = layers.Dropout(DROPOUTS)(third_layer1_2)

    second_layer1_3 = layers.Dense(NUM_DIMS, activation="relu")(first_layer3)
    third_layer1_3 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer1_3, axis=-1)
    )
    fourth_layer1_3 = layers.Dropout(DROPOUTS)(third_layer1_3)

    combined_output1 = layers.concatenate(
        [fourth_layer1_0, fourth_layer1_1, fourth_layer1_2, fourth_layer1_3]
    )

    # Y1 output will be fed from the fourth layer
    y1_output = layers.Dense(1, name="start")(combined_output1)
    # constrain_output1 = Maximum(name='start')([constrain_output0, y1_output])

    second_layer2_0 = layers.Dense(NUM_DIMS, activation="relu")(first_layer0)
    third_layer2_0 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer2_0, axis=-1)
    )
    fourth_layer2_0 = layers.Dropout(DROPOUTS)(third_layer2_0)

    second_layer2_1 = layers.Dense(NUM_DIMS, activation="relu")(first_layer1)
    third_layer2_1 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer2_1, axis=-1)
    )
    fourth_layer2_1 = layers.Dropout(DROPOUTS)(third_layer2_1)

    second_layer2_2 = layers.Dense(NUM_DIMS, activation="relu")(first_layer2)
    third_layer2_2 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer2_2, axis=-1)
    )
    fourth_layer2_2 = layers.Dropout(DROPOUTS)(third_layer2_2)

    second_layer2_3 = layers.Dense(NUM_DIMS, activation="relu")(first_layer3)
    third_layer2_3 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer2_3, axis=-1)
    )
    fourth_layer2_3 = layers.Dropout(DROPOUTS)(third_layer2_3)

    combined_output2 = layers.concatenate(
        [fourth_layer2_0, fourth_layer2_1, fourth_layer2_2, fourth_layer2_3]
    )
    # Y2 output will be fed from the fourth layer
    y2_output = layers.Dense(1, name="stop")(combined_output2)
    # constrain_output2 = Maximum(name='stop')([constrain_output1, y2_output])

    second_layer3_0 = layers.Dense(NUM_DIMS, activation="relu")(first_layer0)
    third_layer3_0 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer3_0, axis=-1)
    )
    fourth_layer3_0 = layers.Dropout(DROPOUTS)(third_layer3_0)

    second_layer3_1 = layers.Dense(NUM_DIMS, activation="relu")(first_layer1)
    third_layer3_1 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer3_1, axis=-1)
    )
    fourth_layer3_1 = layers.Dropout(DROPOUTS)(third_layer3_1)

    second_layer3_2 = layers.Dense(NUM_DIMS, activation="relu")(first_layer2)
    third_layer3_2 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer3_2, axis=-1)
    )
    fourth_layer3_2 = layers.Dropout(DROPOUTS)(third_layer3_2)

    second_layer3_3 = layers.Dense(NUM_DIMS, activation="relu")(first_layer3)
    third_layer3_3 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer3_3, axis=-1)
    )
    fourth_layer3_3 = layers.Dropout(DROPOUTS)(third_layer3_3)

    combined_output3 = layers.concatenate(
        [fourth_layer3_0, fourth_layer3_1, fourth_layer3_2, fourth_layer3_3]
    )
    # Y3 output will be fed from the fourth layer
    y3_output = layers.Dense(1, name="post")(combined_output3)
    # constrain_output3 = Maximum(name='post')([constrain_output2, y3_output])

    second_layer4_0 = layers.Dense(NUM_DIMS, activation="relu")(first_layer0)
    third_layer4_0 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer4_0, axis=-1)
    )
    fourth_layer4_0 = layers.Dropout(DROPOUTS)(third_layer4_0)

    second_layer4_1 = layers.Dense(NUM_DIMS, activation="relu")(first_layer1)
    third_layer4_1 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer4_1, axis=-1)
    )
    fourth_layer4_1 = layers.Dropout(DROPOUTS)(third_layer4_1)

    second_layer4_2 = layers.Dense(NUM_DIMS, activation="relu")(first_layer2)
    third_layer4_2 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer4_2, axis=-1)
    )
    fourth_layer4_2 = layers.Dropout(DROPOUTS)(third_layer4_2)

    second_layer4_3 = layers.Dense(NUM_DIMS, activation="relu")(first_layer3)
    third_layer4_3 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer4_3, axis=-1)
    )
    fourth_layer4_3 = layers.Dropout(DROPOUTS)(third_layer4_3)

    combined_output4 = layers.concatenate(
        [fourth_layer4_0, fourth_layer4_1, fourth_layer4_2, fourth_layer4_3]
    )
    # Y4 output will be fed from the fourth layer
    y4_output = layers.Dense(1, name="blip1")(combined_output4)
    # constrain_output4 = Maximum(name='blip1')([constrain_output3, y4_output])

    second_layer5_0 = layers.Dense(NUM_DIMS, activation="relu")(first_layer0)
    third_layer5_0 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer5_0, axis=-1)
    )
    fourth_layer5_0 = layers.Dropout(DROPOUTS)(third_layer5_0)

    second_layer5_1 = layers.Dense(NUM_DIMS, activation="relu")(first_layer1)
    third_layer5_1 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer5_1, axis=-1)
    )
    fourth_layer5_1 = layers.Dropout(DROPOUTS)(third_layer5_1)

    second_layer5_2 = layers.Dense(NUM_DIMS, activation="relu")(first_layer2)
    third_layer5_2 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer5_2, axis=-1)
    )
    fourth_layer5_2 = layers.Dropout(DROPOUTS)(third_layer5_2)

    second_layer5_3 = layers.Dense(NUM_DIMS, activation="relu")(first_layer3)
    third_layer5_3 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer5_3, axis=-1)
    )
    fourth_layer5_3 = layers.Dropout(DROPOUTS)(third_layer5_3)

    combined_output5 = layers.concatenate(
        [fourth_layer5_0, fourth_layer5_1, fourth_layer5_2, fourth_layer5_3]
    )
    # Y5 output will be fed from the fourth layer
    y5_output = layers.Dense(1, name="blip2")(combined_output5)
    # constrain_output5 = Maximum(name='blip2')([constrain_output4, y5_output])

    second_layer6_0 = layers.Dense(NUM_DIMS, activation="relu")(first_layer0)
    third_layer6_0 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer6_0, axis=-1)
    )
    fourth_layer6_0 = layers.Dropout(DROPOUTS)(third_layer6_0)

    second_layer6_1 = layers.Dense(NUM_DIMS, activation="relu")(first_layer1)
    third_layer6_1 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer6_1, axis=-1)
    )
    fourth_layer6_1 = layers.Dropout(DROPOUTS)(third_layer6_1)

    second_layer6_2 = layers.Dense(NUM_DIMS, activation="relu")(first_layer2)
    third_layer6_2 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer6_2, axis=-1)
    )
    fourth_layer6_2 = layers.Dropout(DROPOUTS)(third_layer6_2)

    second_layer6_3 = layers.Dense(NUM_DIMS, activation="relu")(first_layer3)
    third_layer6_3 = layers.MaxPooling1D(pool_size=POOL_SIZE)(
        tf.expand_dims(second_layer6_3, axis=-1)
    )
    fourth_layer6_3 = layers.Dropout(DROPOUTS)(third_layer6_3)

    combined_output6 = layers.concatenate(
        [fourth_layer6_0, fourth_layer6_1, fourth_layer6_2, fourth_layer6_3]
    )
    # Y6 output will be fed from the fourth layer
    y6_output = layers.Dense(1, name="blip3")(combined_output6)
    # constrain_output6 = Maximum(name='blip3')([constrain_output5, y6_output])
    # constrain_output6 = Minimum()([constrain_output6_0, HOW_MANY_DATA_POINTS_PER_ROW*PRECISION])

    # Define the model with the input layer
    # and a list of output layers
    csv_model = tf.keras.Model(
        inputs=[input_layer0, input_layer1, input_layer2, input_layer3],
        outputs=[
            y0_output,
            y1_output,
            y2_output,
            y3_output,
            y4_output,
            y5_output,
            y6_output,
        ],
    )

    # Specify the optimizer, and compile the model with loss functions for both outputs
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    csv_model.compile(
        optimizer=optimizer,
        loss={
            "good": tf.keras.losses.BinaryCrossentropy(),
            "start": tf.keras.losses.MeanAbsoluteError(),
            "stop": tf.keras.losses.MeanAbsoluteError(),
            "post": tf.keras.losses.MeanAbsoluteError(),
            "blip1": tf.keras.losses.MeanAbsoluteError(),
            "blip2": tf.keras.losses.MeanAbsoluteError(),
            "blip3": tf.keras.losses.MeanAbsoluteError(),
        },
        metrics={
            "good": tf.keras.metrics.BinaryCrossentropy(),
            "start": tf.keras.metrics.MeanAbsoluteError(),
            "stop": tf.keras.metrics.MeanAbsoluteError(),
            "post": tf.keras.metrics.MeanAbsoluteError(),
            "blip1": tf.keras.metrics.MeanAbsoluteError(),
            "blip2": tf.keras.metrics.MeanAbsoluteError(),
            "blip3": tf.keras.metrics.MeanAbsoluteError(),
        },
    )

    # Split data into training and test sets.
    (
        x_train0,
        x_test0,
        x_train1,
        x_test1,
        x_train2,
        x_test2,
        x_train3,
        x_test3,
        y_train0,
        y_test0,
        y_train1,
        y_test1,
        y_train2,
        y_test2,
        y_train3,
        y_test3,
        y_train4,
        y_test4,
        y_train5,
        y_test5,
        y_train6,
        y_test6,
    ) = sklearn.model_selection.train_test_split(
        time_features,
        diss_features,
        freq_features,
        diff_features,
        data_label0,
        data_label1,
        data_label2,
        data_label3,
        data_label4,
        data_label5,
        data_label6,
        test_size=0.2,
    )

    history = csv_model.fit(
        [x_train0, x_train1, x_train2, x_train3],
        [y_train0, y_train1, y_train2, y_train3, y_train4, y_train5, y_train6],
        validation_data=(
            [x_test0, x_test1, x_test2, x_test3],
            [
                y_test0,
                y_test1,
                y_test2,
                y_test3,
                y_test4,
                y_test5,
                y_test6,
            ],
        ),
        epochs=30,
        batch_size=(len(x_train0) if len(x_train0) < 100 else 100),
        verbose=1,
    )

    # save and reload the model - uncomment following if you want to save the model and reload it
    # MODEL_PATH = 'SavedModel.tf'
    # csv_model.save(MODEL_PATH)
    # reloaded_model = tf.keras.models.load_model(MODEL_PATH)

    csv_model.evaluate(
        [x_test0, x_test1, x_test2, x_test3],
        [
            y_test0,
            y_test1,
            y_test2,
            y_test3,
            y_test4,
            y_test5,
            y_test6,
        ],
    )

    history_dict = history.history
    history_dict.keys()

    loss = history.history["loss"]
    test_loss = history.history["val_loss"]

    epochs = range(1, len(loss) + 1)
    plot_loss(history)
    print("Press any key to close . . .")
    input()

    predictions = csv_model.predict([x_test0, x_test1, x_test2, x_test3])
    # Convert probabilities to actual decisions based on a threshold (e.g., 0.5

    # Print decisions
    print("Decisions:", predictions)
    print("Press any key to close . . .")
    input()

    # correct = []
    # for input_diss in training_diss:
    #     i = len(correct)

    #     predictions = csv_model(
    #         [training_time[i], input_diss, training_freq[i], training_diff[i]]
    #     )
    #     val0 = np.round(predictions[0].numpy()[0][0]).astype(int)
    #     val1 = np.round(predictions[1].numpy()[0][0]).astype(int)
    #     val2 = np.round(predictions[2].numpy()[0][0]).astype(int)
    #     val3 = np.round(predictions[3].numpy()[0][0]).astype(int)
    #     val4 = np.round(predictions[4].numpy()[0][0]).astype(int)
    #     val5 = np.round(predictions[5].numpy()[0][0]).astype(int)
    #     val6 = np.round(predictions[6].numpy()[0][0]).astype(int)
    #     vals = (val0, val1, val2, val3, val4, val5, val6)
    #     pois = (
    #         training_poi1[i],
    #         training_poi2[i],
    #         training_poi3[i],
    #         training_poi4[i],
    #         training_poi5[i],
    #         training_poi6[i],
    #     )

    #     valid = True
    #     # valid = (val0 < val1 < val2 < val3 < val4 < val5 < val6)
    #     valid = valid and val0 == good

    #     diffs = []
    #     for i in range(len(pois)):
    #         diff = int(abs(vals[i + 1] - pois[i]))
    #         diffs.append(diff)
    #         if diff > PRECISION:
    #             valid = False

    #     correct.append(valid)

    #     print(training_name[i], valid)
    #     pois = tuple(int(item) for item in pois)
    #     print(pois, vals, diffs)

    #     # num_pts = len(training_time[i])
    #     # idxs_predict = tuple(max(0, min(num_pts-1, int(num_pts*item/1000))) for item in vals)
    #     # idxs_actual = tuple(max(0, min(num_pts-1, int(num_pts*item/1000))) for item in pois)
    #     # print("ids: ", idxs, num_pts)

    #     if training_name[i] in plot_names:
    #         plt.figure()
    #         plt.title(f"{training_name[i]}: good = {valid}")
    #         plt.plot(training_time[i], training_diss[i], "r-")
    #         for y in range(len(pois)):
    #             # # try:
    #             # #   temp1 = next(x for x,t in enumerate(training_time[i]) if t > vals[y+1]/PRECISION)
    #             # # except:
    #             # #   temp1 = 0
    #             # # temp2 = next(x for x,t in enumerate(training_time[i]) if t > pois[y]/PRECISION)
    #             # plt.plot(vals[y+1]/PRECISION, training_diff[i][temp1], 'bx')
    #             # plt.plot(pois[y]/PRECISION, training_diff[i][temp2], 'gx')
    #             idx = max(0, min(PPR - 1, vals[y + 1]))
    #             plt.plot(training_time[i][idx], training_diss[i][idx], "bx")
    #             plt.plot(training_time[i][pois[y]], training_diss[i][pois[y]], "gx")

    # (unique, counts) = np.unique(correct, return_counts=True)
    # frequencies = np.asarray((unique, counts)).T

    # print("Results")
    # print(frequencies)
