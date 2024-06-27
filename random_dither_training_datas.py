import os
import pandas as pd
import numpy as np
from numpy.random import RandomState, MT19937

DEBUG_SHOW_PLOTS = False
DIFF_FACTOR = 1.5 # TODO: dynamic

MIN_START_OFFSET = 3.0
MIN_START_DITHER = 0.5

MIN_STOP_OFFSET = 3.0
MIN_STOP_DITHER = 0.5

MAX_TIME_OFFSET = 0.0
MAX_TIME_DITHER = 1.0

MAX_FREQ_OFFSET = 10e3
MAX_FREQ_DITHER = 2.0

MAX_DISS_OFFSET = 50e-6
MAX_DISS_DITHER = 2.0

poi_files = []
xml_files = []
csv_files = []

num_files_per_run = 3

def find_files(directory):
    for root, dirs, files in os.walk(directory):
        file_poi = None
        file_xml = None
        file_csv = None
        if os.path.basename(dithered_data_path) in root:
            continue # don't dither already dithered data
        for file in files:
            if file.endswith("_poi.csv"):
                file_poi = os.path.join(root, file)
            if file.endswith(".xml"):
                file_xml = os.path.join(root, file)
            if file.endswith("_3rd.csv"):
                file_csv = os.path.join(root, file)
        if all([file_poi, file_xml, file_csv]):
            poi_files.append(file_poi)
            xml_files.append(file_xml)
            csv_files.append(file_csv)
    return len(csv_files)

def normalize(arr, t_min = 0, t_max = 1):
        norm_arr = arr.copy()
        try:
            diff = t_max - t_min
            diff_arr = max(arr) - min(arr)
            if diff_arr == 0:
                diff_arr = 1
            norm_arr -= min(arr)
            norm_arr *= diff
            norm_arr /= diff_arr
            norm_arr += t_min
        except Exception as e:
            print("ERROR:" + str(e))
        return norm_arr

def dither_file(idx):
    file_poi = poi_files[idx]
    file_xml = xml_files[idx]
    file_csv = csv_files[idx]

    poi_data = np.loadtxt(file_poi, dtype=int)
    poi_orig = poi_data.copy()
    csv_data = pd.read_csv(file_csv)

    rs = RandomState(MT19937()) # reseed random
    
    headers = csv_data.columns
    date, time, rel_time, ambient, temp, raw, freq, diss = headers
    dataset = {}

    for col in headers:
        dataset[col] = list(csv_data[col])

    total_time = dataset[rel_time][-1]

    # 1st step: handle stop dithering
    stop_time = dataset[rel_time][poi_data[-1]]
    if total_time - stop_time > MIN_STOP_OFFSET + MIN_STOP_DITHER:

        maxval = total_time - stop_time - MIN_STOP_OFFSET
        dither = rs.random() * maxval
        remove = next(x for x, y in enumerate(dataset[rel_time]) if y > total_time - dither)

        # apply stop time dithering to all datasets:
        for col in headers:                         # truncate starts from all datasets
            dataset[col] = dataset[col][:remove]
        
    # 2nd step: handle start dithering
    start_time = dataset[rel_time][poi_data[0]]
    if start_time > MIN_START_OFFSET + MIN_START_DITHER:

        maxval = start_time - MIN_START_OFFSET
        dither = rs.random() * maxval
        remove = next(x for x, y in enumerate(dataset[rel_time]) if y > dither)

        # apply start time dithering to all datasets:
        for i in range(len(poi_data)):              # adjust POI offsets
            poi_data[i] -= remove
        for col in headers:                         # truncate starts from all datasets
            dataset[col] = dataset[col][remove:]
        for i in range(len(dataset[rel_time])):   # re-align relative times to zero
            dataset[rel_time][i] = np.round(dataset[rel_time][i] - dither, 4)

    # 3rd step: handle relative time dithering
    maxval = MAX_TIME_OFFSET
    offset = rs.random() * maxval # forward offset only for time
    for i in range(len(dataset[rel_time])):
        if not i == len(dataset[rel_time])-1: # skipping last value
            diff = (dataset[rel_time][i+1] - dataset[rel_time][i]) / 2
        dither = rs.random() * diff # up to next value
        dataset[rel_time][i] = np.round(dataset[rel_time][i] + offset + dither, 4)

    # 4th step: handle frequency dithering
    maxval = MAX_FREQ_OFFSET
    offset = rs.random() * maxval - maxval / 2
    diffs = np.diff(dataset[freq])
    stdev = np.std(diffs) * MAX_FREQ_DITHER
    for i in range(len(dataset[freq])):
        dither = rs.random() * stdev
        dataset[freq][i] = np.round(dataset[freq][i] + offset + dither, 0)

    # 5th step: handle dissipation dithering
    maxval = MAX_DISS_OFFSET
    offset = rs.random() * maxval - maxval / 2
    diffs = np.diff(dataset[diss])
    stdev = np.std(diffs) * MAX_DISS_DITHER
    for i in range(len(dataset[diss])):
        dither = rs.random() * stdev
        dataset[diss][i] = dataset[diss][i] + offset + dither

    # write it all back to files
    run_identifier = os.path.basename(os.path.dirname(file_csv)) + "-{}"
    save_to_folder = os.path.join(dithered_data_path, run_identifier)
    i = 1
    while os.path.isdir(save_to_folder.format(i)):
        i += 1
    save_to_folder = save_to_folder.format(i)
    os.makedirs(save_to_folder)

    dithered_poi_file = os.path.join(save_to_folder, "Dithered_" + os.path.basename(file_poi))
    np.savetxt(dithered_poi_file, poi_data, fmt='%i')

    dithered_xml_file = os.path.join(save_to_folder, "Dithered_" + os.path.basename(file_xml))
    with open(file_xml, "r") as r:
        with open(dithered_xml_file, "w") as w:
            w.write(r.read()) # copy entire file, unchanged (for simplicity since it's not looked at for training)

    dithered_csv_file = os.path.join(save_to_folder, "Dithered_" + os.path.basename(file_csv))
    with open(dithered_csv_file, "w") as f:
        f.write(",".join(headers) + "\n")
        for line in range(len(dataset[rel_time])):
            data = []
            for col in headers:
                data.append(str(dataset[col][line]))
            f.write(",".join(data) + "\n")
            f.flush()

    print(f"Dithered run {idx} to folder {os.path.basename(save_to_folder)}")

    if DEBUG_SHOW_PLOTS:
        import matplotlib.pyplot as plt
        import statistics as stats
        fig, (before, after) = plt.subplots(1, 2)
        fig.suptitle(os.path.basename(dithered_csv_file))

        avg_resonance_frequency = stats.mode(csv_data[freq][:poi_orig[0]])
        dissipation_base = stats.mode(csv_data[diss][:poi_orig[0]])
        before_diss = (csv_data[diss] - dissipation_base) * avg_resonance_frequency / 2
        before_freq = avg_resonance_frequency - csv_data[freq]
        before_diff = before_freq - DIFF_FACTOR * before_diss

        before.set_title("Original")
        before.plot(csv_data[rel_time], before_diss, color="red")
        before.plot(csv_data[rel_time], before_freq, color="green")
        before.plot(csv_data[rel_time], before_diff, color="blue")
        for pt in poi_orig:
            xs = csv_data[rel_time][pt]
            before.axvline(xs, color="black")

        avg_resonance_frequency = stats.mode(dataset[freq][:poi_data[0]])
        dissipation_base = stats.mode(dataset[diss][:poi_data[0]])
        after_diss = (dataset[diss] - dissipation_base) * avg_resonance_frequency / 2
        after_freq = avg_resonance_frequency - dataset[freq]
        after_diff = after_freq - DIFF_FACTOR * after_diss

        after.set_title("Dithered")
        after.plot(dataset[rel_time], after_diss, color="red")
        after.plot(dataset[rel_time], after_freq, color="green")
        after.plot(dataset[rel_time], after_diff, color="blue")
        for pt in poi_data:
            xs = dataset[rel_time][pt]
            after.axvline(xs, color="black")
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()


training_data_path = r"C:\Users\Alexander J. Ross\Documents\QATCH Work\ML\training_data"
dithered_data_path = os.path.join(training_data_path, "Dithered_Data")

count = find_files(training_data_path)
print(f"Found {count} files to dither...")
for i in range(count):
    dither_file(i)
