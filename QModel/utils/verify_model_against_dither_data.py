from io import BytesIO
import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import random
from datetime import datetime
from tqdm import tqdm
# from scipy.optimize import curve_fit

MAX_TRUE_DIFF_ERROR = 5
EXPONENTIAL_DIFF_ERROR = False
STD_BIN_SIZE = 1
MIN_NUM_BINS = 10
MAX_NUM_BINS = 100
STOP_AFTER = None
F1_SCORE_SPLIT_AT = 3
USE_MODELDATA = False
HAS_GLOBAL_PREDICT = True

if USE_MODELDATA:
    from ModelData import ModelData
else:
    from QModel.QModel import QModelPredict
    
class VerifyQModel():

    def __init__(self):
        self.poi_files = []
        self.xml_files = []
        self.csv_files = []

        self.query_time_min = np.nan
        self.query_time_max = np.nan
        self.query_time_avg = np.nan # avg = sum / num (calc'd @ end)
        self.query_time_sum = 0
        self.query_time_num = 0

        self.poi_order_good = 0
        self.poi_order_bad = 0
        self.poi_index_bad = [0,0,0,0,0,0]

    def find_files(self, directory):
        for root, dirs, files in os.walk(directory):
            file_poi = None
            file_xml = None
            file_csv = None
            for file in files:
                if file.endswith("_poi.csv"):
                    file_poi = os.path.join(root, file)
                if file.endswith(".xml"):
                    file_xml = os.path.join(root, file)
                if file.endswith("_3rd.csv"):
                    file_csv = os.path.join(root, file)
            if all([file_poi, file_xml, file_csv]):
                self.poi_files.append(file_poi)
                self.xml_files.append(file_xml)
                self.csv_files.append(file_csv)
        return len(self.csv_files)

    def predict(self, data_path):
        query_time_start = datetime.now()
        if USE_MODELDATA:
            calculated_pois = ModelData().IdentifyPoints(data_path)
            peaks = []
            bounds = []
            for poi in calculated_pois:
                if isinstance(poi, list):
                    peaks.append(poi[0][0])
                    bounds.append(100 * (1 - poi[0][1]))
                elif isinstance(poi, int):
                    peaks.append(poi)
                    bounds.append(0)
                else:
                    print("ERROR: Unexpected type:", type(poi))
                    peaks.append(-1)
                    bounds.append(-1)
            with open(data_path, 'r') as f:
                length = len(f.readlines())-1
        elif HAS_GLOBAL_PREDICT:
            print()
            model_paths = []
            for i in np.linspace(1, 6, 6).astype(int):
                model_paths.append(f"QModel/SavedModels/QModel_{i}.json")
            qpredictor = QModelPredict(model_paths[0], 
                                       model_paths[1], 
                                       model_paths[2], 
                                       model_paths[3], 
                                       model_paths[4], 
                                       model_paths[5])
            peaks = qpredictor.predict(data_path)
            bounds = [1, 1, 1, 1, 1, 1]
            with open(data_path, 'r') as f:
                length = len(f.readlines())-1
        else:
            print() # new line for tqdm
            qpredictors = []
            for i in np.linspace(1, 6, 6).astype(int):
                qpredictors.append(
                    QModelPredict(
                        model_path=f"QModel/SavedModels/QModel_{i}.json"
                    )
                )
            qpredictions = []
            with open(data_path, 'r') as f:
                length = len(f.readlines())-1
                for qpredictor in qpredictors:
                    f.seek(0) # beginning of file
                    try:
                        qpredictions.append(
                            qpredictor.predict(f)[1]
                        )
                    except:
                        qpredictions.append(None) # exception while predicting point
            peaks = []
            bounds = []
            for i, qprediction in enumerate(qpredictions):
                print(f"Prediction #{i+1}:", qprediction)
                if isinstance(qprediction, list):
                    these_peaks = []
                    these_bounds = []
                    for qpeak in qprediction:
                        these_peaks.append(np.average(qpeak).astype(int))
                        bound_diff = np.diff(qpeak)[0]
                        these_bounds.append(bound_diff)
                    if len(these_peaks) == 1:
                        peaks.append(these_peaks[0])
                        bounds.append(these_bounds[0])
                    else:
                        biggest_bound = 0
                        biggest_index = 0
                        for index, size in enumerate(these_bounds):
                            if size > biggest_bound:
                                biggest_bound = size
                                biggest_index = index
                        peaks.append(these_peaks[biggest_index])
                        bounds.append(these_bounds[biggest_index])
                        # peaks.append(np.average(these_peaks).astype(int))
                        # bounds.append(np.sum(these_bounds))
                elif isinstance(qprediction, int):
                    peaks.append(qprediction)
                    bounds.append(0)
                elif qprediction is None:
                    print("ERROR: Model returned an exception predicting point")
                    peaks.append(-1)
                    bounds.append(100)
                else:
                    print("ERROR: Model returned an unexpected type predicting point")
                    peaks.append(-1)
                    bounds.append(100)
        query_time_stop = datetime.now()
        query_time_diff = (query_time_stop - query_time_start).total_seconds()
        if np.isnan(self.query_time_min) or self.query_time_min > query_time_diff:
            self.query_time_min = query_time_diff
        if np.isnan(self.query_time_max) or self.query_time_max < query_time_diff:
            self.query_time_max = query_time_diff
        self.query_time_sum += query_time_diff
        self.query_time_num += 1
        sorted_peaks = peaks.copy()
        sorted_peaks.sort()
        unique_peaks = np.unique(peaks).tolist() #.sort()
        if peaks == sorted_peaks and len(peaks) == len(unique_peaks):
            self.poi_order_good += 1
        elif peaks[-1] != -1:
            self.poi_order_bad += 1
            print("\n", os.path.basename(data_path), "- Bad sequence order given:", peaks)
            for p in range(len(sorted_peaks)):
                if sorted_peaks[p] != peaks[p]:
                    self.poi_index_bad[p] += 1
            if len(peaks) != len(unique_peaks):
                print("Duplicate POIs detected!")
                for x in range(len(peaks)):
                    for y in range(len(peaks)):
                        if x == y: continue
                        if peaks[x] == peaks[y]:
                            self.poi_index_bad[x] += 1
                            # self.poi_index_bad[y] += 1
        else:
            self.poi_order_good += 1 # treat as 'good'
            print("\n", os.path.basename(data_path), "- Short run, last point was -1")
        # peaks = np.sort(peaks)
        if isinstance(peaks, list) and len(peaks) == 6:
            return peaks, bounds, length
        else:
            try:
                print(f"Run {os.path.basename(data_path)} returned invalid result ({len(peaks)}): {peaks}")
            except:
                print("Things went so wrong, I can't even begin to explain it to you...")
            return [-1], [-1], -1 # invalid result

    def benchmark_model(self):
        training_data_path = r"C:\Users\Alexander J. Ross\Documents\QATCH Work\ML\training_data"
        dithered_data_path = os.path.join(training_data_path, "Dithered_Data")
        num_runs = self.find_files(dithered_data_path)
        random_order = np.arange(0, num_runs, 1)
        random.shuffle(random_order)

        run_count = 0
        histo_difference = []
        histo_diff = [] #np.random.randn(num_runs, 6) # np.array([])
        histo_bounds = [] #np.array([])
        histo_error = [] #np.array([])
        histo_confidence = [] #np.array([])
        histo_accuracy = []
        histo_precision = []
        tp, fp, fn = 0, 0, 0
        tp_later, fp_later, fn_later = 0, 0, 0
        desc_text = "Verify ModelData" if USE_MODELDATA else "Verify QModel"
        for run in tqdm(random_order, desc=desc_text, unit="runs"):
            run_count += 1

            data_path = self.csv_files[run]
            poi_actual, poi_bounds, poi_length = self.predict(data_path)
            poi_path = self.poi_files[run]
            poi_expected = np.loadtxt(poi_path, dtype=int)

            poi_actual = np.array(poi_actual)
            poi_expected = np.array(poi_expected)
            poi_diff = poi_actual - poi_expected
            poi_bounds = np.array(poi_bounds)
            pct_diff = 100 * (poi_actual - poi_expected) / poi_length
            pct_bounds = 100 * poi_bounds / poi_length
            poi_error = 100 * poi_diff / poi_length # (poi_expected[-1] - poi_expected[0])
            poi_confidence = 100 * (poi_length - poi_bounds) / poi_length
            poi_accuracy = 100 - np.abs(
                (poi_expected - poi_actual) / poi_expected * 100
                )
            poi_precision = 100 - pct_bounds
            for x in range(6):
                diff_error = MAX_TRUE_DIFF_ERROR
                if EXPONENTIAL_DIFF_ERROR:
                    diff_error ^= (x+3) # START AT 2^3 = 8 WITH 1ST POI, THEN 16, 32, 64, 128, 256
                if poi_diff[x] < diff_error:
                    if x < F1_SCORE_SPLIT_AT:
                        tp += 1
                    else:
                        tp_later += 1
                elif poi_diff[x] > 0:
                    if x < F1_SCORE_SPLIT_AT:
                        fp += 1
                    else:
                        fp_later += 1
                else:
                    if x < F1_SCORE_SPLIT_AT:
                        fn += 1
                    else:
                        fn_later += 1
                if poi_accuracy[x] < 0:
                    poi_accuracy[x] = 0
                if poi_precision[x] < 0:
                    poi_precision[x] = 0

            histo_difference.append(poi_diff)
            histo_diff.append(pct_diff)
            histo_bounds.append(pct_bounds)
            histo_error.append(poi_error)
            histo_confidence.append(poi_confidence)
            histo_accuracy.append(poi_accuracy)
            histo_precision.append(poi_precision)

            # print(f"{run_count}/{num_runs}: ({run}) {os.path.basename(data_path)}")
            print("Expected:  ", poi_expected.tolist())
            print("Predicted: ", poi_actual.tolist())
            print("Difference:", poi_diff)
            # print("Error:     ", poi_error.tolist())
            # print("Confidence:", poi_confidence.tolist())
            # print("Accuracy:  ", poi_accuracy.tolist())
            # print("Precision: ", poi_precision.tolist())

            # if poi_actual == -1:
            #     continue
            if STOP_AFTER != None and run_count >= STOP_AFTER:
                break

            SHOW_SINGLE_PREDICTION_PLOTS = False
            if SHOW_SINGLE_PREDICTION_PLOTS:
                plt.figure(figsize=(10,10))
                plt.suptitle(f"{run_count}/{num_runs}: ({run}) {os.path.basename(data_path)}")
                for x in range(len(poi_expected)):
                    plt.axvline(poi_expected[x], color="black")
                for y in range(len(poi_actual)):
                    plt.plot(poi_actual[y], y+1, color="red", marker="*")
                plt.show()

        # cast 'list' to 'numpy.ndarray' for 'hist()' to be happy
        histo_diff = np.array(histo_diff)
        histo_bounds = np.array(histo_bounds)
        histo_error = np.array(histo_error)
        histo_confidence = np.array(histo_confidence)
        histo_accuracy = np.array(histo_accuracy)
        histo_precision = np.array(histo_precision)

        fig10, ((ax11, ax12), (ax13, ax14)) = plt.subplots(nrows=2, ncols=2)

        # colors = ['red', 'orange', 'yellow', 'green', 'blue', 'pink']
        # labels = ['start', 'eof', 'post', 'ch1', 'ch2', 'ch3']

        n_bins = min(MAX_NUM_BINS, max(MIN_NUM_BINS, int((histo_diff.max() - histo_diff.min()) / STD_BIN_SIZE)))
        ax11.hist(histo_diff, n_bins, density=True, histtype='bar', stacked=True)
        ax11.set_title('Overall Percent Difference')

        n_bins = min(MAX_NUM_BINS, max(MIN_NUM_BINS, int((histo_bounds.max() - histo_bounds.min()) / STD_BIN_SIZE)))
        ax12.hist(histo_bounds, n_bins, density=True, histtype='bar', stacked=True)
        ax12.set_title('Overall Percent Bounds Size')

        n_bins = min(MAX_NUM_BINS, max(MIN_NUM_BINS, int((histo_accuracy.max() - histo_accuracy.min()) / STD_BIN_SIZE)))
        ax13.hist(histo_accuracy, n_bins, density=True, histtype='bar', stacked=True)
        ax13.set_title('Overall Prediction Accuracy')

        n_bins = min(MAX_NUM_BINS, max(MIN_NUM_BINS, int((histo_precision.max() - histo_precision.min()) / STD_BIN_SIZE)))
        ax14.hist(histo_precision, n_bins, density=True, histtype='bar', stacked=True)
        ax14.set_title('Overall Prediction Precision')

        f1_score_early = 2*tp / (2*tp + fp + fn)
        f1_score_later = 2*tp_later / (2*tp_later + fp_later + fn_later)
        f1_score_all = 2*(tp + tp_later) / (2*(tp + tp_later) + (fp + fp_later) + (fn+fn_later))
        diff_error_text = f"{MAX_TRUE_DIFF_ERROR:0.0f}"
        if EXPONENTIAL_DIFF_ERROR:
            diff_error_text += "^x"
        fig10.suptitle(f"F1-Scores: First {F1_SCORE_SPLIT_AT} POIs = {f1_score_early:2.3f}, Last {6-F1_SCORE_SPLIT_AT} POIs = {f1_score_later:2.3f}, Overall = {f1_score_all:2.3f} (Max delta for True Positive = {diff_error_text} indices)"
                    #  f"First {F1_SCORE_SPLIT_AT} POIs F1-Score within {MAX_TRUE_DIFF_ERROR} indices: {f1_score_early}\n" + 
                    #  f"Last {6-F1_SCORE_SPLIT_AT} POIs F1-Score within {MAX_TRUE_DIFF_ERROR} indices: {f1_score_later}\n" +
                    #  f"Overall F1-score within {MAX_TRUE_DIFF_ERROR} indices: {f1_score_all}"
                    )

        fig10.show()

        fig20, ((ax21, ax22, ax23), (ax24, ax25, ax26)) = plt.subplots(nrows=2, ncols=3)
        differences = np.array(histo_difference)
        diff1 = differences[:,0]
        diff2 = differences[:,1]
        diff3 = differences[:,2]
        diff4 = differences[:,3]
        diff5 = differences[:,4]
        diff6 = differences[:,5]

        n_bins = min(MAX_NUM_BINS, max(MIN_NUM_BINS, int((diff1.max() - diff1.min()) / STD_BIN_SIZE)))
        ax21.hist(diff1, n_bins, density=False, histtype='bar', stacked=True)
        ax21.set_title('POI1: Start-of-fill')

        n_bins = min(MAX_NUM_BINS, max(MIN_NUM_BINS, int((diff2.max() - diff2.min()) / STD_BIN_SIZE)))
        ax22.hist(diff2, n_bins, density=False, histtype='bar', stacked=True)
        ax22.set_title('POI2: End-of-fill')

        n_bins = min(MAX_NUM_BINS, max(MIN_NUM_BINS, int((diff3.max() - diff3.min()) / STD_BIN_SIZE)))
        ax23.hist(diff3, n_bins, density=False, histtype='bar', stacked=True)
        ax23.set_title('POI3: Post-point')

        n_bins = min(MAX_NUM_BINS, max(MIN_NUM_BINS, int((diff4.max() - diff4.min()) / STD_BIN_SIZE)))
        ax24.hist(diff4, n_bins, density=False, histtype='bar', stacked=True)
        ax24.set_title('POI4: CH1')

        n_bins = min(MAX_NUM_BINS, max(MIN_NUM_BINS, int((diff5.max() - diff5.min()) / STD_BIN_SIZE)))
        ax25.hist(diff5, n_bins, density=False, histtype='bar', stacked=True)
        ax25.set_title('POI5: CH2')

        n_bins = min(MAX_NUM_BINS, max(MIN_NUM_BINS, int((diff6.max() - diff6.min()) / STD_BIN_SIZE)))
        ax26.hist(diff6, n_bins, density=False, histtype='bar', stacked=True)
        ax26.set_title('POI6: CH3')

        fig20.show()

        print()
        print("ModelData" if USE_MODELDATA else "QModel", "Query Times:")
        self.query_time_avg = self.query_time_sum / self.query_time_num
        print(f"Min = {self.query_time_min}secs")
        print(f"Avg = {self.query_time_avg}secs")
        print(f"Max = {self.query_time_max}secs")

        print()
        print("ModelData" if USE_MODELDATA else "QModel", "Sequence Order:")
        good_pct = 100 * (self.poi_order_good / (self.poi_order_good + self.poi_order_bad))
        bad_pct = 100 * (self.poi_order_bad / (self.poi_order_good + self.poi_order_bad))
        print(f"Good = {self.poi_order_good} ({good_pct:2.0f}%)")
        print(f"Bad = {self.poi_order_bad} ({bad_pct:2.0f}%)")

        print()
        print("ModelData" if USE_MODELDATA else "QModel", "Out-Of-Order Indices:")
        print(self.poi_index_bad)

        input() # pause, keep fig open


    def plot_time_vs_distance(self):
        training_data_path = r"C:\Users\Alexander J. Ross\Documents\QATCH Work\ML\training_data"
        dithered_data_path = os.path.join(training_data_path, "Dithered_Data")
        num_runs = self.find_files(dithered_data_path)

        self.random_order = np.arange(0, num_runs, 1)
        random.shuffle(self.random_order)
        self.run_count = 0

        self.show_next_plot()
        input() # wait for user prompted close

        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        fig.suptitle("Time vs. Distance fit line")
        ax1.set_xlabel('Time (normalized)')
        ax1.set_ylabel('Distance (mm)')

        distances = [1.15, 1.61, 2.17, 5.00, 10.90, 16.2]
        ax1.add_patch(
            patches.Rectangle((-0.01, distances[0]-0.5), 0.04, 2, edgecolor='g', facecolor='g')
        )
        ax1.add_patch(
            patches.Rectangle((0.03, distances[3]-0.5), 0.17, 1, edgecolor='g', facecolor='g')
        )
        ax1.add_patch(
            patches.Rectangle((0.35, distances[4]-0.5), 0.40, 1, edgecolor='g', facecolor='g')
        )
        ax1.add_patch(
            patches.Rectangle((0.20, distances[3]-0.5), 0.15, 6.9, edgecolor='r', facecolor='none')
        )
        ax1.add_patch(
            patches.Rectangle((0.75, distances[4]-0.5), 0.15, 6.3, edgecolor='r', facecolor='none')
        )

        run_count = 0
        desc_text = "Time vs. Distance"
        for run in tqdm(self.random_order, desc=desc_text, unit="runs"):
            run_count += 1

            data_path = self.csv_files[run]
            with open(data_path, 'r') as f:
                csv_headers = next(f)
                if isinstance(csv_headers, bytes):
                    csv_headers = csv_headers.decode()
                if "Ambient" in csv_headers:
                    csv_cols = (2,4,6,7)
                else:
                    csv_cols = (2,3,5,6)
                data  = np.loadtxt(f.readlines(), delimiter = ',', skiprows = 0, usecols = csv_cols)
            
            poi_path = self.poi_files[run]
            poi_expected = np.loadtxt(poi_path, dtype=int)

            try:
                all_times = data[:,0]
                times = all_times[poi_expected]
                times = self.normalize(times)
                distances = [1.15, 1.61, 2.17, 5.00, 10.90, 16.2]
            except:
                print("\n ERROR skipping run:", os.path.basename(data_path))
                continue

            D0 = 0 # distances[0]
            Dss = 1 # 20 - D0
            RC = times[-1]
            t0 = times[0]
            t = np.linspace(times[0], times[-1], 1000)
            fit_curve_y = Dss*(1-np.exp(-(t-t0)/RC))+D0
            fit_curve_y = self.normalize(fit_curve_y, distances[0]+distances[2], distances[-1])

            i = next(x for x,y in enumerate(fit_curve_y) if y >= distances[3])
            x_offset = np.linspace(distances[2], 0, i)
            for x,y in enumerate(x_offset):
                fit_curve_y[x] -= y

            if run_count == 1:
                ax1.plot(times, distances, color="blue", marker="x", mec="black", label="Actual Data")
                ax1.plot(t, fit_curve_y, color="red", label="Trendline")
            else:
                ax1.plot(times, distances, color="blue", marker="x", mec="black")
                ax1.plot(t, fit_curve_y, color="red")

        def generate_zone_probabilities(region_size):
            amplitude = distances[-1]
            poi4_min_val = 0.2 * amplitude
            periodicity4 = 9
            period_skip4 = False
            poi5_min_val = 0.3 * amplitude
            periodicity5 = 6
            period_skip5 = True
            t = np.linspace(0, 1, region_size)
            
            signal_region_equation_POI4 = 1 - np.cos(periodicity4*t*np.pi)
            signal_region_equation_POI4 = self.normalize(signal_region_equation_POI4, poi4_min_val, amplitude)
            if period_skip4:
                signal_region_equation_POI4 = np.where(t < 2/periodicity4, poi4_min_val, signal_region_equation_POI4)
                signal_region_equation_POI4 = np.where(t > 4/periodicity4, poi4_min_val, signal_region_equation_POI4)
            else:
                signal_region_equation_POI4 = np.where(t > 2/periodicity4, poi4_min_val, signal_region_equation_POI4)

            signal_region_equation_POI5 = 1 - np.cos(periodicity5*t*np.pi)
            signal_region_equation_POI5 = self.normalize(signal_region_equation_POI5, poi5_min_val, amplitude)
            if period_skip5:
                signal_region_equation_POI5 = np.where(t < 2/periodicity5, poi5_min_val, signal_region_equation_POI5)
                signal_region_equation_POI5 = np.where(t > 4/periodicity5, poi5_min_val, signal_region_equation_POI5)
            else:
                signal_region_equation_POI5 = np.where(t > 2/periodicity5, poi5_min_val, signal_region_equation_POI5)

            signal_region_equation_POI4 = np.where(t < 0.03, 0, signal_region_equation_POI4)
            if period_skip5:
                signal_region_equation_POI4 = np.where(t > 2/periodicity5, 0, signal_region_equation_POI4)
            if not period_skip4:
                signal_region_equation_POI5 = np.where(t < 2/periodicity4, 0, signal_region_equation_POI5)
            signal_region_equation_POI5 = np.where(t > 0.75, poi4_min_val, signal_region_equation_POI5)
            signal_region_equation_POI5 = np.where(t > 0.90, 0, signal_region_equation_POI5)

            return [signal_region_equation_POI4, signal_region_equation_POI5]
        
        signal_region_equation_POI4, signal_region_equation_POI5 = generate_zone_probabilities(1000)

        ax1.plot(t, signal_region_equation_POI4, color="lime", label="POI4 probability")
        ax1.plot(t, signal_region_equation_POI5, color="orange", label="POI5 probability")

        fig.legend(loc="lower right")
        fig.show()
        input()

    def normalize(self, arr, t_min = 0, t_max = 1):
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

    def show_next_plot(self, event = None):
        try:
            run = self.random_order[self.run_count]
        except:
            print("Finished!")
        self.run_count += 1
        data_path = self.csv_files[run]
        with open(data_path, 'r') as f:
            csv_headers = next(f)
            if isinstance(csv_headers, bytes):
                csv_headers = csv_headers.decode()
            if "Ambient" in csv_headers:
                csv_cols = (2,4,6,7)
            else:
                csv_cols = (2,3,5,6)
            data  = np.loadtxt(f.readlines(), delimiter = ',', skiprows = 0, usecols = csv_cols)
        
        poi_path = self.poi_files[run]
        poi_expected = np.loadtxt(poi_path, dtype=int)

        try:
            all_times = data[:,0]
            times = all_times[poi_expected]
            distances = [1.15, 1.61, 2.17, 5.00, 10.90, 16.2]
        except:
            print("\n ERROR skipping run:", os.path.basename(data_path))
            self.show_next_plot()
            return

        if True:
            D0 = 0 # distances[0]
            Dss = 1 # 20 - D0
            RC = times[-1]
            t0 = times[0]
            t = np.linspace(times[0], times[-1], 1000)
            fit_curve_y = Dss*(1-np.exp(-(t-t0)/RC))+D0
            fit_curve_y = self.normalize(fit_curve_y, distances[0]+distances[2], distances[-1])

            i = next(x for x,y in enumerate(fit_curve_y) if y >= distances[3])
            x_offset = np.linspace(distances[2], 0, i)
            for x,y in enumerate(x_offset):
                fit_curve_y[x] -= y

        # def curveCapChargeFit(x, a, b, c, d):
        #     return a*(1-np.exp(-(x-b)/c))+d
        # 
        # p0 = (Dss, t0, RC, D0) # start with values near those we expect
        # a, b, c, d = p0 # default, not yet optimized
        # best_fit_pts = fit_curve_y # default, not yet optimized
        # try:
        #     params, cv = curve_fit(curveCapChargeFit, times, distances, p0)
        #     a, b, c, d = params
        #     best_fit_pts = curveCapChargeFit(t, a, b, c, d)
        #     print(f"Normalized fit coeffs: {params}")
        # except Exception as e:
        #     print("Curve fit failed to find optimal parameters")
        #     # raise e

        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        fig.suptitle("Time vs. Distance fit line")
        ax1.set_title(os.path.basename(data_path))
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Distance (mm)')

        ax1.plot(times, distances, color="blue", marker="x", mec="black")
        ax1.plot(t, fit_curve_y, color="red")
        # ax1.plot(t, best_fit_pts, color="green")

        fig.canvas.mpl_connect('close_event', self.show_next_plot)
        fig.show()


    def run(self):
        # self.benchmark_model()
        self.plot_time_vs_distance()


if __name__ == '__main__':
    VerifyQModel().run()