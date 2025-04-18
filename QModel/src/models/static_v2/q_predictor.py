
# from ModelData import ModelData
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.signal import find_peaks
import pickle
from scipy.interpolate import interp1d
import joblib
import tensorflow as tf
from q_data_pipeline import QPartialDataPipeline
from q_image_clusterer import QClusterer
Architecture_found = False
try:
    if not Architecture_found:
        from QATCH.common.architecture import Architecture
    Architecture_found = True
except:
    Architecture_found = False
    # Not finding this if OK: will use 'cwd' as 'root' path

QConstants_found = False
try:
    if not QConstants_found:
        from q_constants import *
    QConstants_found = True
except:
    QConstants_found = False
try:
    if not QConstants_found:
        from QATCH.QModel.QConstants import *
    QConstants_found = True
except:
    QConstants_found = False
if not QConstants_found:
    raise ImportError("Cannot find 'QConstants' in any expected location.")

ModelData_found = False
try:
    if not ModelData_found:
        from ModelData import ModelData
    ModelData_found = True
except:
    ModelData_found = False
try:
    if not ModelData_found:
        from QATCH.models.ModelData import ModelData
    ModelData_found = True
except:
    ModelData_found = False
if not ModelData_found:
    raise ImportError("Cannot find 'ModelData' in any expected location.")

QDataPipeline_found = False
try:
    if not QDataPipeline_found:
        from q_data_pipeline import QDataPipeline
    QDataPipeline_found = True
except:
    QDataPipeline_found = False
try:
    if not QDataPipeline_found:
        from QATCH.QModel.q_data_pipeline import QDataPipeline
    QDataPipeline_found = True
except:
    QDataPipeline_found = False
if not QDataPipeline_found:
    raise ImportError("Cannot find 'QDataPipeline' in any expected location.")


class QPredictor:
    def __init__(self, model_path: str = None, scaler_path: str = None, label_encoder_path: str = None):
        self._partial_model = tf.keras.models.load_model(model_path)
        self._partial_scaler = joblib.load(scaler_path)
        self._partial_label_encoder = joblib.load(label_encoder_path)

    def predict(self, input_filepath: str = None):
        # Extract features from the input file
        qpd = QPartialDataPipeline(input_filepath)
        self._qdp = QDataPipeline(input_filepath, multi_class=True)
        qpd.preprocess()
        features = qpd.get_features()

        # Ensure all required features are present in the correct order
        feature_df = pd.DataFrame([features])
        feature_df.fillna(0, inplace=True)  # Fill missing values with 0

        # Scale features
        scaled_features = self._partial_scaler.transform(feature_df)

        # Make predictions
        # Single input, get first output
        probabilities = self._partial_model.predict(scaled_features)[0]
        predicted_class_index = np.argmax(probabilities)

        # Decode the predicted class label
        predicted_num_channels = self._partial_label_encoder.inverse_transform(
            [predicted_class_index])[0]

        # Map probabilities to class labels
        class_probabilities = {
            self._partial_label_encoder.inverse_transform([i])[0]: prob
            for i, prob in enumerate(probabilities)
        }
        if predicted_num_channels == "no_fill":
            return predicted_num_channels, None
        elif predicted_num_channels == "channel_1_partial":
            q1p = QChannel1Predictor(
                model_path="QModel\SavedModels\QMulti_Channel_1.json")
            return predicted_num_channels, q1p.predict(input_filepath)
        elif predicted_num_channels == "channel_2_partial":
            q2p = QChannel2Predictor(
                model_path="QModel\SavedModels\QMulti_Channel_2.json")
            return predicted_num_channels, q2p.predict(input_filepath)
        elif predicted_num_channels == "full_fill":
            qcr = QClusterer(model_path="QModel/SavedModels/cluster.joblib")
            full_fill_type = qcr.predict_label(input_filepath)
            if full_fill_type == 0:
                qfp = QFullPredictor("QModel/SavedModels/QMultiType_0.json")
                return predicted_num_channels, qfp.predict(
                    input_filepath, run_type=full_fill_type)
            elif full_fill_type == 1:
                qfp = QFullPredictor("QModel/SavedModels/QMultiType_1.json")
                return predicted_num_channels, qfp.predict(
                    input_filepath, run_type=full_fill_type)
            elif full_fill_type == 2:
                qfp = QFullPredictor("QModel/SavedModels/QMultiType_2.json")
                return predicted_num_channels, qfp.predict(
                    input_filepath, run_type=full_fill_type)
            else:
                raise ValueError(
                    f"Invalid predicted full-fill type was: {full_fill_type}")
        else:
            raise ValueError(
                f"Invalid predicted channel type was: {predicted_num_channels} with probabilty: {class_probabilities} ")


class QChannel1Predictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            raise ValueError("[QModelPredict __init__()] No model path given")

        if Architecture_found:
            relative_root = os.path.join(Architecture.get_path(), "QATCH")
        else:
            relative_root = os.getcwd()

        # Load the pre-trained models from the specified paths
        self._model = xgb.Booster()
        self._model.load_model(model_path)

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def extract_results(self, results):

        num_indices = len(results[0])
        extracted = [[] for _ in range(num_indices)]

        for sublist in results:
            for idx in range(num_indices):
                extracted[idx].append(sublist[idx])

        return extracted

    def find_and_sort_peaks(self, signal):
        # Find peaks
        peaks, properties = find_peaks(signal)
        # Get the peak heights
        peak_heights = []
        for p in peaks:
            peak_heights.append(signal[p])

        # Sort peaks by height in descending order
        sorted_indices = np.argsort(peak_heights)[::-1]
        sorted_peaks = peaks[sorted_indices]
        return sorted_peaks

    def predict(self, file_buffer):
        if isinstance(file_buffer, str):
            df = pd.read_csv(file_buffer)
        elif isinstance(file_buffer, pd.DataFrame):
            df = file_buffer
        # columns_to_drop = ["Date", "Time", "Ambient", "Temperature"]
        # if not all(col in df.columns for col in columns_to_drop):
        #     raise ValueError(
        #         f"[QModelPredict predict()]: Input data must contain the following columns: {columns_to_drop}"
        #     )

        # df = df.drop(columns=columns_to_drop)

        if not isinstance(file_buffer, str) and not isinstance(file_buffer, pd.DataFrame):
            if hasattr(file_buffer, "seekable") and file_buffer.seekable():
                # reset ByteIO buffer to beginning of stream
                file_buffer.seek(0)

            csv_headers = next(file_buffer)

            if isinstance(csv_headers, bytes):
                csv_headers = csv_headers.decode()

            if "Ambient" in csv_headers:
                csv_cols = (2, 4, 6, 7)
            else:
                csv_cols = (2, 3, 5, 6)

            file_data = np.loadtxt(
                file_buffer.readlines(), delimiter=",", skiprows=0, usecols=csv_cols
            )
            data_path = "QModel Passthrough"
            relative_time = file_data[:, 0]
            # temperature = file_data[:,1]
            resonance_frequency = file_data[:, 2]
            data = file_data[:, 3]
        file_buffer_2 = file_buffer
        if not isinstance(file_buffer_2, str) and not isinstance(file_buffer_2, pd.DataFrame):
            if hasattr(file_buffer_2, "seekable") and file_buffer_2.seekable():
                # reset ByteIO buffer to beginning of stream
                file_buffer_2.seek(0)
            else:
                # ERROR: 'file_buffer_2' must be 'BytesIO' type here, but it's not seekable!
                raise Exception(
                    "Cannot 'seek' stream prior to passing to 'QDataPipeline'."
                )
        else:
            # Assuming 'file_buffer_2' is a string to a file path, this will work fine as-is
            pass

        # Process data using QDataPipeline
        qdp = QDataPipeline(file_buffer_2)
        qdp.preprocess(poi_filepath=None)
        df = qdp.get_dataframe()
        f_names = self._model.feature_names
        df = df[f_names]
        d_data = xgb.DMatrix(df)

        results = self._model.predict(d_data)
        normalized_results = self.normalize(results)
        extracted_results = self.extract_results(normalized_results)

        poi_1 = np.argmax(extracted_results[1])
        poi_2 = np.argmax(extracted_results[2])
        poi_3 = np.argmax(extracted_results[3])
        candidates_1 = self.find_and_sort_peaks(extracted_results[1])
        candidates_2 = self.find_and_sort_peaks(extracted_results[2])
        candidates_3 = self.find_and_sort_peaks(extracted_results[3])

        def sort_and_remove_point(arr, point):
            arr = np.array(arr)
            if len(arr) > MAX_GUESSES - 1:
                arr = arr[: MAX_GUESSES - 1]
            arr.sort()

            return arr[arr != point]

        candidates_list = [
            candidates_1,
            candidates_2,
            candidates_3,
        ]
        poi_list = [poi_1, poi_2, poi_3]
        extracted_confidences = extracted_results[1:len(poi_list) + 1]

        candidates = []

        for i in range(len(poi_list) - 1):

            # Sort and remove point
            candidates_i = sort_and_remove_point(
                candidates_list[i], poi_list[i])
            filtered_points = candidates_i
            if i < 3:
                mean = np.mean(candidates_i)
                std_dev = np.std(candidates_i)
                threshold = 2
                # Filter points within the specified threshold
                filtered_points = [point for point in candidates_i if abs(
                    point - mean) <= threshold * std_dev]

            # Extract and sort confidence
            confidence_i = np.sort(np.array(extracted_confidences[i])[filtered_points])[
                ::-1
            ]

            # Insert POI at the start, remove the last element
            if len(candidates_i) > 1:
                candidates_i = np.insert(candidates_i, 0, poi_list[i])[:-1]
            else:
                candidates_i = np.insert(candidates_i, 0, poi_list[i])

            # Append to candidates list
            candidates.append((candidates_i, confidence_i))
        return candidates


class QChannel2Predictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            raise ValueError("[QModelPredict __init__()] No model path given")

        if Architecture_found:
            relative_root = os.path.join(Architecture.get_path(), "QATCH")
        else:
            relative_root = os.getcwd()

        # Load the pre-trained models from the specified paths
        self._model = xgb.Booster()
        self._model.load_model(model_path)

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def extract_results(self, results):

        num_indices = len(results[0])
        extracted = [[] for _ in range(num_indices)]

        for sublist in results:
            for idx in range(num_indices):
                extracted[idx].append(sublist[idx])

        return extracted

    def find_and_sort_peaks(self, signal):
        # Find peaks
        peaks, properties = find_peaks(signal)
        # Get the peak heights
        peak_heights = []
        for p in peaks:
            peak_heights.append(signal[p])

        # Sort peaks by height in descending order
        sorted_indices = np.argsort(peak_heights)[::-1]
        sorted_peaks = peaks[sorted_indices]
        return sorted_peaks

    def predict(self, file_buffer):
        if isinstance(file_buffer, str):
            df = pd.read_csv(file_buffer)
        elif isinstance(file_buffer, pd.DataFrame):
            df = file_buffer
        # columns_to_drop = ["Date", "Time", "Ambient", "Temperature"]
        # if not all(col in df.columns for col in columns_to_drop):
        #     raise ValueError(
        #         f"[QModelPredict predict()]: Input data must contain the following columns: {columns_to_drop}"
        #     )

        # df = df.drop(columns=columns_to_drop)
        if not isinstance(file_buffer, str) and not isinstance(file_buffer, pd.DataFrame):
            if hasattr(file_buffer, "seekable") and file_buffer.seekable():
                # reset ByteIO buffer to beginning of stream
                file_buffer.seek(0)

            csv_headers = next(file_buffer)

            if isinstance(csv_headers, bytes):
                csv_headers = csv_headers.decode()

            if "Ambient" in csv_headers:
                csv_cols = (2, 4, 6, 7)
            else:
                csv_cols = (2, 3, 5, 6)

            file_data = np.loadtxt(
                file_buffer.readlines(), delimiter=",", skiprows=0, usecols=csv_cols
            )
            data_path = "QModel Passthrough"
            relative_time = file_data[:, 0]
            # temperature = file_data[:,1]
            resonance_frequency = file_data[:, 2]
            data = file_data[:, 3]
        file_buffer_2 = file_buffer
        if not isinstance(file_buffer_2, str) and not isinstance(file_buffer_2, pd.DataFrame):
            if hasattr(file_buffer_2, "seekable") and file_buffer_2.seekable():
                # reset ByteIO buffer to beginning of stream
                file_buffer_2.seek(0)
            else:
                # ERROR: 'file_buffer_2' must be 'BytesIO' type here, but it's not seekable!
                raise Exception(
                    "Cannot 'seek' stream prior to passing to 'QDataPipeline'."
                )
        else:
            # Assuming 'file_buffer_2' is a string to a file path, this will work fine as-is
            pass

        # Process data using QDataPipeline
        qdp = QDataPipeline(file_buffer_2)
        qdp.preprocess(poi_filepath=None)
        df = qdp.get_dataframe()
        f_names = self._model.feature_names
        df = df[f_names]
        d_data = xgb.DMatrix(df)

        results = self._model.predict(d_data)
        normalized_results = self.normalize(results)
        extracted_results = self.extract_results(normalized_results)

        poi_1 = np.argmax(extracted_results[1])
        poi_2 = np.argmax(extracted_results[2])
        poi_3 = np.argmax(extracted_results[3])
        poi_4 = np.argmax(extracted_results[4])
        poi_5 = np.argmax(extracted_results[5])
        candidates_1 = self.find_and_sort_peaks(extracted_results[1])
        candidates_2 = self.find_and_sort_peaks(extracted_results[2])
        candidates_3 = self.find_and_sort_peaks(extracted_results[3])
        candidates_4 = self.find_and_sort_peaks(extracted_results[4])
        candidates_5 = self.find_and_sort_peaks(extracted_results[5])

        def sort_and_remove_point(arr, point):
            arr = np.array(arr)
            if len(arr) > MAX_GUESSES - 1:
                arr = arr[: MAX_GUESSES - 1]
            arr.sort()

            return arr[arr != point]

        candidates_list = [
            candidates_1,
            candidates_2,
            candidates_3,
            candidates_4,
            candidates_5,
        ]
        poi_list = [poi_1, poi_2, poi_3, poi_4, poi_5]
        extracted_confidences = extracted_results[1:len(poi_list) + 1]

        candidates = []

        for i in range(len(poi_list) - 1):

            # Sort and remove points
            candidates_i = sort_and_remove_point(
                candidates_list[i], poi_list[i])
            filtered_points = candidates_i
            if i < 3:
                mean = np.mean(candidates_i)
                std_dev = np.std(candidates_i)
                threshold = 2
                # Filter points within the specified threshold
                filtered_points = [point for point in candidates_i if abs(
                    point - mean) <= threshold * std_dev]

            # Extract and sort confidence
            confidence_i = np.sort(np.array(extracted_confidences[i])[filtered_points])[
                ::-1
            ]

            # Insert POI at the start, remove the last element
            if len(candidates_i) > 1:
                candidates_i = np.insert(candidates_i, 0, poi_list[i])[:-1]
            else:
                candidates_i = np.insert(candidates_i, 0, poi_list[i])

            # Append to candidates list
            candidates.append((candidates_i, confidence_i))
        return candidates


class QFullPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            raise ValueError("[QModelPredict __init__()] No model path given")

        # Load the pre-trained models from the specified paths
        self.__model__ = xgb.Booster()

        self.__model__.load_model(model_path)
        if Architecture_found:
            relative_root = os.path.join(Architecture.get_path(), "QATCH")
        else:
            relative_root = os.getcwd()
        pickle_path = os.path.join(
            relative_root, "QModel/SavedModels/label_{}.pkl")
        for i in range(3):
            with open(pickle_path.format(i), "rb") as file:
                setattr(self, f"__label_{i}__", pickle.load(file))

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def compute_bounds(self, indices):
        bounds = []
        start = indices[0]
        end = indices[0]

        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                end = indices[i]
            else:
                bounds.append((start, end))
                start = indices[i]
                end = indices[i]

        # Append the last run
        bounds.append((start, end))

        return bounds

    def extract_results(self, results):

        num_indices = len(results[0])
        extracted = [[] for _ in range(num_indices)]

        for sublist in results:
            for idx in range(num_indices):
                extracted[idx].append(sublist[idx])

        return extracted

    def adjust_predictions(self, prediction, rel_time, poi_num, type, i, j):
        rel_time = rel_time[i:j]
        rel_time_norm = self.normalize(rel_time)
        bounds = None

        if type == 0:
            bounds = self.__label_0__["Class_" + str(poi_num)]
        elif type == 1:
            bounds = self.__label_1__["Class_" + str(poi_num)]
        elif type == 2:
            bounds = self.__label_2__["Class_" + str(poi_num)]

        lq = bounds["lq"]
        uq = bounds["uq"]
        adj = np.where((rel_time_norm >= lq) & (rel_time_norm <= uq), 1, 0)
        adjustment = np.concatenate(
            (np.zeros(i), np.array(adj), (np.zeros(len(prediction) - j)))
        )
        if len(prediction) == len(adjustment):
            adj_prediction = prediction * adjustment
            lq_idx = next((i for i, x in enumerate(adj) if x == 1), -1) + i
            uq_idx = (
                next((i for i, x in reversed(list(enumerate(adj))) if x == 1), -1) + i
            )
            return adj_prediction, (lq_idx, uq_idx)

        lq_idx = next((i for i, x in enumerate(adj) if x == 1), -1) + i
        uq_idx = next((i for i, x in reversed(
            list(enumerate(adj))) if x == 1), -1) + i
        return prediction, (lq_idx, uq_idx)

    def find_and_sort_peaks(self, signal):
        # Find peaks
        peaks, properties = find_peaks(signal)
        # Get the peak heights
        peak_heights = []
        for p in peaks:
            peak_heights.append(signal[p])

        # Sort peaks by height in descending order
        sorted_indices = np.argsort(peak_heights)[::-1]
        sorted_peaks = peaks[sorted_indices]
        return sorted_peaks

    def peak_density(self, signal, peaks, segment_le):
        density = len(peaks) / len(signal)
        return density

    def find_dynamic_increase(
        self, data, target_points=5, multiplier=3.0, reduction_factor=0.5, window_size=3
    ):
        if window_size < 2 or window_size >= len(data):
            return -1

        # Calculate the first derivative (differences between consecutive points)
        diff = np.diff(data)
        slope_change = []
        for i in range(len(diff) - window_size + 1):
            # Calculate the slope change over the current window
            window_slope_change = np.mean(np.diff(diff[i: i + window_size]))
            slope_change.append(window_slope_change)

        slope_change = np.array(slope_change)

        # Calculate baseline threshold (std deviation of slope changes in the sliding windows)
        baseline = np.std(slope_change)

        significant_points = []
        current_multiplier = multiplier

        # Iteratively reduce the threshold until enough significant points are found
        while len(significant_points) < target_points and current_multiplier > 0:
            # Calculate the current dynamic threshold
            dynamic_threshold = current_multiplier * baseline

            # Find new points where the change in slope exceeds the dynamic threshold
            new_points = np.where(slope_change > dynamic_threshold)[0]

            # Add new significant points to the list, ensuring no duplicates
            for point in new_points:
                if point + window_size - 1 not in [
                    p for p in significant_points
                ]:  # Adjust for window size
                    significant_points.append(
                        point + window_size - 1
                    )  # Store index and slope change
                    if len(significant_points) >= target_points:
                        break

            # Reduce the multiplier for the next iteration
            current_multiplier *= reduction_factor

        return significant_points[:target_points]

    def find_zero_slope_regions(self, data, threshold=1e-3, min_region_length=1):
        # Calculate the first derivative (approximate slope)
        slopes = np.diff(data)

        # Identify regions where the slope is approximately zero
        zero_slope_mask = np.abs(slopes) < threshold

        # Find consecutive regions of near-zero slope
        regions = []
        start_idx = None

        for i, is_zero_slope in enumerate(zero_slope_mask):
            if is_zero_slope and start_idx is None:
                start_idx = i
            elif not is_zero_slope and start_idx is not None:
                if i - start_idx >= min_region_length:
                    # Store the region as (start, end)
                    regions.append((start_idx, i))
                start_idx = None

        # Handle case where the last region extends to the end of the array
        if start_idx is not None and len(data) - start_idx - 1 >= min_region_length:
            regions.append((start_idx, len(data) - 1))

        return regions

    def adjustment_poi_1(self, initial_guess, dissipation_data):
        """Adjusts the point of interest (POI) based on the nearest zero-slope region or peak.

        This method refines the initial guess of the point of interest by finding
        nearby regions with zero slope and selecting the most suitable peak. If no
        valid zero-slope region is found, it adjusts the guess to the nearest peak
        in the dissipation data.

        Args:
            initial_guess (int): Initial point of interest guess, typically an index within the dataset.
            dissipation_data (np.ndarray): 1D array of dissipation values.

        Returns:
            int: Adjusted index of the point of interest, either at or near a zero-slope
                region or closest peak.
        """
        zero_slope_regions = self.find_zero_slope_regions(
            self.normalize(dissipation_data), threshold=0.0075, min_region_length=100
        )
        adjusted_guess = initial_guess

        if len(zero_slope_regions) >= 2:
            left_bound = zero_slope_regions[0][1]
            right_bound = zero_slope_regions[1][0]

            peaks_between, _ = find_peaks(
                dissipation_data[left_bound:right_bound])
            peaks_indices = [peak + left_bound for peak in peaks_between]

            if peaks_indices:
                peaks_array = np.array(peaks_indices)
                distances = np.abs(peaks_array - initial_guess)

                if adjusted_guess >= right_bound:
                    furthest_index = np.argmax(distances)
                    distances[furthest_index] = np.min(
                        distances
                    )  # Ignore the furthest peak for the second furthest
                    second_furthest_index = np.argmax(distances)
                    adjusted_guess = peaks_array[second_furthest_index]
                else:
                    nearest_index = np.argmin(distances)
                    adjusted_guess = peaks_array[nearest_index]

        else:
            all_peaks, _ = find_peaks(dissipation_data)
            distances_to_peaks = np.abs(all_peaks - adjusted_guess)
            nearest_peak = all_peaks[np.argmin(distances_to_peaks)]
            adjusted_guess = nearest_peak

        return adjusted_guess

    def adjustment_poi_2(self, initial_guess, dissipation_data, bounds, poi_1_estimate):
        """Adjusts the point of interest (POI) within specified bounds, influenced by another POI.

        This method refines the initial guess for a point of interest based on specified
        bounds and a previous POI estimate. It calculates the nearest peak within the bounds,
        weighted toward the initial guess, and returns an adjusted POI if valid.

        Args:
            initial_guess (int): Initial guess for the point of interest, typically an index.
            dissipation_data (np.ndarray): 1D array of dissipation values to be processed.
            bounds (tuple[int, int]): Start and stop bounds within which to adjust the POI.
            poi_1_estimate (int): Estimate of a prior POI that affects the adjustment range.

        Returns:
            int: Adjusted index of the point of interest within the specified bounds.
        """
        dissipation_data = self.normalize(dissipation_data)
        dissipation_inverted = -dissipation_data
        start_bound, stop_bound = bounds

        if initial_guess < start_bound:
            start_bound = initial_guess
        elif initial_guess > stop_bound:
            stop_bound = initial_guess

        if start_bound < poi_1_estimate:
            start_bound = poi_1_estimate + 1
        if stop_bound < poi_1_estimate:
            range_delta = stop_bound - start_bound
            start_bound = poi_1_estimate
            stop_bound = start_bound + range_delta - 1

        peaks, _ = find_peaks(dissipation_inverted)
        if len(peaks) == 0:
            return initial_guess

        valid_peaks = peaks[peaks < initial_guess]
        distances_to_guess = np.abs(valid_peaks - initial_guess)
        closest_peak_idx = np.argmin(distances_to_guess)
        adjusted_guess = valid_peaks[closest_peak_idx]

        if adjusted_guess < poi_1_estimate:
            return initial_guess

        points = np.array([adjusted_guess, initial_guess])
        # Biases adjustment toward the initial guess
        weights = np.array([0.25, 0.75])
        weighted_mean = np.average(points, weights=weights)
        final_adjustment = int(weighted_mean)

        if abs(initial_guess - final_adjustment) > 75:
            final_adjustment = initial_guess

        return final_adjustment

    def adjustmet_poi_4(self, df, candidates, guess, actual, bounds, poi_1_guess):
        dissipation = self.normalize(df["Dissipation"])
        rf = self.normalize(df["Resonance_Frequency"])
        difference = self.normalize(df["Difference"])
        candidates = np.append(candidates, guess)

        # Temporary variable for the adjusted point.
        adjusted_point = -1

        def find_zero_slope_regions(data, threshold=0.1):
            # Calculate the slope (difference between consecutive points)
            slopes = np.diff(data)

            # Identify indices where the slope is within the threshold
            zero_slope_indices = np.where(np.abs(slopes) < threshold)[0]

            # Group indices into continuous regions
            zero_slope_regions = []
            if zero_slope_indices.size > 0:
                current_region = [zero_slope_indices[0]]

                for i in range(1, len(zero_slope_indices)):
                    if zero_slope_indices[i] == zero_slope_indices[i - 1] + 1:
                        current_region.append(zero_slope_indices[i])
                    else:
                        zero_slope_regions.append(current_region)
                        current_region = [zero_slope_indices[i]]

                # Append the last region
                zero_slope_regions.append(current_region)

            return zero_slope_regions

        peaks, _ = find_peaks(rf)

        # Check if there is a filtering of peaks such that they appear within our predefined, bounded
        # region.  If not, just report the guessed POI.
        filtered_peaks = [
            point for point in peaks if bounds[0] <= point <= bounds[1]]

        if len(filtered_peaks) == 0:
            # TODO: Adjust POI if there are no Resonance frequency peaks in the bounded region.  Potentially look at something like
            # Difference peaks or dissipation valleys and try to correlate.
            adjusted_point = guess
        else:
            # filtered_peaks = [point for point in peaks if bounds[0] <= point <= bounds[1]]
            filtered_peaks = [point for point in peaks if poi_1_guess <= point]
            # Interpolate between peaks to create the upper envelope
            t = np.arange(len(rf))
            envelope_interpolator = interp1d(
                t[peaks], rf[peaks], kind="linear", fill_value="extrapolate"
            )
            upper_envelope = envelope_interpolator(t)

            # Compute regions of approximately zero slope in the RF curve.
            zsr = find_zero_slope_regions(upper_envelope, threshold=0.00001)
            moved = False
            for region in zsr:
                lb = region[0]
                rb = region[-1]
                if lb <= filtered_peaks[0] <= rb:
                    adjusted_point = filtered_peaks[1]
                    moved = True
                    break
            if not moved:
                adjusted_point = filtered_peaks[0]
        # if abs(actual[3] - adjusted_point) > 50:
        #     fig, ax = plt.subplots()
        #     ax.plot(dissipation, label="Dissipation", color="black")
        #     ax.plot(difference, label="Difference", color="tan")
        #     ax.fill_betweenx(
        #         [0, max(dissipation)], bounds[0], bounds[1], color=f"yellow", alpha=0.1
        #     )
        #     ax.plot(rf, label="rf", color="grey")
        #     ax.plot(upper_envelope, label="upper_envelope", color="brown")
        #     ax.scatter(actual, upper_envelope[actual], color="blue", label="Actual")
        #     ax.scatter(
        #         candidates,
        #         upper_envelope[candidates],
        #         color="red",
        #         label="Candidates",
        #         marker="x",
        #     )
        #     for region in zsr:
        #         ax.axvspan(
        #             region[0],
        #             region[-1],
        #             color="red",
        #             alpha=0.5,
        #             label="Zero Slope Region",
        #         )
        #     ax.scatter(
        #         filtered_peaks,
        #         upper_envelope[filtered_peaks],
        #         color="green",
        #         label="Peaks",
        #     )

        #     ax.axvline(adjusted_point, color="orange", label="adjusted_point")
        #     plt.legend()
        #     plt.show()
        return adjusted_point

    def adjustmet_poi_5(
        self, df, candidates, guess, actual, bounds, poi_4_guess, poi_6_guess
    ):
        diss = df["Dissipation"]
        rf = df["Resonance_Frequency"]
        diff = df["Difference"]

        diss_peaks, _ = find_peaks(diss)
        rf_peaks, _ = find_peaks(rf)
        diff_peaks, _ = find_peaks(diff)
        initial_guess = np.array(guess)
        candidate_points = np.array(candidates)
        rf_points = np.array(rf_peaks)

        diss_points = np.array(diss_peaks)
        diff_points = np.array(diff_peaks)
        np.concatenate((rf_points, diff_points, diss_points))
        x_min, x_max = bounds
        if x_min < poi_4_guess:
            x_min = poi_4_guess + 1
        if x_max > poi_6_guess:
            x_max = poi_6_guess - 1
        candidate_density = len(candidates) / (x_max - x_min)
        adjusted_point = -1
        if candidate_density < 0.01:
            zero_slope = self.find_zero_slope_regions(rf)
            # Filter RF points within the bounds
            # rf_points = np.concatenate((rf_points, diss_points))
            candidates = np.append(candidates, guess)
            within_bounds = (rf_points >= x_min) & (rf_points <= x_max)
            filtered_rf_points = rf_points[within_bounds]

            for l, r in zero_slope:
                np.append(rf_points, l)
                np.append(rf_points, r)
            # If no RF points within the bounds, return None or handle accordingly
            if filtered_rf_points.size == 0:
                return guess

            # Calculate weights for filtered RF points
            weights = np.array([1 for rf_point in filtered_rf_points])

            # Calculate weighted distances from initial guess to each filtered RF point
            distances_to_rf = np.abs(filtered_rf_points - initial_guess)
            weighted_distances = distances_to_rf / weights  # Adjust distance by weight

            # Find the closest RF point to the initial guess, considering weights
            closest_rf_idx = np.argmin(weighted_distances)
            closest_rf_point = filtered_rf_points[closest_rf_idx]

            # Calculate distances from candidate points to the closest RF point
            distances_to_closest_rf = np.abs(
                candidate_points - closest_rf_point)

            # Find the closest candidate point to the closest RF point
            closest_candidate_idx = np.argmin(distances_to_closest_rf)
            adjusted_point = candidate_points[closest_candidate_idx]

        else:
            adjusted_point = guess

        # if abs(adjusted_point - actual) > 50:
        #     fig, ax = plt.subplots()
        #     ax.plot(diss, color="grey")
        #     ax.fill_betweenx(
        #         [0, max(diss)], bounds[0], bounds[1], color=f"yellow", alpha=0.5
        #     )
        #     ax.scatter(diss_peaks, diss[diss_peaks], color="red", label="diss peaks")
        #     ax.scatter(diff_peaks, diss[diff_peaks], color="green", label="diff peaks")
        #     ax.scatter(
        #         emp_guess, diss[emp_guess], color="pink", marker="x", label="emp guess"
        #     )
        #     ax.scatter(candidates, diss[candidates], color="black", label="candidates")
        #     ax.axvline(guess, color="purple", linestyle="dotted", label="guess")

        #     ax.axvline(adjusted_point, color="brown", label="adjusted")
        #     ax.axvline(actual, color="orange", linestyle="--", label="actual")
        #     ax.scatter(actual, diss[actual], color="orange", marker="*")
        #     plt.legend()
        #     plt.show()
        return adjusted_point

    def adjustment_poi_6(
        self,
        poi_6_guess,
        candidates,
        dissipation,
        difference,
        rf,
        signal,
        t_delta,
        actual,
    ):
        # The following are normalized datasets collected from the raw raun data.
        rf = self.normalize(rf)
        dissipation = self.normalize(dissipation)
        difference = self.normalize(difference)
        signal = self.normalize(signal)

        # Diff peaks are peaks in the difference curve
        diff_peaks, _ = find_peaks(difference)
        # The list of candidates appends the intial guess of POI6 to the list of potential candidates.
        candidates = np.append(candidates, [poi_6_guess])

        def classify_tail(data, a, b):
            slope = data[a] - data[b]
            if slope < 0:
                return "increasing"
            elif slope > 0:
                return "decreasing"
            else:
                # TODO: Implement a method for determining noise at the end of the signal.
                return "noisy"

        # Subtract the difference from dissipation curve to get segments where the normalized difference
        # signal interesects witht the noramlized dissipation signal.
        diff_signal = difference - dissipation
        crossings = np.where(np.diff(np.sign(diff_signal)))[0]

        # Take the last crossing in where the difference and dissipation signal instersect.
        nearest_crossing = max(crossings)

        # Next, get the peak in the difference curve that minimizes the distance between to the final
        # interesction.  This peak indicates the the peak of interest ending the run.
        distances = np.abs(diff_peaks - nearest_crossing)
        nearest_peak = diff_peaks[np.argmin(distances)]

        # The nearest peak and crossing can be out of order so the following ensures that these two points
        # are in ascending order.
        a, b = (
            (nearest_peak, nearest_crossing)
            if nearest_peak < nearest_crossing
            else (nearest_crossing, nearest_peak)
        )
        # Get the type of tail the last 10% of the run classifies as.  10% is an artbirary
        # fraction of the run and can be adjusted.
        tail_class = classify_tail(difference, a, b)
        if tail_class == "increasing":
            # For an increasing tail in the difference curve, currently no adjustment is provided.
            # TODO: For an increasing tail, implelment an adjustment.
            # I think this could be something along the lines of looking at nearest valley in both
            # difference and dissipation and moving poi guess to that but I am not sure.
            filtered_candidates = candidates
            adjusted_poi_6 = poi_6_guess
        elif tail_class == "decreasing":
            # For decreasing tails, there are 2 cases: (1) Between the nearest peak and crossing point,
            # there exists some candidates.  Pick the candidate closest to the base of the increase segment
            # in the dissipation curve. (2) There are no candidates, in which case, pick the point on the
            # dissipation curve which has the most significant change in slope over the baseline slope of
            # that region.
            filtered_candidates = [
                point for point in candidates if a <= point <= b]

            if len(filtered_candidates) > 0:
                adjusted_poi_6 = min(filtered_candidates,
                                     key=lambda x: abs(x - a))
                # adjusted_poi_6 = max(filtered_candidates,
                #                      key=lambda p: signal[p])
                tail_class = tail_class + "_A"
            else:
                slope = np.diff(dissipation[a:b])
                if len(slope) == 0:
                    adjusted_poi_6 = poi_6_guess
                else:
                    average_slope = np.mean(slope)
                    increasing_index = np.argmax(slope > average_slope)
                    significant_point = increasing_index + 1
                    adjusted_poi_6 = significant_point + a
        else:
            # TODO: Noise case
            # The final case is instended to handle the case where the end of the run is noisy.
            filtered_candidates = candidates
            adjusted_poi_6 = poi_6_guess

        # if abs(actual[5] - adjusted_poi_6) > 30 and tail_class == "increasing":
        #     plt.figure(figsize=(8, 8))
        #     plt.plot(dissipation, label="Dissipation", color="grey")
        #     plt.scatter(
        #         filtered_candidates,
        #         dissipation[filtered_candidates],
        #         color="red",
        #         label="Candidates",
        #         marker="x",
        #     )

        #     plt.plot(difference, label="Difference", color="brown")
        #     plt.plot(rf, label="Resonance frequency", color="tan")

        #     plt.scatter(actual, dissipation[actual], label="actual", color="blue")
        #     # plt.scatter(rf_max, dissipation[rf_max], label="rf_peaks", color="green")
        #     plt.scatter(
        #         nearest_peak,
        #         dissipation[nearest_peak],
        #         label="Nearest Peak",
        #         color="yellow",
        #         marker="*",
        #     )
        #     plt.scatter(
        #         crossings,
        #         difference[crossings],
        #         color="red",
        #         label="Crossings",
        #         zorder=5,
        #     )
        #     plt.scatter(
        #         nearest_crossing,
        #         difference[nearest_crossing],
        #         color="yellow",
        #         marker="*",
        #         label="Nearest Crossing",
        #         zorder=5,
        #     )

        #     if t_delta > 0:
        #         plt.axvline(t_delta, label="t_delta", color="black", linestyle="dotted")
        #     plt.axvline(adjusted_poi_6, label="poi_6", color="purple")
        #     plt.legend()
        #     plt.title(tail_class)
        #     plt.show()

        return adjusted_poi_6

    def predict(self, file_buffer, run_type=-1, start=-1, stop=-1, act=[None] * 6):
        # Load CSV data and drop unnecessary columns
        if isinstance(file_buffer, str):
            df = pd.read_csv(file_buffer)
        elif isinstance(file_buffer, pd.DataFrame):
            df = file_buffer
        # columns_to_drop = ["Date", "Time", "Ambient", "Temperature"]
        # if not all(col in df.columns for col in columns_to_drop):
        #     raise ValueError(
        #         f"[QModelPredict predict()]: Input data must contain the following columns: {columns_to_drop}"
        #     )

        # df = df.drop(columns=columns_to_drop)

        if not isinstance(file_buffer, str) and not isinstance(file_buffer, pd.DataFrame):
            if hasattr(file_buffer, "seekable") and file_buffer.seekable():
                # reset ByteIO buffer to beginning of stream
                file_buffer.seek(0)

            csv_headers = next(file_buffer)

            if isinstance(csv_headers, bytes):
                csv_headers = csv_headers.decode()

            if "Ambient" in csv_headers:
                csv_cols = (2, 4, 6, 7)
            else:
                csv_cols = (2, 3, 5, 6)

            file_data = np.loadtxt(
                file_buffer.readlines(), delimiter=",", skiprows=0, usecols=csv_cols
            )
            data_path = "QModel Passthrough"
            relative_time = file_data[:, 0]
            # temperature = file_data[:,1]
            resonance_frequency = file_data[:, 2]
            data = file_data[:, 3]

            emp_predictions = ModelData().IdentifyPoints(
                data_path=data_path,
                times=relative_time,
                freq=resonance_frequency,
                diss=data,
            )
        else:
            emp_predictions = ModelData().IdentifyPoints(file_buffer)
        emp_points = []
        start_bound = -1
        if isinstance(emp_predictions, list):
            for pt in emp_predictions:
                if isinstance(pt, int):
                    emp_points.append(pt)
                elif isinstance(pt, list):
                    max_pair = max(pt, key=lambda x: x[1])
                    emp_points.append(max_pair[0])
            start_bound = emp_points[0]

        ############################
        # MAIN PREDICTION PIPELINE #
        ############################

        file_buffer_2 = file_buffer
        if not isinstance(file_buffer_2, str) and not isinstance(file_buffer_2, pd.DataFrame):
            if hasattr(file_buffer_2, "seekable") and file_buffer_2.seekable():
                # reset ByteIO buffer to beginning of stream
                file_buffer_2.seek(0)
            else:
                # ERROR: 'file_buffer_2' must be 'BytesIO' type here, but it's not seekable!
                raise Exception(
                    "Cannot 'seek' stream prior to passing to 'QDataPipeline'."
                )
        else:
            # Assuming 'file_buffer_2' is a string to a file path, this will work fine as-is
            pass

        # Process data using QDataPipeline
        qdp = QDataPipeline(file_buffer_2)
        diss_raw = qdp.__dataframe__["Dissipation"]
        rel_time = qdp.__dataframe__["Relative_time"]
        qdp.preprocess(poi_filepath=None)
        diff_raw = qdp.__difference_raw__
        df = qdp.get_dataframe()
        f_names = self.__model__.feature_names
        df = df[f_names]
        d_data = xgb.DMatrix(df)

        results = self.__model__.predict(d_data)
        normalized_results = self.normalize(results)
        extracted_results = self.extract_results(normalized_results)

        # extracted_1 = emp_points[0]
        extracted_1 = np.argmax(extracted_results[1])
        extracted_2 = np.argmax(extracted_results[2])
        extracted_3 = np.argmax(extracted_results[3])
        extracted_4 = np.argmax(extracted_results[4])
        extracted_5 = np.argmax(extracted_results[5])
        extracted_6 = np.argmax(extracted_results[6])

        if not isinstance(emp_predictions, list):
            extracted_1 = np.argmax(extracted_results[1])

        if start > -1:
            extracted_1 = start
        if stop > -1:
            extracted_6 = stop

        if len(emp_points) <= 0:
            start_1 = extracted_1
            start_2 = extracted_2
            start_3 = extracted_3
            start_4 = extracted_4
            start_5 = extracted_5
            start_6 = extracted_6
        else:
            start_1 = emp_points[0]
            start_2 = emp_points[1]
            start_3 = emp_points[2]
            start_4 = emp_points[3]
            start_5 = emp_points[4]
            start_6 = emp_points[5]
        adj_1 = start_1
        poi_1 = self.adjustment_poi_1(
            initial_guess=start_1, dissipation_data=diss_raw)
        adj_6 = extracted_results[6]
        candidates_6 = self.find_and_sort_peaks(adj_6)
        if diff_raw.mean() < 0:
            poi_6 = start_6
        else:
            t_delta = qdp.find_time_delta()
            poi_6 = self.adjustment_poi_6(
                poi_6_guess=np.argmax(adj_6),
                candidates=candidates_6,
                dissipation=df["Dissipation"],
                difference=df["Difference"],
                rf=df["Resonance_Frequency"],
                signal=adj_6,
                t_delta=t_delta,
                actual=act,
            )
        adj_2, bounds_2 = self.adjust_predictions(
            prediction=extracted_results[2],
            rel_time=rel_time,
            poi_num=2,
            type=run_type,
            i=poi_1,
            j=poi_6,
        )
        adj_3, bounds_3 = self.adjust_predictions(
            prediction=extracted_results[3],
            rel_time=rel_time,
            poi_num=3,
            type=run_type,
            i=poi_1,
            j=poi_6,
        )
        adj_4, bounds_4 = self.adjust_predictions(
            prediction=extracted_results[4],
            rel_time=rel_time,
            poi_num=4,
            type=run_type,
            i=poi_1,
            j=poi_6,
        )
        adj_5, bounds_5 = self.adjust_predictions(
            prediction=extracted_results[5],
            rel_time=rel_time,
            poi_num=5,
            type=run_type,
            i=poi_1,
            j=poi_6,
        )

        candidates_1 = self.find_and_sort_peaks(extracted_results[1])
        candidates_2 = self.find_and_sort_peaks(adj_2)
        candidates_3 = self.find_and_sort_peaks(adj_3)
        candidates_4 = self.find_and_sort_peaks(adj_4)
        candidates_5 = self.find_and_sort_peaks(adj_5)

        # skip adjustment of point 6 when inverted (drop applied to outlet)

        poi_2 = self.adjustment_poi_2(
            initial_guess=start_2,
            dissipation_data=diss_raw,
            bounds=bounds_2,
            poi_1_estimate=poi_1,
        )
        poi_4 = self.adjustmet_poi_4(
            df=df,
            candidates=candidates_4,
            guess=extracted_4,
            actual=act,
            bounds=bounds_4,
            poi_1_guess=poi_1,
        )
        poi_5 = self.adjustmet_poi_5(
            df=df,
            candidates=candidates_5,
            guess=extracted_5,
            actual=act[4],
            bounds=bounds_5,
            poi_4_guess=poi_4,
            poi_6_guess=poi_6,
        )

        if poi_1 >= poi_2:
            poi_1 = adj_1
        poi_3 = np.argmax(adj_3)

        def sort_and_remove_point(arr, point):
            arr = np.array(arr)
            if len(arr) > MAX_GUESSES - 1:
                arr = arr[: MAX_GUESSES - 1]
            arr.sort()

            return arr[arr != point]

        candidates_list = [
            candidates_1,
            candidates_2,
            candidates_3,
            candidates_4,
            candidates_5,
            candidates_6,
        ]
        poi_list = [poi_1, poi_2, poi_3, poi_4, poi_5, poi_6]
        extracted_confidences = extracted_results[1:7]

        candidates = []

        for i in range(len(poi_list)):

            # Sort and remove point
            candidates_i = sort_and_remove_point(
                candidates_list[i], poi_list[i])
            filtered_points = candidates_i
            if i < 3:
                mean = np.mean(candidates_i)
                std_dev = np.std(candidates_i)
                threshold = 2
                # Filter points within the specified threshold
                filtered_points = [point for point in candidates_i if abs(
                    point - mean) <= threshold * std_dev]

            # Extract and sort confidence
            confidence_i = np.sort(np.array(extracted_confidences[i])[filtered_points])[
                ::-1
            ]

            # Insert POI at the start, remove the last element
            if len(candidates_i) > 1:
                candidates_i = np.insert(candidates_i, 0, poi_list[i])[:-1]
            else:
                candidates_i = np.insert(candidates_i, 0, poi_list[i])

            # Append to candidates list
            candidates.append((candidates_i, confidence_i))
        return candidates
