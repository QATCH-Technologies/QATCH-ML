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
from q_data_pipeline import QPartialDataPipeline, QDataPipeline
from q_image_clusterer import QClusterer
# from ModelData import ModelData
FILL_TYPE_R = {0: "full_fill", 1: "channel_1_partial",
               2: "channel_2_partial", 3: "no_fill"}
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


class PredictorUtils:
    @staticmethod
    def _validate_columns(df, required_columns):
        """Validate that the required columns exist in the DataFrame."""
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"[QModelPredict predict()]: Input data must contain the following columns: {required_columns}"
            )

    @staticmethod
    def _drop_columns(df, columns_to_drop):
        """Drop unnecessary columns from the DataFrame."""
        return df.drop(columns=columns_to_drop)

    @staticmethod
    def _reset_file_buffer(file_buffer):
        """Reset the file buffer to the beginning if it's seekable."""
        if hasattr(file_buffer, "seekable") and file_buffer.seekable():
            file_buffer.seek(0)

    @staticmethod
    def _parse_csv_headers(file_buffer):
        """Parse headers from the CSV file buffer."""
        csv_headers = next(file_buffer)
        if isinstance(csv_headers, bytes):
            csv_headers = csv_headers.decode()
        return csv_headers

    @staticmethod
    def _load_file_data(file_buffer, csv_cols):
        """Load numerical data from the file buffer using specified columns."""
        return np.loadtxt(
            file_buffer.readlines(), delimiter=",", skiprows=0, usecols=csv_cols
        )

    @staticmethod
    def _extract_emp_points(emp_predictions):
        """Extract EMP points from predictions."""
        emp_points = []
        if isinstance(emp_predictions, list):
            for pt in emp_predictions:
                if isinstance(pt, int):
                    emp_points.append(pt)
                elif isinstance(pt, list):
                    max_pair = max(pt, key=lambda x: x[1])
                    emp_points.append(max_pair[0])
        return emp_points

    @staticmethod
    def _process_file_buffer(file_buffer):
        """Process file buffer and return data for ModelData."""
        PredictorUtils._reset_file_buffer(file_buffer)
        csv_headers = PredictorUtils._parse_csv_headers(file_buffer)

        # Determine columns to use based on headers
        csv_cols = (2, 4, 6, 7) if "Ambient" in csv_headers else (2, 3, 5, 6)

        file_data = PredictorUtils._load_file_data(file_buffer, csv_cols)
        data_path = "QModel Passthrough"
        relative_time = file_data[:, 0]
        resonance_frequency = file_data[:, 2]
        data = file_data[:, 3]

        return {
            "data_path": data_path,
            "relative_time": relative_time,
            "resonance_frequency": resonance_frequency,
            "data": data
        }

    @staticmethod
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    @staticmethod
    def extract_results(results):
        return [list(group) for group in zip(*results)]

    @staticmethod
    def find_and_sort_peaks(signal):
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

    @staticmethod
    def _find_zero_slope_regions(data, threshold=1e-3, min_region_length=1):
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

    @staticmethod
    def setup_predictor(file_buffer):
        df = pd.read_csv(file_buffer)
        columns_to_drop = ["Date", "Time", "Ambient", "Temperature"]
        PredictorUtils._validate_columns(df, columns_to_drop)
        df = PredictorUtils._drop_columns(df, columns_to_drop)

        if not isinstance(file_buffer, str):
            file_data = PredictorUtils._process_file_buffer(file_buffer)
            emp_predictions = ModelData().IdentifyPoints(
                data_path=file_data["data_path"],
                times=file_data["relative_time"],
                freq=file_data["resonance_frequency"],
                diss=file_data["data"],
            )
        else:
            emp_predictions = ModelData().IdentifyPoints(file_buffer)
        emp_points = PredictorUtils._extract_emp_points(emp_predictions)
        start_bound = emp_points[0] if emp_points else -1

        return emp_points, start_bound

    @staticmethod
    def distribution_adjustment(prediction, rel_time, poi_num, type, i, j):
        import os
        import pickle

        # Normalize relative time
        rel_time = rel_time[i:j]
        rel_time_norm = PredictorUtils.normalize(rel_time)

        # Determine the root path
        if Architecture_found:
            relative_root = os.path.join(Architecture.get_path(), "QATCH")
        else:
            relative_root = os.getcwd()

        # Load models into a list
        models = []
        pickle_path = os.path.join(
            relative_root, "QModel/SavedModels/label_{}.pkl")
        for idx in range(3):
            with open(pickle_path.format(idx), "rb") as file:
                models.append(pickle.load(file))

        # Select bounds based on type
        bounds = None
        if type == 0:
            bounds = models[0]["Class_" + str(poi_num)]
        elif type == 1:
            bounds = models[1]["Class_" + str(poi_num)]
        elif type == 2:
            bounds = models[2]["Class_" + str(poi_num)]

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

    @staticmethod
    def adjustment_poi_1(initial_guess, dissipation_data):
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
        zero_slope_regions = PredictorUtils._find_zero_slope_regions(
            PredictorUtils.normalize(dissipation_data), threshold=0.0075, min_region_length=100
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

            if all_peaks.size > 0:
                distances_to_peaks = np.abs(all_peaks - adjusted_guess)
                nearest_peak = all_peaks[np.argmin(distances_to_peaks)]
                adjusted_guess = nearest_peak
            else:
                adjusted_guess = initial_guess

        return adjusted_guess

    @staticmethod
    def adjustment_poi_2(initial_guess, dissipation_data, bounds, poi_1_estimate):
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
        dissipation_data = PredictorUtils.normalize(dissipation_data)
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
        if len(valid_peaks) == 0:
            return initial_guess

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

    @staticmethod
    def adjustmet_poi_4(df, candidates, guess, actual, bounds, poi_1_guess):
        rf = PredictorUtils.normalize(df["Resonance_Frequency"])
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
            t = np.arange(len(rf))
            # Validate peaks and bounds
            if len(peaks) <= 0:
                return guess

            # Filter peaks within bounds
            filtered_peaks = [
                point for point in peaks if bounds[0] <= point <= bounds[1]]
            if not filtered_peaks:
                return guess
            # Ensure enough points for interpolation
            unique_peaks = np.unique(filtered_peaks)
            if len(unique_peaks) < 2:
                return guess

            # Validate rf and t arrays
            if len(t) != len(rf):
                return guess

            if np.any(np.isnan(rf)):
                return guess

            # Interpolate between peaks to create the upper envelope
            envelope_interpolator = interp1d(
                t[unique_peaks], rf[unique_peaks], kind="linear", fill_value="extrapolate"
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
                if len(filtered_peaks) == 0:
                    adjusted_point = guess
                else:
                    adjusted_point = filtered_peaks[0]
        return adjusted_point

    @staticmethod
    def adjustmet_poi_5(
        df, candidates, guess, actual, bounds, poi_4_guess, poi_6_guess
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
            zero_slope = PredictorUtils._find_zero_slope_regions(rf)
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
        return adjusted_point

    @staticmethod
    def adjustment_poi_6(
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
        rf = PredictorUtils.normalize(rf)
        dissipation = PredictorUtils.normalize(dissipation)
        difference = PredictorUtils.normalize(difference)
        signal = PredictorUtils.normalize(signal)

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
        if len(distances) != 0:
            nearest_peak = diff_peaks[np.argmin(distances)]
        else:
            nearest_peak = poi_6_guess

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

        return adjusted_poi_6

    @staticmethod
    def channel_fill_post_process(class_probabilities, file_buffer, true_fill):
        predicted_fill_type = FILL_TYPE_R[np.argmax(class_probabilities)]
        full_booster_1 = xgb.Booster()
        full_booster_2 = xgb.Booster()
        full_booster_3 = xgb.Booster()
        partial_1_booster = xgb.Booster()
        partial_2_booster = xgb.Booster()
        full_booster_1.load_model("QModel/SavedModels/QMultiType_0.json")
        full_booster_2.load_model("QModel/SavedModels/QMultiType_1.json")
        full_booster_3.load_model("QModel/SavedModels/QMultiType_2.json")
        partial_1_booster.load_model(
            "QModel/SavedModels/QMulti_Channel_1.json")
        partial_2_booster.load_model(
            "QModel/SavedModels/QMulti_Channel_2.json")
        qdp = QDataPipeline(file_buffer)
        qdp.preprocess()
        df = qdp.get_dataframe()

        f_names = full_booster_1.feature_names
        df = df[f_names]
        ddata = xgb.DMatrix(df)
        pfb1 = PredictorUtils.extract_results(full_booster_1.predict(ddata))
        pfb2 = PredictorUtils.extract_results(full_booster_2.predict(ddata))
        pfb3 = PredictorUtils.extract_results(full_booster_3.predict(ddata))
        ppb1 = PredictorUtils.extract_results(partial_1_booster.predict(ddata))
        ppb2 = PredictorUtils.extract_results(partial_2_booster.predict(ddata))
        # print(max(pfb1))
        # print(max(pfb2))
        # print(max(pfb3))
        # print(max(ppb1))
        # print(max(ppb2))

        # plt.figure()
        # plt.plot(df['Dissipation'])
        # plt.title(f'Pred: {predicted_fill_type} Was: {true_fill}')

        # plt.show()
        return predicted_fill_type


class QChannelPredictor():
    def __init__(self, file_buffer, fill_type: str, actual=None):
        self.actual = actual
        self._full_fill_predictors = {
            0: "QModel/SavedModels/QMultiType_0.json",
            1: "QModel/SavedModels/QMultiType_1.json",
            2: "QModel/SavedModels/QMultiType_2.json",
        }
        self._model_paths = {
            "channel_1_partial": "QModel/SavedModels/QMulti_Channel_1.json",
            "channel_2_partial": "QModel/SavedModels/QMulti_Channel_2.json",
            "full_fill": self._full_fill_predictors
        }

        self._model = xgb.Booster()
        self._prediction_results = ("not_set", None)
        self._process(file_buffer, fill_type)

    def get_prediction_results(self):
        return self._prediction_results

    def _process(self, file_buffer, fill_type: str):
        self._emp_points, self._start_bound = PredictorUtils.setup_predictor(
            file_buffer)
        if fill_type == "full_fill":
            full_fill_type = QClusterer(
                r"QModel/SavedModels/cluster.joblib").predict_label(file_buffer)
            full_fill_type = int(full_fill_type)
            self._model.load_model(
                self._model_paths["full_fill"][full_fill_type])
            self._prediction_results = fill_type, self._full_fill_process(
                file_buffer=file_buffer, full_fill_type=full_fill_type)
        elif fill_type == "no_fill":
            self._prediction_results = fill_type, self._no_fill_process()
        elif fill_type == "channel_1_partial":
            self._model.load_model(
                self._model_paths["channel_1_partial"])
            self._prediction_results = fill_type, self._channel_1_process(
                file_buffer=file_buffer)
        elif fill_type == "channel_2_partial":
            self._model.load_model(
                self._model_paths["channel_2_partial"])
            self._prediction_results = fill_type, self._channel_2_process(
                file_buffer=file_buffer)

    def _initialize_file_buffer(self, file_buffer):
        file_buffer_2 = file_buffer
        if not isinstance(file_buffer_2, str):
            if hasattr(file_buffer_2, "seekable") and file_buffer_2.seekable():
                file_buffer_2.seek(0)
            else:
                raise Exception(
                    "Cannot 'seek' stream prior to passing to 'QDataPipeline'.")
        return file_buffer_2

    def _process_qdp(self, file_buffer_2):
        qdp = QDataPipeline(file_buffer_2)
        raw_df = qdp.get_dataframe().copy()
        qdp.preprocess(poi_filepath=None)
        df = qdp.get_dataframe()
        return qdp, df, raw_df

    def _prepare_dmatrix(self, df):
        f_names = self._model.feature_names
        df = df[f_names]
        return xgb.DMatrix(df)

    def _get_predictions(self, d_data):
        results = self._model.predict(d_data)
        normalized_results = PredictorUtils.normalize(results)
        return PredictorUtils.extract_results(normalized_results)

    def _sort_and_filter_candidates(self, candidates_list, poi_list, extracted_confidences, max_guesses=5):
        def sort_and_remove_point(arr, point):
            arr = np.array(arr)
            if len(arr) > max_guesses - 1:
                arr = arr[: max_guesses - 1]
            arr.sort()
            return arr[arr != point]

        candidates = []
        for i, (candidates_i, poi) in enumerate(zip(candidates_list, poi_list)):

            candidates_i = sort_and_remove_point(candidates_i, poi)

            filtered_points = candidates_i
            if i < 3:
                mean = np.mean(candidates_i)
                std_dev = np.std(candidates_i)
                threshold = 2
                filtered_points = [
                    point for point in candidates_i if abs(point - mean) <= threshold * std_dev
                ]
            confidence_i = np.sort(np.array(extracted_confidences[i])[
                                   filtered_points])[::-1]

            if len(candidates_i) > 1:
                candidates_i = np.insert(candidates_i, 0, poi)[:-1]
            else:
                candidates_i = np.insert(candidates_i, 0, poi)

            candidates.append((candidates_i, confidence_i))
        return candidates

    def _full_fill_process(self, file_buffer, full_fill_type: int):
        file_buffer_2 = self._initialize_file_buffer(file_buffer)
        # Process data using QDataPipeline
        qdp, df, raw_df = self._process_qdp(file_buffer_2)
        d_data = self._prepare_dmatrix(df)
        extracted_results = self._get_predictions(d_data)

        diss_raw = raw_df["Dissipation"]
        rel_time = qdp.__dataframe__["Relative_time"]
        qdp.preprocess(poi_filepath=None)
        diff_raw = qdp.__difference_raw__

        # extracted_1 = emp_points[0]
        extracted_1 = np.argmax(extracted_results[1])
        extracted_2 = np.argmax(extracted_results[2])
        extracted_4 = np.argmax(extracted_results[4])
        extracted_5 = np.argmax(extracted_results[5])
        extracted_6 = np.argmax(extracted_results[6])

        if len(self._emp_points) <= 0:
            start_1 = extracted_1
            start_2 = extracted_2
            start_6 = extracted_6
        else:
            start_1 = self._emp_points[0]
            start_2 = self._emp_points[1]
            start_6 = self._emp_points[5]

        adj_1 = start_1
        poi_1 = PredictorUtils.adjustment_poi_1(
            initial_guess=start_1, dissipation_data=diss_raw)
        adj_6 = extracted_results[6]
        candidates_6 = PredictorUtils.find_and_sort_peaks(adj_6)
        if diff_raw.mean() < 0:
            poi_6 = start_6
        else:
            t_delta = qdp.find_time_delta()
            poi_6 = PredictorUtils.adjustment_poi_6(
                poi_6_guess=np.argmax(adj_6),
                candidates=candidates_6,
                dissipation=df["Dissipation"],
                difference=df["Difference"],
                rf=df["Resonance_Frequency"],
                signal=adj_6,
                t_delta=t_delta,
                actual=self.actual,
            )
        adj_2, bounds_2 = PredictorUtils.distribution_adjustment(
            prediction=extracted_results[2],
            rel_time=rel_time,
            poi_num=2,
            type=full_fill_type,
            i=poi_1,
            j=poi_6,
        )
        adj_3, bounds_3 = PredictorUtils.distribution_adjustment(
            prediction=extracted_results[3],
            rel_time=rel_time,
            poi_num=3,
            type=full_fill_type,
            i=poi_1,
            j=poi_6,
        )
        adj_4, bounds_4 = PredictorUtils.distribution_adjustment(
            prediction=extracted_results[4],
            rel_time=rel_time,
            poi_num=4,
            type=full_fill_type,
            i=poi_1,
            j=poi_6,
        )
        adj_5, bounds_5 = PredictorUtils.distribution_adjustment(
            prediction=extracted_results[5],
            rel_time=rel_time,
            poi_num=5,
            type=full_fill_type,
            i=poi_1,
            j=poi_6,
        )

        candidates_1 = PredictorUtils.find_and_sort_peaks(extracted_results[1])
        candidates_2 = PredictorUtils.find_and_sort_peaks(adj_2)
        candidates_3 = PredictorUtils.find_and_sort_peaks(adj_3)
        candidates_4 = PredictorUtils.find_and_sort_peaks(adj_4)
        candidates_5 = PredictorUtils.find_and_sort_peaks(adj_5)

        # skip adjustment of point 6 when inverted (drop applied to outlet)

        poi_2 = PredictorUtils.adjustment_poi_2(
            initial_guess=start_2,
            dissipation_data=diss_raw,
            bounds=bounds_2,
            poi_1_estimate=poi_1,
        )
        poi_4 = PredictorUtils.adjustmet_poi_4(
            df=df,
            candidates=candidates_4,
            guess=extracted_4,
            actual=self.actual,
            bounds=bounds_4,
            poi_1_guess=poi_1,
        )
        poi_5 = PredictorUtils.adjustmet_poi_5(
            df=df,
            candidates=candidates_5,
            guess=extracted_5,
            actual=self.actual,
            bounds=bounds_5,
            poi_4_guess=poi_4,
            poi_6_guess=poi_6,
        )

        if poi_1 >= poi_2:
            poi_1 = adj_1
        poi_3 = np.argmax(adj_3)

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
        return self._sort_and_filter_candidates(candidates_list, poi_list, extracted_confidences)

    def _channel_1_process(self, file_buffer):
        file_buffer_2 = self._initialize_file_buffer(file_buffer)
        # Process data using QDataPipeline
        qdp, df, raw_df = self._process_qdp(file_buffer_2)

        d_data = self._prepare_dmatrix(df)
        extracted_results = self._get_predictions(d_data)

        diss_raw = raw_df["Dissipation"]
        qdp.preprocess(poi_filepath=None)
        extracted_1 = np.argmax(extracted_results[1])
        extracted_2 = np.argmax(extracted_results[2])
        extracted_4 = np.argmax(extracted_results[4])

        if len(self._emp_points) <= 0:
            start_1 = extracted_1
            start_2 = extracted_2
        else:
            start_1 = self._emp_points[0]
            start_2 = self._emp_points[1]
        adj_1 = start_1
        poi_1 = PredictorUtils.adjustment_poi_1(
            initial_guess=start_1, dissipation_data=diss_raw)

        candidates_1 = PredictorUtils.find_and_sort_peaks(extracted_results[1])
        candidates_2 = PredictorUtils.find_and_sort_peaks(extracted_results[2])
        candidates_3 = PredictorUtils.find_and_sort_peaks(extracted_results[3])
        candidates_4 = PredictorUtils.find_and_sort_peaks(extracted_results[4])

        poi_3 = np.argmax(extracted_results[3])
        bounds_2 = (poi_1, poi_3)
        bounds_4 = (poi_3, len(extracted_results[4]))
        # skip adjustment of point 6 when inverted (drop applied to outlet)

        poi_2 = PredictorUtils.adjustment_poi_2(
            initial_guess=start_2,
            dissipation_data=diss_raw,
            bounds=bounds_2,
            poi_1_estimate=poi_1,
        )

        poi_4 = PredictorUtils.adjustmet_poi_4(
            df=df,
            candidates=candidates_4,
            guess=extracted_4,
            actual=self.actual,
            bounds=bounds_4,
            poi_1_guess=poi_1,
        )

        if poi_1 >= poi_2:
            poi_1 = adj_1

        candidates_list = [
            candidates_1,
            candidates_2,
            candidates_3,
            candidates_4,
        ]
        poi_list = [poi_1, poi_2, poi_3, poi_4]
        extracted_confidences = extracted_results[0:4]
        # plt.figure()
        # plt.plot(diss_raw, color="black")
        # plt.axvline(extracted_1, color='r')
        # plt.axvline(extracted_2, color='g')
        # plt.axvline(poi_3, color='b')
        # plt.axvline(extracted_4, color='y')
        # plt.show()
        # Increase the grid to 3 rows and 2 columns
        # fig, axs = plt.subplots(3, 2, figsize=(10, 8))

        # axs[0, 0].plot(extracted_results[1])
        # axs[0, 0].plot(PredictorUtils.normalize(diss_raw),
        #                color='grey', linestyle='dotted')
        # axs[0, 0].axvline(poi_1)
        # axs[0, 0].set_title("POI 1")

        # axs[0, 1].plot(extracted_results[2], color='r')
        # axs[0, 1].plot(PredictorUtils.normalize(diss_raw),
        #                color='grey', linestyle='dotted')
        # axs[0, 1].axvline(poi_2)
        # axs[0, 1].set_title("POI 2")

        # axs[1, 0].plot(extracted_results[3], color='g')
        # axs[1, 0].plot(PredictorUtils.normalize(diss_raw),
        #                color='grey', linestyle='dotted')
        # axs[1, 0].axvline(poi_3)
        # axs[1, 0].set_title("POI 3")
        # axs[1, 0].set_ylim(-5, 5)  # Limit y-axis to avoid extreme values

        # axs[1, 1].plot(extracted_results[4], color='m')
        # axs[1, 1].plot(PredictorUtils.normalize(diss_raw),
        #                color='grey', linestyle='dotted')
        # axs[1, 1].axvline(poi_4)
        # axs[1, 1].set_title("POI 4")

        # # Add the 5th subplot
        # # Assuming there's a 5th result
        # axs[2, 0].plot(extracted_results[0], color='b')
        # axs[2, 0].plot(PredictorUtils.normalize(diss_raw),
        #                color='grey', linestyle='dotted')
        # axs[2, 0].axvline(poi_1)
        # axs[2, 0].axvline(poi_2)
        # axs[2, 0].axvline(poi_3)
        # axs[2, 0].axvline(poi_4)
        # axs[2, 0].set_title("Not POI")
        # axs[2, 0].set_xlabel("X-axis label")  # Example label, adjust as needed
        # axs[2, 0].set_ylabel("Y-axis label")

        # # Remove the empty subplot (if any)
        # axs[2, 1].plot(diss_raw, color="black")
        # axs[2, 1].axvline(poi_1, color='r')
        # axs[2, 1].axvline(poi_2, color='g')
        # axs[2, 1].axvline(poi_3, color='b')
        # axs[2, 1].axvline(poi_4, color='y')

        # plt.tight_layout()
        # plt.show()
        # print("CHANNEL 1")
        # print(len(poi_list), len(extracted_confidences))
        return self._sort_and_filter_candidates(candidates_list, poi_list, extracted_confidences)

    def _channel_2_process(self, file_buffer):
        file_buffer_2 = self._initialize_file_buffer(file_buffer)
        # Process data using QDataPipeline
        qdp, df, raw_df = self._process_qdp(file_buffer_2)
        d_data = self._prepare_dmatrix(df)
        extracted_results = self._get_predictions(d_data)

        diss_raw = raw_df["Dissipation"]
        qdp.preprocess(poi_filepath=None)

        results = self._model.predict(d_data)
        normalized_results = PredictorUtils.normalize(results)
        extracted_results = PredictorUtils.extract_results(normalized_results)

        # extracted_1 = emp_points[0]
        extracted_1 = np.argmax(extracted_results[1])
        extracted_2 = np.argmax(extracted_results[2])
        extracted_4 = np.argmax(extracted_results[4])
        extracted_5 = np.argmax(extracted_results[5])

        if len(self._emp_points) <= 0:
            start_1 = extracted_1
            start_2 = extracted_2
        else:
            start_1 = self._emp_points[0]
            start_2 = self._emp_points[1]

        adj_1 = start_1
        poi_1 = PredictorUtils.adjustment_poi_1(
            initial_guess=start_1, dissipation_data=diss_raw)

        candidates_1 = PredictorUtils.find_and_sort_peaks(extracted_results[1])
        candidates_2 = PredictorUtils.find_and_sort_peaks(extracted_results[2])
        candidates_3 = PredictorUtils.find_and_sort_peaks(extracted_results[3])
        candidates_4 = PredictorUtils.find_and_sort_peaks(extracted_results[4])
        candidates_5 = PredictorUtils.find_and_sort_peaks(extracted_results[5])

        # skip adjustment of point 6 when inverted (drop applied to outlet)
        poi_3 = np.argmax(extracted_results[5])
        bounds_2 = (poi_1, poi_3)
        bounds_4 = (poi_3, extracted_5)
        poi_2 = PredictorUtils.adjustment_poi_2(
            initial_guess=start_2,
            dissipation_data=diss_raw,
            bounds=bounds_2,
            poi_1_estimate=poi_1,
        )

        poi_4 = PredictorUtils.adjustmet_poi_4(
            df=df,
            candidates=candidates_4,
            guess=extracted_4,
            actual=self.actual,
            bounds=bounds_4,
            poi_1_guess=poi_1,
        )
        bounds_5 = (poi_4, len(extracted_results[5]))
        poi_5 = PredictorUtils.adjustmet_poi_5(
            df=df,
            candidates=candidates_5,
            guess=extracted_5,
            actual=self.actual,
            bounds=bounds_5,
            poi_4_guess=poi_4,
            poi_6_guess=len(extracted_results[5]),
        )

        if poi_1 >= poi_2:
            poi_1 = adj_1

        candidates_list = [
            candidates_1,
            candidates_2,
            candidates_3,
            candidates_4,
            candidates_5,
        ]

        poi_list = [poi_1, poi_2, poi_3, poi_4, poi_5]
        extracted_confidences = extracted_results[1:6]
        return self._sort_and_filter_candidates(candidates_list, poi_list, extracted_confidences)

    def _no_fill_process(self):
        return None


class QPredictor:
    def __init__(self, model_path: str = "QModel/SavedModels/partial_qmm.json"):
        self._partial_model = xgb.Booster()
        self._partial_model.load_model(model_path)

    def predict(self, file_buffer: str = None, true_fill: str = None):
        # Extract and preprocess features
        qpd = QPartialDataPipeline(file_buffer)
        qpd.preprocess()
        features = qpd.get_features()
        features_df = pd.DataFrame([features]).fillna(0)
        f_names = self._partial_model.feature_names
        df = features_df[f_names]
        d_data = xgb.DMatrix(df)
        class_probabilities = self._partial_model.predict(d_data)
        predicted_fill_type = PredictorUtils.channel_fill_post_process(
            class_probabilities=class_probabilities, file_buffer=file_buffer, true_fill=true_fill)

        channel_predictor = QChannelPredictor(
            file_buffer=file_buffer, fill_type=predicted_fill_type)
        return channel_predictor.get_prediction_results()
