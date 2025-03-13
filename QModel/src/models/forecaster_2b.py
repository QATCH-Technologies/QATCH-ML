from scipy.signal import detrend
from scipy.signal import butter, filtfilt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from scipy.signal import hilbert, detrend
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
import ruptures as rpt

FEATURES = [
    'Dissipation',
    'Dissipation_rolling_mean',
    'Dissipation_rolling_median',
    'Dissipation_ewm',
    'Dissipation_rolling_std',
    'Dissipation_diff',
    'Dissipation_pct_change',
    'Dissipation_ratio_to_mean',
    'Dissipation_ratio_to_ewm',
    'Dissipation_envelope'
]
NUM_CLASSES = 4
TARGET = "Fill"
DOWNSAMPLE_FACTOR = 5
SPIN_UP_TIME = (1.2, 1.4)
BASELINE_WINDOW = 100


class QForecasterDataprocessor:
    @staticmethod
    def load_content(data_dir: str) -> list:
        """
        Walk through data_dir and return a list of tuples.
        Each tuple contains the path to a CSV file (excluding those ending in "_poi.csv" or "_lower.csv")
        and its corresponding POI file (with '_poi.csv' replacing '.csv').
        """
        print(f"[INFO] Loading content from {data_dir}")
        loaded_content = []
        for root, _, files in tqdm(os.walk(data_dir), desc='Loading files...'):
            for f in files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    poi_file = f.replace(".csv", "_poi.csv")
                    loaded_content.append(
                        (os.path.join(root, f), os.path.join(root, poi_file))
                    )
        return loaded_content

    @staticmethod
    def find_time_delta(df: pd.DataFrame) -> int:
        """
        Compute the first index at which the difference in Relative_time
        deviates significantly from its expanding rolling mean.
        Returns -1 if no significant change is found.
        """
        time_df = pd.DataFrame()
        time_df["Delta"] = df["Relative_time"].diff()
        threshold = 0.032
        rolling_avg = time_df["Delta"].expanding(min_periods=2).mean()
        time_df["Significant_change"] = (
            time_df["Delta"] - rolling_avg).abs() > threshold
        change_indices = time_df.index[time_df["Significant_change"]].tolist()
        return change_indices[0] if change_indices else -1

    @staticmethod
    def reassign_region(fill):
        """
        Reassign numeric fill values to string region labels.
        """
        if fill == 0:
            return 'no_fill'
        elif fill in [1, 2, 3]:
            return 'init_fill'
        elif fill == 4:
            return 'ch_1'
        elif fill == 5:
            return 'ch_2'
        elif fill == 6:
            return 'full_fill'
        else:
            return fill  # fallback if needed

    @staticmethod
    def compute_additional_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute additional features for the Dissipation column and live compute the difference curve.
        This includes rolling statistics, differences, ratios, signal envelope, and a Difference curve.
        An FFT is computed for the Resonance_Frequency data and then detrended.
        Additionally, this function computes several signal processing features to exaggerate
        changes in the signal:
        - First and second derivatives (finite differences)
        - High-pass filtered signal (signal minus its rolling mean)
        - Derivative-of-Gaussian filtering for smoothed differentiation
        - Wavelet transform detail coefficients (using a level-1 DWT)
        - Adaptive change detection (normalized first derivative)
        """
        try:
            required_columns = ["Dissipation",
                                "Resonance_Frequency", "Relative_time"]
            if not all(column in df.columns for column in required_columns):
                raise ValueError(
                    f"Input DataFrame must contain the following columns: {required_columns}"
                )

            window = 10
            span = 10
            run_length = len(df)
            # Ensure window_length is at least 3
            window_length = max(3, int(np.ceil(0.01 * run_length)))

            # Adjust window_length if it's larger than the Dissipation data size
            diss_length = len(df['Dissipation'])
            if window_length > diss_length:
                window_length = diss_length if diss_length % 2 == 1 else diss_length - 1

            polyorder = 2 if window_length > 2 else 1

            # Basic rolling statistics and differences
            df['Dissipation_rolling_mean'] = df['Dissipation'].rolling(
                window=window, min_periods=1).mean()
            df['Resonance_Frequency_rolling_mean'] = df['Resonance_Frequency'].rolling(
                window=window, min_periods=1).mean()
            df['Dissipation_rolling_median'] = df['Dissipation'].rolling(
                window=window, min_periods=1).median()
            df['Dissipation_ewm'] = df['Dissipation'].ewm(
                span=span, adjust=False).mean()
            df['Dissipation_rolling_std'] = df['Dissipation'].rolling(
                window=window, min_periods=1).std()
            df['Dissipation_diff'] = df['Dissipation'].diff()
            df['Dissipation_pct_change'] = df['Dissipation'].pct_change()
            df['Dissipation_ratio_to_mean'] = df['Dissipation'] / \
                df['Dissipation_rolling_mean']
            df['Dissipation_ratio_to_ewm'] = df['Dissipation'] / \
                df['Dissipation_ewm']
            df['Dissipation_envelope'] = np.abs(
                hilbert(df['Dissipation'].values))

            # Existing Difference curve based on Relative_time windowing
            if "Difference" not in df.columns:
                df['Difference'] = [0] * len(df)

            xs = df["Relative_time"]
            i = next((x for x, t in enumerate(xs) if t > 0.5), None)
            j = next((x for x, t in enumerate(xs) if t > 2.5), None)

            if i is not None and j is not None:
                avg_resonance_frequency = df["Resonance_Frequency"][i:j].mean()
                avg_dissipation = df["Dissipation"][i:j].mean()

                df["ys_diss"] = (df["Dissipation"] - avg_dissipation) * \
                    avg_resonance_frequency / 2
                df["ys_freq"] = avg_resonance_frequency - \
                    df["Resonance_Frequency"]
                difference_factor = 3
                df["Difference"] = df["ys_freq"] - \
                    difference_factor * df["ys_diss"]

            # -------------------------------
            # Additional Signal Processing Features
            # -------------------------------

            # 1. Derivative-of-Gaussian Filtering: Smooth then differentiate.
            from scipy.ndimage import gaussian_filter1d
            sigma = 2  # Adjust sigma as needed
            df['Dissipation_DoG'] = gaussian_filter1d(
                df['Dissipation'], sigma=sigma, order=1)

            # 2. Compute a baseline for the DoG signal using a rolling median.
            baseline_window = max(3, int(np.ceil(0.05 * len(df))))
            df['DoG_baseline'] = df['Dissipation_DoG'].rolling(
                window=baseline_window, center=True, min_periods=1).median()

            # 3. Create a baseline-corrected DoG signal (shift from baseline)
            df['DoG_shift'] = df['Dissipation_DoG'] - df['DoG_baseline']

            # 4. Apply Kalman filtering to the DoG_shift signal for additional smoothing.
            from pykalman import KalmanFilter
            initial_state = df['DoG_shift'].iloc[0]
            kf = KalmanFilter(
                transition_matrices=[1],
                observation_covariance=1,
                observation_matrices=[1],
                initial_state_mean=initial_state,
                initial_state_covariance=1,
                transition_covariance=0.01
            )
            state_means, _ = kf.filter(df['DoG_shift'].values)
            df['DoG_shift_kalman'] = state_means.flatten()

            # 5. Compute the upper envelope of the DoG_shift signal.
            # Use the Hilbert transform to get the analytic amplitude (i.e. the envelope)
            df['DoG_shift_envelope'] = np.abs(hilbert(df['DoG_shift'].values))

            # 6. Smooth the computed upper envelope.
            # Apply a separate Kalman filter to the envelope
            initial_env = df['DoG_shift_envelope'].iloc[0]
            kf_env = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=initial_env,
                initial_state_covariance=1,
                observation_covariance=1,
                transition_covariance=0.01
            )
            env_state_means, _ = kf_env.filter(df['DoG_shift_envelope'].values)
            df['DoG_shift_upper_envelope'] = env_state_means.flatten()

            # 7. Optionally, assign the smoothed envelope to a column for downstream use.
            df['Dissipation_Envelope'] = df['DoG_shift_upper_envelope']

            # 8. PCA-based high-energy outlier detection on non-envelope data.
            from sklearn.decomposition import PCA
            features = ['Dissipation', 'Dissipation_DoG',
                        'DoG_shift', 'Difference']
            X = df[features].fillna(0).values
            pca = PCA(n_components=len(features))
            X_pca = pca.fit_transform(X)
            energy = np.sum(X_pca**2, axis=1)
            df['PCA_Energy'] = energy
            threshold = np.mean(energy) + 2 * np.std(energy)
            df['High_Energy_Outlier'] = energy > threshold
            # 1a. Incorporate SVM into the Dissipation_DoG signal.
            # Here we use a One-Class SVM to detect potential anomalies in the DoG signal.
            from sklearn.svm import OneClassSVM
            X_dog = df['DoG_shift'].values.reshape(-1, 1)
            ocsvm = OneClassSVM(nu=0.05, kernel='rbf',
                                gamma='scale', shrinking=False)
            df['DoG_SVM_Label'] = ocsvm.fit_predict(X_dog)
            df['DoG_SVM_Score'] = ocsvm.decision_function(X_dog)

            return df
        except Exception as e:
            raise RuntimeError(f"Error computing features: {e}")

    @staticmethod
    def _process_fill(df: pd.DataFrame, poi_file: str) -> pd.DataFrame:
        """
        Helper to process fill information from the poi_file.
        Reads the poi CSV (with no header) and adds a Fill column to df.
        If the poi file does not contain a header, the method treats the first column
        as change indices (adding 1 to Fill from that index onward).
        Optionally, if check_unique is True, the method returns None when the number
        of unique fill values is not 7.
        """
        fill_df = pd.read_csv(poi_file, header=None)
        if "Fill" in fill_df.columns:
            df["Fill"] = fill_df["Fill"]
        else:
            df["Fill"] = 0
            change_indices = sorted(fill_df.iloc[:, 0].values)
            for idx in change_indices:
                df.loc[idx:, "Fill"] += 1

        df["Fill"] = pd.Categorical(df["Fill"]).codes
        df["Fill"] = df["Fill"].apply(QForecasterDataprocessor.reassign_region)
        mapping = {'no_fill': 0, 'init_fill': 1,
                   'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
        df["Fill"] = df["Fill"].map(mapping)
        return df

    @staticmethod
    def load_and_preprocess_data(data_dir: str, num_datasets: int):
        runs = []
        content = QForecasterDataprocessor.load_content(data_dir)
        random.shuffle(content)

        if num_datasets < len(content):
            content = content[:num_datasets]

        for file, poi_file in content:
            df = pd.read_csv(file)
            required_cols = ["Relative_time", "Dissipation"]
            if df.empty or not all(col in df.columns for col in required_cols):
                continue

            df = df[required_cols]

            try:
                df = QForecasterDataprocessor._process_fill(
                    df, poi_file=poi_file)
            except FileNotFoundError:
                df = None
            if df is None:
                continue

            df = df[df["Relative_time"] >= random.uniform(
                SPIN_UP_TIME[0], SPIN_UP_TIME[1])]
            df = df.iloc[::DOWNSAMPLE_FACTOR]
            df = df.reset_index()
            init_fill_point = QForecasterDataprocessor.init_fill_point(
                df, BASELINE_WINDOW, 10)
            df = df.iloc[init_fill_point:]
            if df is None or df.empty:
                continue
            df.loc[df['Fill'] == 0, 'Fill'] = 1
            # Method 2 (in-place subtraction):
            df['Fill'] -= 1
            df = QForecasterDataprocessor.compute_additional_features(df)
            df.reset_index(inplace=True)

            runs.append(df)

        training_data = pd.concat(runs).sort_values(
            "Relative_time").reset_index(drop=True)
        training_data.drop(columns=['Relative_time'], inplace=True)
        return training_data

    @staticmethod
    def viterbi_decode(prob_matrix, transition_matrix):
        T, N = prob_matrix.shape
        dp = np.full((T, N), -np.inf)
        backpointer = np.zeros((T, N), dtype=int)
        dp[0, 0] = np.log(prob_matrix[0, 0])
        for t in range(1, T):
            for j in range(N):
                allowed_prev = [0] if j == 0 else [j-1, j]
                best_state = allowed_prev[0]
                best_score = dp[t-1, best_state] + \
                    np.log(transition_matrix[best_state, j])
                for i in allowed_prev:
                    if transition_matrix[i, j] <= 0:
                        continue
                    score = dp[t-1, i] + np.log(transition_matrix[i, j])
                    if score > best_score:
                        best_score = score
                        best_state = i
                dp[t, j] = np.log(prob_matrix[t, j]) + best_score
                backpointer[t, j] = best_state
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(dp[T-1])
        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]
        return best_path

    @staticmethod
    def compute_dynamic_transition_matrix(training_data, num_states=5, smoothing=1e-6):
        states = training_data["Fill"].values
        transition_counts = np.zeros((num_states, num_states))
        for i in range(num_states):
            transition_counts[i, i] = smoothing
            if i + 1 < num_states:
                transition_counts[i, i+1] = smoothing
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            if next_state == current_state or next_state == current_state + 1:
                transition_counts[current_state, next_state] += 1
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return transition_counts / row_sums

    @staticmethod
    def plot_fft_windows(signal, window_size=100, step_size=50):
        """
        Applies FFT to each detrended window of the signal and plots the FFT magnitude spectrum.

        Parameters:
            signal (np.ndarray): 1D array containing the signal data.
            window_size (int): Number of samples in each window.
            step_size (int): Number of samples to move the window (allows for overlap).

        Notes:
            This function loops over the signal in windows, detrends each window, computes the FFT,
            and then plots the magnitude spectrum for the positive frequencies.
        """
        n = len(signal)

        # Loop through the signal using a sliding window
        for start in range(0, n, step_size):
            end = start + window_size
            if end > n:
                end = n

            # Extract and detrend the current window slice
            window_slice = signal[start:end]
            detrended_slice = detrend(window_slice)

            # Compute FFT of the detrended slice
            fft_result = np.fft.fft(detrended_slice)
            fft_magnitude = np.abs(fft_result)

            # Only consider positive frequencies
            half = len(fft_magnitude) // 2
            fft_magnitude = fft_magnitude[:half]
            freq_axis = np.fft.fftfreq(end - start, d=1)[:half]

            # Plot the FFT magnitude spectrum for the current window
            plt.figure(figsize=(10, 5))
            plt.plot(freq_axis, fft_magnitude)
            plt.xlabel("Frequency")
            plt.ylabel("Magnitude")
            plt.title(
                f"FFT Magnitude Spectrum (Window starting at index {start})")
            plt.grid(True)
            plt.show()

    @staticmethod
    def init_fill_point(df: pd.DataFrame) -> int:
        """
        Finds the first major baseline shift in the 'Dissipation_DoG' column.

        A major baseline shift is defined as the first data point after 3.0 seconds (in the 
        'Relative_time' column) where the absolute difference from the baseline (mean of the 
        values prior to 3.0 seconds) exceeds a dynamically calculated threshold. The threshold 
        is determined as the maximum of:
            - 2 times the standard deviation of the baseline, and
            - 1000 times the absolute value of the baseline mean.

        This dynamic approach helps capture shifts like those expected from ~1e-16 to ~1e-13.
        If the total time is less than 3.0 seconds or if no such shift is detected, 1 is returned.

        Parameters:
            df (pd.DataFrame): DataFrame with columns 'Dissipation_DoG' and 'Relative_time'

        Returns:
            int: The index of the first major baseline shift, or 1 if conditions are not met.
        """
        # Check if at least 3.0 seconds have passed.
        if df['Relative_time'].max() < 3.0:
            print('[INIT_FILL] Waiting for 3.0s')
            return -1

        # Use data before 3.0 seconds as the baseline.
        baseline_mask = df['Relative_time'] < 2.0
        baseline_values = df.loc[baseline_mask, 'Dissipation_DoG']

        if baseline_values.empty:
            print('[INIT_FILL] No baseline')
            return -1

        baseline_mean = baseline_values.mean()
        baseline_std = baseline_values.std()

        # Define dynamic threshold: maximum of (2 * std, 1000 * |mean|).
        dynamic_threshold = max(2 * baseline_std, abs(baseline_mean) * 750)
        # Fallback if dynamic threshold is 0.
        if dynamic_threshold == 0:
            dynamic_threshold = 0.001

        # Find the first row (with time >= 3.0 sec) where the absolute difference exceeds the dynamic threshold.
        shift_mask = (df['Relative_time'] >= 2.0) & (
            abs(df['Dissipation_DoG'] - baseline_mean) >= dynamic_threshold)

        if not shift_mask.any():
            print('[INIT_FILL] No shifts')
            return -1
        else:
            first_shift_idx = df[shift_mask].index[0]
            return int(first_shift_idx)

    @staticmethod
    def ch1_point(df: pd.DataFrame, init_fill_idx: int) -> int:
        """
        Finds the first location after the initial fill shift point where the signal drops back to a sustained baseline.

        The baseline is computed from the 100 data points immediately preceding the init_fill shift point.
        "Dropping back to baseline" is defined as the first point after the shift where the absolute difference
        between 'Dissipation_DoG' and the baseline mean is less than a dynamic threshold, and this condition is
        sustained for a set number of consecutive data points.

        The dynamic threshold is defined as the maximum of:
            - 2 times the baseline standard deviation, and
            - 1000 times the absolute value of the baseline mean.
        A fallback minimal threshold of 0.001 is used if the calculated threshold is zero.

        A sustained baseline is defined here as having at least 10 consecutive data points (starting from the candidate)
        that remain within the threshold of the baseline.

        Parameters:
            df (pd.DataFrame): DataFrame with columns 'Dissipation_DoG' and 'Relative_time'
            init_fill_idx (int): The index (from init_fill_point) where the initial shift occurs

        Returns:
            int: The index of the first point after the shift where the signal returns to a sustained baseline,
                 or the last index of the DataFrame if none is found.
        """
        # If init_fill_idx indicates no valid shift was found, return 1.
        if init_fill_idx == 1:
            return 1

        # Convert init_fill_idx to a positional index in case df.index is not a simple range.
        try:
            pos_init = df.index.get_loc(init_fill_idx)
        except KeyError:
            pos_init = init_fill_idx

        # Compute the baseline using 100 points before the init_fill shift.
        baseline_start = max(0, pos_init - 100)
        baseline_values = df['Dissipation_DoG'].iloc[baseline_start:pos_init]
        if baseline_values.empty:
            return 1

        baseline_mean = baseline_values.mean()
        baseline_std = baseline_values.std()
        dynamic_threshold = max(2 * baseline_std, abs(baseline_mean) * 1000)
        if dynamic_threshold == 0:
            dynamic_threshold = 0.001

        # Define the required number of consecutive points to consider the baseline "sustained."
        sustain_count = 10

        # Iterate over rows after the init_fill shift.
        for pos in range(pos_init + 1, len(df)):
            current_val = df['Dissipation_DoG'].iloc[pos]
            # Check if the candidate point is within the dynamic threshold.
            if abs(current_val - baseline_mean) < dynamic_threshold:
                # Check for sustained baseline: candidate and the next sustain_count-1 points.
                sustained = True
                for check_pos in range(pos, min(pos + sustain_count, len(df))):
                    if abs(df['Dissipation_DoG'].iloc[check_pos] - baseline_mean) >= dynamic_threshold:
                        sustained = False
                        break
                if sustained:
                    return int(df.index[pos])

        # If no sustained baseline is found, return the last index.
        return int(df.index[-1])

    @staticmethod
    def ch2_point(df, init_fill_point, ch1_fill_point, anomaly_indices):

        if ch1_fill_point == -1 or init_fill_point == -1:
            return -1

        # Compute expected Relative_time for ch2_fill_point
        time_diff = df.loc[ch1_fill_point, "Relative_time"] - \
            df.loc[init_fill_point, "Relative_time"]
        expected_relative_time = df.loc[ch1_fill_point,
                                        "Relative_time"] + 5 * time_diff

        # Check if the dataframe has reached the expected Relative_time
        max_relative_time = df["Relative_time"].max()
        if max_relative_time < expected_relative_time:
            return -1

        expected_index = df[df["Relative_time"]
                            >= expected_relative_time].index.min()

        # If there are any anomalous indices, select those that are after the expected_index.
        # if anomaly_indices is not None:
        #     anomalies_after_expected = [
        #         idx for idx in anomaly_indices if idx > expected_index]
        #     if anomalies_after_expected:
        #         nearest_anomaly = min(
        #             anomalies_after_expected, key=lambda x: x - expected_index)
        #         return nearest_anomaly

        # If no valid anomaly is found, return the expected_index.
        return expected_index

    @staticmethod
    def ch3_point(df, init_fill_point, ch1_fill_point, anomaly_indices):

        if ch1_fill_point == -1 or init_fill_point == -1:
            return -1

        # Compute expected Relative_time for ch2_fill_point
        time_diff = df.loc[ch1_fill_point, "Relative_time"] - \
            df.loc[init_fill_point, "Relative_time"]
        expected_relative_time = df.loc[ch1_fill_point,
                                        "Relative_time"] + 10 * time_diff

        # Check if the dataframe has reached the expected Relative_time
        max_relative_time = df["Relative_time"].max()
        if max_relative_time < expected_relative_time:
            return -1

        expected_index = df[df["Relative_time"]
                            >= expected_relative_time].index.min()

        # if anomaly_indices is not None:
        #     anomalies_after_expected = [
        #         idx for idx in anomaly_indices if idx > expected_index]
        #     if anomalies_after_expected is not None:
        #         nearest_anomaly = min(
        #             anomaly_indices, key=lambda x: abs(x - expected_index))
        #         return nearest_anomaly

        # If no anomalies are found, return the expected_index.
        return expected_index

    def exit_point(df, ch3_fill_point, threshold):
        if ch3_fill_point == -1:
            return -1

        if ch3_fill_point < 0 or ch3_fill_point >= len(df) - 1:
            return -1
        for i in range(ch3_fill_point + 1, len(df)):
            change = df.loc[i, "Difference"] - df.loc[i - 1, "Difference"]
            if abs(change) >= threshold:
                return i

        return -1
###############################################################################
# Trainer Class: Handles model training, hyperparameter tuning, and saving.
###############################################################################


class QForecasterTrainer:
    def __init__(self, features, target='Fill', save_dir=None):
        """
        Args:
            numerical_features (list): List of numerical feature names.
            target (str): Target column name.
            save_dir (str): Directory to save trained objects.
        """
        self.features = features
        self.target = target
        self.save_dir = save_dir

        self.model = None
        self.preprocessors = None
        self.params = None
        self.transition_matrix = None

    def _build_preprocessors(self, X):
        """Fit and return preprocessors for numerical data."""
        num_imputer = SimpleImputer(strategy='mean')
        X_num = num_imputer.fit_transform(X[self.features])
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
        preprocessors = {'num_imputer': num_imputer, 'scaler': scaler}
        X_processed = X_num
        return X_processed, preprocessors

    def _tune_parameters(self, dtrain, base_params, max_evals=10):
        """Tune XGBoost hyperparameters using Hyperopt."""
        space = {
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1)
        }

        def objective(hyperparams):
            params = base_params.copy()
            params['max_depth'] = int(hyperparams['max_depth'])
            params['min_child_weight'] = int(hyperparams['min_child_weight'])
            params['learning_rate'] = hyperparams['learning_rate']
            params['subsample'] = hyperparams['subsample']
            params['colsample_bytree'] = hyperparams['colsample_bytree']

            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=200,
                nfold=5,
                metrics={'auc'},
                early_stopping_rounds=10,
                seed=42,
                verbose_eval=False
            )
            best_score = cv_results['test-auc-mean'].max()
            best_rounds = len(cv_results)
            return {'loss': -best_score, 'status': STATUS_OK, 'num_rounds': best_rounds}

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest,
                    max_evals=max_evals, trials=trials)
        best['max_depth'] = int(best['max_depth'])
        best['min_child_weight'] = int(best['min_child_weight'])
        params = base_params.copy()
        params.update(best)
        best_trial = min(trials.results, key=lambda x: x['loss'])
        optimal_rounds = best_trial['num_rounds']
        return params, optimal_rounds

    def train_model_native(self, training_data, tune=True):
        """
        Train an XGBoost model on the provided training data.

        Returns:
            model: Trained XGBoost model.
            preprocessors (dict): Fitted preprocessors.
            params (dict): Model parameters.
        """
        features = self.features
        X = training_data[features].copy()
        y = training_data[self.target].values

        X_processed, preprocessors = self._build_preprocessors(X)
        dtrain = xgb.DMatrix(X_processed, label=y)

        base_params = {
            'objective': 'multi:softprob',
            'num_class': NUM_CLASSES,
            'eval_metric': 'auc',
            'seed': 42,
            'device': 'cuda',
        }

        if tune:
            params, optimal_rounds = self._tune_parameters(dtrain, base_params)
        else:
            params = base_params.copy()
            params.update({'max_depth': 5, 'learning_rate': 0.1})
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=200,
                nfold=5,
                metrics={'auc'},
                early_stopping_rounds=10,
                seed=42,
                verbose_eval=False
            )
            optimal_rounds = len(cv_results)

        model = xgb.train(params, dtrain, num_boost_round=optimal_rounds)
        return model, preprocessors, params

    def _apply_preprocessors(self, X, preprocessors):
        """
        Apply loaded preprocessors to input data.
        """
        X_copy = X.copy()
        X_num = preprocessors['num_imputer'].transform(
            X_copy[self.features])
        X_num = preprocessors['scaler'].transform(X_num)
        return X_num

    def predict_native(self, model: xgb.Booster, preprocessors, X):
        """
        Generate predictions using a loaded XGBoost model.
        """
        X_processed = self._apply_preprocessors(X, preprocessors)
        dmatrix = xgb.DMatrix(X_processed)
        return model.predict(dmatrix)

    def train_model(self, training_data, tune=True):

        self.transition_matrix = QForecasterDataprocessor.compute_dynamic_transition_matrix(
            training_data)

        self.model, self.preprocessors, self.params = self.train_model_native(
            training_data, tune=tune)
        print("[INFO] Base model training complete.")

    def save_models(self):
        """
        Save base models, meta classifier, preprocessors, parameters, and transition matrices.
        """
        if not self.save_dir:
            raise ValueError("Save directory not specified.")
        os.makedirs(self.save_dir, exist_ok=True)

        # Save short model and its objects.
        self.model.save_model(os.path.join(
            self.save_dir, "model.json"))
        with open(os.path.join(self.save_dir, "preprocessors.pkl"), "wb") as f:
            pickle.dump(self.preprocessors, f)
        with open(os.path.join(self.save_dir, "params.pkl"), "wb") as f:
            pickle.dump(self.params, f)
        with open(os.path.join(self.save_dir, "transition_matrix.pkl"), "wb") as f:
            pickle.dump(self.transition_matrix, f)

        print("[INFO] All models and associated objects have been saved successfully.")

    def load_models(self):
        if not self.save_dir:
            raise ValueError("Save directory not specified.")

        self.model = xgb.Booster()
        self.model.load_model(os.path.join(
            self.save_dir, "model.json"))
        with open(os.path.join(self.save_dir, "preprocessors.pkl"), "rb") as f:
            self.preprocessors = pickle.load(f)
        with open(os.path.join(self.save_dir, "params.pkl"), "rb") as f:
            self.params = pickle.load(f)
        with open(os.path.join(self.save_dir, "transition_matrix.pkl"), "rb") as f:
            self.transition_matrix = pickle.load(f)

        print("[INFO] All models and associated objects have been loaded successfully.")


class QForecasterPredictor:
    """Predictor class for QForecaster that handles model loading, data accumulation, and prediction.

    This class loads a pre-trained XGBoost model along with its associated preprocessors and transition
    matrix. It provides methods to apply preprocessing to input data, generate predictions using the
    model and Viterbi decoding, and maintain a history of predictions to assess stability.
    """

    def __init__(
        self,
        numerical_features: List[str] = FEATURES,
        target: str = 'Fill',
        save_dir: Optional[str] = None,
        batch_threshold: int = 60
    ) -> None:
        """Initializes the QForecasterPredictor.

        Args:
            numerical_features (List[str], optional): List of feature names used for numerical processing.
                Defaults to FEATURES.
            target (str, optional): The target variable name. Defaults to 'Fill'.
            save_dir (Optional[str], optional): Directory path from which to load model files and preprocessors.
                Defaults to None.
            batch_threshold (int, optional): The rate at which to process batches of data. Defaults to 60.
        """
        self.numerical_features: List[str] = numerical_features
        self.target: str = target
        self.save_dir: Optional[str] = save_dir
        self.batch_threshold: int = batch_threshold

        self.model: Optional[xgb.Booster] = None
        self.preprocessors: Optional[Dict[str, Any]] = None
        self.transition_matrix: Optional[np.ndarray] = None

        self.accumulated_data: Optional[pd.DataFrame] = None
        self.batch_num: int = 0
        self.prediction_history: Dict[int, List[Tuple[int, int]]] = {}

        self.init_fill_points = []
        self.ch1_fill_points = []
        self.ch2_fill_points = []
        self.ch3_fill_points = []
        self.exit_fill_points = []

    def load_models(self) -> None:
        """Loads the XGBoost model, preprocessors, and transition matrix from the specified directory.

        The method expects to find:
          - "model.json" for the XGBoost model,
          - "preprocessors.pkl" for the preprocessing objects, and
          - "transition_matrix.pkl" for the transition matrix.

        Raises:
            ValueError: If `save_dir` is not specified.
        """
        if not self.save_dir:
            raise ValueError("Save directory not specified.")

        model_path = os.path.join(self.save_dir, "model.json")
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        preprocessors_path = os.path.join(self.save_dir, "preprocessors.pkl")
        with open(preprocessors_path, "rb") as f:
            self.preprocessors = pickle.load(f)

        transition_matrix_path = os.path.join(
            self.save_dir, "transition_matrix.pkl")
        with open(transition_matrix_path, "rb") as f:
            self.transition_matrix = pickle.load(f)

    def _apply_preprocessors(
        self, X: pd.DataFrame, preprocessors: Dict[str, Any]
    ) -> np.ndarray:
        """Applies the loaded preprocessors to the input data.

        The method copies the input DataFrame, extracts the numerical features, applies the numerical
        imputer, and then scales the features.

        Args:
            X (pd.DataFrame): Input DataFrame containing the numerical features.
            preprocessors (Dict[str, Any]): Dictionary of preprocessor objects. Must include:
                - 'num_imputer': for imputing missing values.
                - 'scaler': for scaling the data.

        Returns:
            np.ndarray: The preprocessed numerical features as a NumPy array.
        """
        X_copy = X.copy()
        X_num = preprocessors['num_imputer'].transform(
            X_copy[self.numerical_features])
        X_num = preprocessors['scaler'].transform(X_num)
        return X_num

    def predict_native(
        self, model: xgb.Booster, preprocessors: Dict[str, Any], X: pd.DataFrame
    ) -> np.ndarray:
        """Generates predictions using the provided XGBoost model.

        The input DataFrame is filtered to contain only numerical features, preprocessed, converted to an
        XGBoost DMatrix, and then used to produce predictions.

        Args:
            model (xgb.Booster): The pre-trained XGBoost model.
            preprocessors (Dict[str, Any]): Dictionary containing preprocessing objects.
            X (pd.DataFrame): Input DataFrame from which predictions are to be generated.

        Returns:
            np.ndarray: An array of prediction probabilities or scores.
        """
        X = X[self.numerical_features]
        X_processed = self._apply_preprocessors(X, preprocessors)
        dmatrix = xgb.DMatrix(X_processed)
        return model.predict(dmatrix)

    def reset_accumulator(self) -> None:
        """Resets the accumulated data and batch counter.

        This method clears the internal DataFrame that accumulates new data and resets the batch number to zero.
        """
        self.accumulated_data = None
        self.batch_num = 0

    def is_prediction_stable(
        self,
        pred_list: List[Tuple[int, int]],
        stability_window: int = 5,
        frequency_threshold: float = 0.8,
        location_tolerance: int = 0
    ) -> Tuple[bool, Optional[int]]:
        """Determines if the prediction is stable based on its occurrence frequency and location consistency.

        Each element in `pred_list` is a tuple consisting of (location, prediction). A prediction is deemed
        stable if, among the last `stability_window` entries:
          - It appears with a frequency at least equal to `frequency_threshold`, and
          - The difference between the maximum and minimum locations does not exceed `location_tolerance`.

        Args:
            pred_list (List[Tuple[int, int]]): List of tuples where each tuple is (location, prediction).
            stability_window (int, optional): Number of recent predictions to consider for stability. Defaults to 5.
            frequency_threshold (float, optional): Required frequency fraction for stability. Defaults to 0.8.
            location_tolerance (int, optional): Maximum allowed difference between locations for stability.
                Defaults to 0.

        Returns:
            Tuple[bool, Optional[int]]:
                - A boolean indicating if a stable prediction was found.
                - The stable prediction class if stable, otherwise None.
        """
        if len(pred_list) < stability_window:
            return False, None

        recent = pred_list[-stability_window:]
        groups: Dict[int, Dict[str, Any]] = {}
        for loc, pred in recent:
            if pred not in groups:
                groups[pred] = {'count': 0, 'locations': []}
            groups[pred]['count'] += 1
            groups[pred]['locations'].append(loc)

        for pred, data in groups.items():
            frequency = data['count'] / stability_window
            if frequency >= frequency_threshold:
                if max(data['locations']) - min(data['locations']) <= location_tolerance:
                    return True, pred
        return False, None

    def update_predictions(
        self,
        new_data: pd.DataFrame,
        stability_window: int = 5,
        frequency_threshold: float = 0.8,
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """Accumulates new data, generates predictions, and assesses prediction stability."""

        # Accumulate new data
        if self.accumulated_data is None:
            self.accumulated_data = pd.DataFrame(columns=new_data.columns)
        self.accumulated_data = pd.concat(
            [self.accumulated_data, new_data], ignore_index=True)
        current_count = len(self.accumulated_data)

        # Compute additional features
        self.accumulated_data = QForecasterDataprocessor.compute_additional_features(
            self.accumulated_data)

        init_fill_point = QForecasterDataprocessor.init_fill_point(
            self.accumulated_data)
        ch1_fill_point = QForecasterDataprocessor.ch1_point(
            self.accumulated_data, init_fill_point)

        ch2_fill_point = -1
        ch3_fill_point = -1
        anomaly_indices_3 = []
        ch3_fill_point = QForecasterDataprocessor.ch3_point(
            self.accumulated_data, init_fill_point, ch1_fill_point, anomaly_indices_3)
        exit_fill_point = QForecasterDataprocessor.exit_point(
            self.accumulated_data, ch3_fill_point, threshold=0)

        # If no fill has started, return waiting status
        if init_fill_point == -1:
            return self._return_waiting_state(current_count)

        # Perform predictions on live data
        predictions, conf = self._generate_predictions(
            init_fill_point, ch1_fill_point, ch2_fill_point)

        # Update prediction history and check stability
        stable_predictions = self._check_stability(
            predictions, conf, init_fill_point, stability_window, frequency_threshold, confidence_threshold)

        self.batch_num += 1

        def safe_average(values):
            # Filter out -1 values if needed; otherwise, use all values
            valid_values = [v for v in values if v != -1]
            return np.median(valid_values) if valid_values else -1

        return {
            "status": "completed",
            "pred": predictions,
            "conf": conf,
            "stable_predictions": stable_predictions,
            "accumulated_data": self.accumulated_data,
            "accumulated_count": current_count,
            "init_fill_point": init_fill_point,
            "ch1_fill_point": ch1_fill_point,
            "ch2_fill_point": ch2_fill_point,
            "ch3_fill_point": ch3_fill_point,
            "exit_fill_point": exit_fill_point,
        }

    # ------------------ Helper Methods ------------------

    def _return_waiting_state(self, current_count: int) -> Dict[str, Any]:
        """Handles case when fill has not started."""
        predictions = np.zeros(current_count, dtype=int)
        conf = np.zeros(current_count)
        for i in range(current_count):
            if i not in self.prediction_history:
                self.prediction_history[i] = []
            self.prediction_history[i].append((i, 0))

        return {
            "status": "waiting",
            "pred": predictions,
            "conf": conf,
            "stable_predictions": {},
            "accumulated_data": self.accumulated_data,
            "accumulated_count": current_count
        }

    def _generate_predictions(self, init_fill_point: int, ch1_fill_point: int, ch2_fill_point: int):
        """Runs predictions and applies Viterbi decoding."""
        predictions = np.zeros(len(self.accumulated_data), dtype=int)
        conf = np.zeros(len(self.accumulated_data))

        # Apply no_fill (0) before init_fill_point
        predictions[:init_fill_point] = 0

        # Get live data for prediction
        X_live = self.accumulated_data[self.numerical_features].iloc[init_fill_point:]
        prob_matrix = self.predict_native(
            self.model, self.preprocessors, X_live)

        # Perform Viterbi decoding
        ml_pred = QForecasterDataprocessor.viterbi_decode(
            prob_matrix, self.transition_matrix)
        ml_pred_offset = ml_pred + 1  # Offset if needed

        predictions[init_fill_point:] = ml_pred_offset
        conf[init_fill_point:] = [prob_matrix[i, ml_pred[i]]
                                  for i in range(len(ml_pred))]

        # Adjust predictions based on fill points
        self._apply_fill_point_constraints(
            predictions, ch1_fill_point, ch2_fill_point)

        return predictions, conf

    def _apply_fill_point_constraints(self, predictions: np.ndarray, ch1_fill_point: int, ch2_fill_point: int):
        """Adjusts predictions based on fill points."""
        if ch1_fill_point > -1:
            predictions[ch1_fill_point:] = 2  # Assign 2 after ch1_fill_point
        if ch2_fill_point > -1:
            predictions[ch2_fill_point:] = 3  # Assign 3 after ch2_fill_point

    def _check_stability(self, predictions: np.ndarray, conf: np.ndarray, init_fill_point: int,
                         stability_window: int, frequency_threshold: float, confidence_threshold: float) -> Dict[int, int]:
        """Checks stability of predictions over a given window."""
        stable_predictions = {}

        for idx, p in enumerate(predictions[init_fill_point:], start=init_fill_point):
            if idx not in self.prediction_history:
                self.prediction_history[idx] = []
            self.prediction_history[idx].append((idx, p))

        for idx, history in self.prediction_history.items():
            stable, stable_class = self.is_prediction_stable(
                history, stability_window, frequency_threshold, int(confidence_threshold))
            if stable:
                stable_predictions[idx] = stable_class

        return stable_predictions

# -----------------------------
# Helper: Simulate Serial Stream
# -----------------------------


def simulate_serial_stream_from_loaded(loaded_data, batch_size=100):
    """
    Yields sequential batches of rows from the loaded data.
    """
    num_rows = len(loaded_data)
    for start_idx in range(0, num_rows, batch_size):
        yield loaded_data.iloc[start_idx:start_idx+batch_size]

# -----------------------------
# Live Prediction Simulator Class
# -----------------------------


def simulate_serial_stream_from_loaded(loaded_data):
    """
    Simulate a serial data stream by yielding random chunks from loaded_data.

    Each chunk will have a random size between 100 and 200 rows.

    Args:
        loaded_data (pd.DataFrame): The loaded dataset to stream.

    Yields:
        pd.DataFrame: A chunk of data with a random number of rows.
    """
    num_rows = len(loaded_data)
    start_idx = 0
    while start_idx < num_rows:
        batch_size = random.randint(10, 30)
        yield loaded_data.iloc[start_idx:start_idx + batch_size]
        start_idx += batch_size


class QForecasterSimulator:
    def __init__(self, predictor: QForecasterPredictor, dataset: pd.DataFrame, poi_file: pd.DataFrame = None, ignore_before=0, delay=1.0):
        """
        Simulator class to stream data in random chunk sizes and update predictions live.
        Args:
            predictor (QForecasterPredictor): Predictor object.
            dataset (pd.DataFrame): Dataset to simulate streaming.
            poi_file (pd.DataFrame, optional): DataFrame containing Points Of Interest indices.
            ignore_before (int): Number of initial rows to ignore.
            delay (float): Delay (in seconds) between processing batches.
        """
        self.predictor = predictor
        self.dataset = dataset
        self.ignore_before = ignore_before
        self.delay = delay

        # If a POI file is provided, extract the actual indices.
        if poi_file is not None:
            self.actual_poi_indices = poi_file
        else:
            self.actual_poi_indices = np.array([])

        # Setup a live plot with a single axis.
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.ion()  # Enable interactive mode.
        plt.show()

    def run(self):
        """
        Runs the simulation by iterating through the dataset in random chunk sizes,
        updating predictions, and plotting the current state.
        """
        self.predictor.reset_accumulator()
        batch_number = 0
        for batch_data in simulate_serial_stream_from_loaded(self.dataset):
            batch_number += 1
            # print(
            #     f"[INFO] Processing batch {batch_number} with {len(batch_data)} rows.")

            # Update predictions using the new batch.
            results = self.predictor.update_predictions(batch_data)

            # Update the live plot.
            self.plot_results(results, batch_number=batch_number)

            # Simulate processing delay.
            plt.pause(self.delay)

        plt.ioff()  # Turn off interactive mode.
        plt.show()

    def plot_results(self, results: dict, batch_number: int, column_to_plot='DoG_SVM_Score'):
        """
        Updates the live plot with two subplots:
        - Top subplot: Normalized curve for the specified column (e.g., Difference).
        - Bottom subplot: Curve for the 'Dissipation' column data.

        Also overlays background shading for predicted class regions and plots anomalous points.
        """
        accumulated_data = results.get("accumulated_data")
        if accumulated_data is None or accumulated_data.empty:
            print("[WARN] No accumulated data available for plotting.")
            return

        # Create x-axis data
        x = np.arange(len(accumulated_data))

        # Clear the entire figure to start fresh
        self.fig.clear()

        # --- Top Plot: Primary Data (e.g., DoG_SVM_Score) ---
        ax1 = self.fig.add_subplot(211)
        data = accumulated_data[column_to_plot].values
        ax1.plot(x, data, color="red", linewidth=1.5, label=column_to_plot)

        # Plot actual POI indices if available
        if hasattr(self, "actual_poi_indices") and self.actual_poi_indices.size > 0:
            valid_actual_indices = [
                int(idx) for idx in self.actual_poi_indices if idx < len(accumulated_data)]
            if valid_actual_indices:
                y_actual = data[valid_actual_indices]
                ax1.scatter(valid_actual_indices, y_actual, color="#2ca02c", marker="o", s=50,
                            label=f"Actual POI {self.actual_poi_indices.size}")

        ax1.set_title(f"{column_to_plot} Curve (Batch {batch_number})",
                      fontsize=14, weight="medium")
        ax1.set_xlabel("Data Index", fontsize=12)
        ax1.set_ylabel("Value", fontsize=12)
        ax1.tick_params(axis="both", which="major", labelsize=10)
        ax1.legend(frameon=False, fontsize=10)

        # --- Bottom Plot: Dissipation Data ---
        if "Dissipation" in accumulated_data.columns:
            ax2 = self.fig.add_subplot(212)
            dissipation_data = accumulated_data["Dissipation"].values
            ax2.plot(x, dissipation_data, color="blue",
                     linewidth=1.5, label="Dissipation")
            # Plot actual POI indices if available
            if hasattr(self, "actual_poi_indices") and self.actual_poi_indices.size > 0:
                valid_actual_indices = [
                    int(idx) for idx in self.actual_poi_indices if idx < len(accumulated_data)]
                if valid_actual_indices:
                    y_actual = dissipation_data[valid_actual_indices]
                    ax2.scatter(valid_actual_indices, y_actual, color="#2ca02c", marker="o", s=50,
                                label=f"Actual POI {self.actual_poi_indices.size}")
            ax2.set_title(
                f"Dissipation Data (Batch {batch_number})", fontsize=14, weight="medium")
            ax2.set_xlabel("Data Index", fontsize=12)
            ax2.set_ylabel("Dissipation", fontsize=12)
            ax2.tick_params(axis="both", which="major", labelsize=10)
            ax2.legend(frameon=False, fontsize=10)
        else:
            print("[WARN] 'Dissipation' column not found in accumulated_data.")

        # Adjust layout for a neat display
        self.fig.tight_layout()

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


TESTING = True
TRAINING = False
# Main execution block.
if __name__ == '__main__':
    SAVE_DIR = r'QModel\SavedModels\forecaster_v3'
    if TRAINING:
        train_dir = r'content\long_tails'
        training_data = QForecasterDataprocessor.load_and_preprocess_data(
            train_dir, num_datasets=100)

        qft = QForecasterTrainer(FEATURES, TARGET, SAVE_DIR)
        qft.train_model(training_data=training_data, tune=True)
        qft.save_models()

    if TESTING:
        test_dir = r"content\test_data"
        test_content = QForecasterDataprocessor.load_content(test_dir)
        random.shuffle(test_content)
        for data_file, poi_file in test_content:
            dataset = pd.read_csv(data_file)
            end = np.random.randint(0, len(dataset))
            end = random.randint(min(end, len(dataset)),
                                 max(end, len(dataset)))
            random_slice = dataset.iloc[0:len(dataset)-1]

            # Load the POI file which is a flat list of 6 indices from the original dataset.
            poi_df = pd.read_csv(poi_file, header=None)
            poi_indices_original = poi_df.to_numpy().flatten()

            # Use these indices to extract the corresponding relative times from the original dataset.
            original_relative_times = dataset.loc[poi_indices_original,
                                                  'Relative_time'].values

            predictor = QForecasterPredictor(
                FEATURES, target=TARGET, save_dir=SAVE_DIR)
            # Ensure models and preprocessors are loaded.
            predictor.load_models()

            delay = dataset['Relative_time'].max() / len(random_slice)
            random_slice = random_slice[random_slice["Relative_time"] >= random.uniform(
                SPIN_UP_TIME[0], SPIN_UP_TIME[1])]
            # Downsample the data.
            random_slice = random_slice.iloc[::DOWNSAMPLE_FACTOR]

            # Get the downsampled relative times.
            downsampled_times = random_slice["Relative_time"].values

            # For each original relative time, find the index in the downsampled data
            # that has the closest relative time.
            mapped_poi_indices = []
            for orig_time in original_relative_times:
                idx = np.abs(downsampled_times - orig_time).argmin()
                mapped_poi_indices.append(idx)
            mapped_poi_indices = np.array(mapped_poi_indices)

            # Create the simulator, passing the mapped POI indices.
            simulator = QForecasterSimulator(
                predictor,
                random_slice,
                poi_file=mapped_poi_indices,
                ignore_before=50,
                delay=delay*5
            )

            # Run the simulation.
            simulator.run()
