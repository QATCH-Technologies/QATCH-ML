from tqdm import tqdm
import os
import logging as logging
import pandas as pd
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch, savgol_filter, argrelextrema
from scipy.stats import skew, kurtosis, zscore
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import random
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
TARGET = "Fill"
DOWNSAMPLE_FACTOR = 5
SPIN_UP_TIME = (1.2, 1.4)
BASELINE_WINDOW = 100


class DataProcessor:
    @staticmethod
    def load_content(data_dir: str, num_datasets: int = np.inf) -> list:
        logging.info(f"Loading content from {data_dir}")
        loaded_content = []

        for root, _, files in tqdm(os.walk(data_dir), desc='Loading files...'):
            for f in files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    poi_file = f.replace(".csv", "_poi.csv")
                    if not os.path.exists(os.path.join(
                            root, poi_file)):
                        continue
                    poi_df = pd.read_csv(os.path.join(
                        root, poi_file), header=None)
                    poi_values = poi_df.values
                    if len(poi_values) != len(np.unique(poi_values)):
                        logging.warning(
                            f'POI file contains duplicate indices: {poi_df}.')
                    else:
                        loaded_content.append(
                            (os.path.join(root, f), os.path.join(root, poi_file))
                        )

        random.shuffle(loaded_content)
        if num_datasets == np.inf:
            return loaded_content

        return loaded_content[:num_datasets]

    @staticmethod
    def find_sampling_shift(df: pd.DataFrame) -> int:
        time_df = pd.DataFrame()
        time_df["Delta"] = df["Relative_time"].diff()
        threshold = 0.032
        rolling_avg = time_df["Delta"].expanding(min_periods=2).mean()
        time_df["Significant_change"] = (
            time_df["Delta"] - rolling_avg).abs() > threshold
        change_indices = time_df.index[time_df["Significant_change"]].tolist()
        return change_indices[0] if change_indices else -1

    @staticmethod
    def compute_dissipation_dog(df: pd.DataFrame, sigma: float = 2) -> pd.Series:
        return pd.Series(gaussian_filter1d(df['Dissipation'], sigma=sigma, order=1),
                         index=df.index)

    @staticmethod
    def compute_rolling_baseline_and_shift(dog_series: pd.Series, window: int) -> tuple:
        baseline = dog_series.rolling(
            window=window, center=True, min_periods=1).median()
        shift = dog_series - baseline
        return baseline, shift

    @staticmethod
    def compute_ocsvm_score(shift_series: pd.Series) -> np.ndarray:
        X = shift_series.values.reshape(-1, 1)
        ocsvm = OneClassSVM(nu=0.05, kernel='rbf',
                            gamma='scale', shrinking=False)
        ocsvm.fit(X)
        return ocsvm.decision_function(X)

    @staticmethod
    def compute_spectral_features(signal: np.ndarray, relative_time: pd.Series) -> tuple:
        fs = 1 / np.mean(np.diff(relative_time))
        f, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))
        psd_norm = Pxx / np.sum(Pxx)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
        dominant_frequency = f[np.argmax(Pxx)]
        return spectral_entropy, dominant_frequency

    @staticmethod
    def smooth_signal(signal: np.ndarray) -> np.ndarray:
        window_size = min(11, len(signal) - 1) if len(signal) > 11 else 5
        return savgol_filter(signal, window_size, polyorder=2)

    @staticmethod
    def compute_rolling_statistics(signal: pd.Series, window: int) -> tuple:
        skew_series = signal.rolling(
            window, min_periods=1).apply(skew, raw=True)
        kurtosis_series = signal.rolling(
            window, min_periods=1).apply(kurtosis, raw=True)
        variability_series = signal.rolling(window, min_periods=1).std()
        return skew_series, kurtosis_series, variability_series

    @staticmethod
    def compute_anomaly_zscore(signal: pd.Series) -> tuple:
        z_scores = pd.Series(
            zscore(signal, nan_policy="omit"), index=signal.index)
        anomaly = (np.abs(z_scores) > 2).astype(int)
        return z_scores, anomaly

    @staticmethod
    def cluster_anomalies(anomaly_flags: pd.Series) -> pd.Series:
        anomaly_indices = anomaly_flags[anomaly_flags == 1].index
        if len(anomaly_indices) > 0:
            dbscan = DBSCAN(eps=5, min_samples=2)
            labels = dbscan.fit_predict(anomaly_indices.values.reshape(-1, 1))
            clustered_anomalies = np.zeros(len(anomaly_flags))
            # Only mark the first index of each cluster as an anomaly
            for cluster in set(labels):
                if cluster != -1:  # Skip noise
                    first_index = anomaly_indices[labels == cluster][0]
                    clustered_anomalies[first_index] = 1
            return pd.Series(clustered_anomalies, index=anomaly_flags.index)
        return anomaly_flags

    def compute_difference_curve(df: pd.DataFrame) -> pd.Series:
        xs = df["Relative_time"]
        i = next((x for x, t in enumerate(xs) if t > 0.5), None)
        j = next((x for x, t in enumerate(xs) if t > 2.5), None)

        if i is not None and j is not None:
            avg_res_freq = df["Resonance_Frequency"][i:j].mean()
            avg_diss = df["Dissipation"][i:j].mean()
            ys_diss = (df["Dissipation"] - avg_diss) * avg_res_freq / 2
            ys_freq = avg_res_freq - df["Resonance_Frequency"]
            difference_factor = 3
            return ys_freq - difference_factor * ys_diss
        return pd.Series(0, index=df.index)

    @staticmethod
    def generate_features(df: pd.DataFrame, live=True) -> pd.DataFrame:
        if live:
            required_columns = ["Dissipation",
                                "Resonance_Frequency", "Relative_time"]
        else:
            required_columns = ["Dissipation",
                                "Resonance_Frequency", "Relative_time", "Fill"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"Input DataFrame must contain the following columns: {required_columns}")
        df = df[required_columns].copy()
        df['Dissipation_DoG'] = DataProcessor.compute_dissipation_dog(df)
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['DoG_baseline'], df['DoG_shift'] = DataProcessor.compute_rolling_baseline_and_shift(
            df['Dissipation_DoG'], baseline_window)
        df['DoG_SVM_Score'] = DataProcessor.compute_ocsvm_score(
            df['DoG_shift'])
        z_scores, anomaly = DataProcessor.compute_anomaly_zscore(
            df["DoG_SVM_Score"])
        df["DoG_SVM_Zscore"] = z_scores
        df["DoG_SVM_Anomaly"] = anomaly
        df["DoG_SVM_Anomaly"] = DataProcessor.cluster_anomalies(
            df["DoG_SVM_Anomaly"])
        df["Difference"] = DataProcessor.compute_difference_curve(df)
        # Smooth signal
        smooth_factor = 21
        if len(df) > smooth_factor:
            smoothed_DoG = savgol_filter(
                df['DoG_SVM_Score'], window_length=smooth_factor, polyorder=3)
        else:
            smoothed_DoG = df['DoG_SVM_Score'].values

        # First derivative (change rate)
        df['DoG_derivative'] = np.gradient(smoothed_DoG)

        # Signal slope (same as derivative or could be a longer trend-based slope)
        df['DoG_slope'] = pd.Series(smoothed_DoG).rolling(window=10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)

        # Sliding window features — we'll extract mean, std, min, max in each window
        window_size = 20
        df['DoG_win_mean'] = pd.Series(
            smoothed_DoG).rolling(window=window_size).mean()
        df['DoG_win_std'] = pd.Series(
            smoothed_DoG).rolling(window=window_size).std()
        df['DoG_win_min'] = pd.Series(
            smoothed_DoG).rolling(window=window_size).min()
        df['DoG_win_max'] = pd.Series(
            smoothed_DoG).rolling(window=window_size).max()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def preprocess_data(df: pd.DataFrame, poi_file: str):

        def process_fill(poi_file: str, length_df: int) -> pd.DataFrame:
            fill_df = pd.read_csv(poi_file, header=None)
            fill_positions = fill_df[0].astype(int).values
            labels = np.zeros(length_df, dtype=int)

            for i, pos in enumerate(fill_positions):
                start = max(0, pos - 100)
                end = min(length_df, pos + 100 + 1)

                if i < 4:
                    label = 1
                elif i == 4:
                    label = 2
                else:  # i == 5
                    label = 3

                labels[start:end] = label

            return pd.DataFrame(labels, columns=['Fill'])
        # Load the dataset
        required_cols = ["Relative_time", "Dissipation", "Resonance_Frequency"]
        if df.empty or not all(col in df.columns for col in required_cols):
            raise ValueError(
                "The dataset is empty or missing required columns.")
        df = df[required_cols].copy()
        try:
            fill_arr = process_fill(poi_file, len(df))
            df['Fill'] = fill_arr[:len(df)]
        except FileNotFoundError:
            raise ValueError("POI file not found.")
        # Filter and downsample the data
        df = df[df["Relative_time"] >= random.uniform(
            SPIN_UP_TIME[0], SPIN_UP_TIME[1])]
        df = df.iloc[::DOWNSAMPLE_FACTOR]
        if df.empty:
            return None
        df = df.reset_index(drop=True)
        # Generate features and clean up the DataFrame
        df = DataProcessor.generate_features(df, live=False)
        df.reset_index(drop=True, inplace=True)
        df.drop(columns=['Relative_time'], inplace=True)

        return df
