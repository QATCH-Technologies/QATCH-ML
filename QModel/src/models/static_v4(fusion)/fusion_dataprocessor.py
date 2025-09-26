import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import random
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.svm import OneClassSVM
from scipy.interpolate import interp1d


class FusionDataprocessor:
    DROP = ["Date", "Time", "Ambient", "Peak Magnitude (RAW)", "Temperature"]
    BASELINE_WIN = 500
    ROLLING_WIN = 50

    @staticmethod
    def load_content(data_dir: str, num_datasets: Union[int, float] = np.inf) -> List[Tuple[str, str]]:

        if not isinstance(data_dir, str) or not data_dir.strip():
            raise ValueError("data_dir must be a non-empty string.")
        if not os.path.isdir(data_dir):
            raise ValueError(f"Directory '{data_dir}' does not exist.")

        # First, collect all valid datasets with their POI characteristics
        dataset_poi_info: Dict[str, Dict] = {}

        for root, _, files in tqdm(os.walk(data_dir), desc='Scanning files...'):
            for f in files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    poi_file = f.replace(".csv", "_poi.csv")
                    poi_path = os.path.join(root, poi_file)
                    if not os.path.exists(poi_path):
                        continue
                    try:
                        poi_df = pd.read_csv(poi_path, header=None)
                    except Exception as e:
                        continue

                    poi_values = poi_df.values.flatten()
                    if len(poi_values) != len(np.unique(poi_values)):
                        continue

                    # Store dataset info with POI characteristics
                    data_path = os.path.join(root, f)
                    dataset_poi_info[data_path] = {
                        'poi_path': poi_path,
                        'poi_positions': poi_values,
                        'poi_stats': {
                            'mean': np.mean(poi_values),
                            'std': np.std(poi_values),
                            'min': np.min(poi_values),
                            'max': np.max(poi_values),
                            'median': np.median(poi_values),
                            'count': len(poi_values)
                        }
                    }

        if not dataset_poi_info:
            return []

        loaded_content = [(path, info['poi_path'])
                          for path, info in dataset_poi_info.items()]
        random.shuffle(loaded_content)
        if num_datasets != np.inf:
            loaded_content = loaded_content[:int(num_datasets)]
        return loaded_content

    @staticmethod
    def compute_difference_curve(df: pd.DataFrame, difference_factor: int = 2) -> pd.Series:
        required_cols = ["Relative_time",
                         "Resonance_Frequency", "Dissipation"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' is missing from DataFrame.")

        xs = df["Relative_time"]

        i = next((x for x, t in enumerate(xs) if t > 0.5), 0)
        j = next((x for x, t in enumerate(xs) if t > 2.5), 1)
        if i == j:
            j = next((x for x, t in enumerate(
                xs) if t > xs[j] + 2.0), j + 1)

        avg_res_freq = df["Resonance_Frequency"].iloc[i:j].mean()
        avg_diss = df["Dissipation"].iloc[i:j].mean()
        ys_diss = (df["Dissipation"] - avg_diss) * avg_res_freq / 2
        ys_freq = avg_res_freq - df["Resonance_Frequency"]
        return ys_freq - difference_factor * ys_diss

    @staticmethod
    def compute_time_aware_baseline(df, baseline_duration=3.0):
        """
        Compute baseline using time duration instead of sample count
        """
        time = df["Relative_time"].values
        baseline_mask = time <= baseline_duration

        base_d = df["Dissipation"][baseline_mask].mean()
        base_rf = df["Resonance_Frequency"][baseline_mask].mean()

        return base_d, base_rf

    @staticmethod
    def weighted_smooth(values, time, window_size=0.1):  # window in seconds
        smoothed = np.zeros_like(values)
        for i, t in enumerate(time):
            mask = np.abs(time - t) <= window_size/2
            if np.sum(mask) > 0:
                distances = np.abs(time[mask] - t)
                weights = np.exp(-distances / (window_size/4))
                weights /= weights.sum()

                smoothed[i] = np.average(values[mask], weights=weights)
            else:
                smoothed[i] = values[i]
        return smoothed

    @staticmethod
    def get_features(df: pd.DataFrame):
        def compute_ocsvm_score(shift_series: pd.Series):
            if shift_series.empty:
                raise ValueError("shift_series is empty.")
            X = shift_series.values.reshape(-1, 1)
            ocsvm = OneClassSVM(nu=0.05, kernel='rbf',
                                gamma='scale', shrinking=False)
            ocsvm.fit(X)
            # get raw decision scores and flip so negative spikes become positive
            scores = ocsvm.decision_function(X)
            scores = -scores
            # baseline at zero
            scores = scores - np.min(scores)
            return scores

        def compute_DoG(df: pd.DataFrame, col: str, sigma: float = 2):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            if sigma <= 0:
                raise ValueError("sigma must be a positive number.")
            result = gaussian_filter1d(df[col], sigma=sigma, order=1)
            return pd.Series(result, index=df.index)

        def compute_rolling_baseline_and_shift(dog_series: pd.Series, window: int):
            if window <= 0:
                raise ValueError("window must be a positive integer.")
            baseline = dog_series.rolling(
                window=window, center=True, min_periods=1).median()
            shift = dog_series - baseline
            return baseline, shift
        df = df.copy()
        df = df.drop(columns=FusionDataprocessor.DROP)

        baseline_duration = FusionDataprocessor.BASELINE_WIN * 0.008  # convert to seconds
        base_d, base_rf = FusionDataprocessor.compute_time_aware_baseline(
            df, baseline_duration)

        # Apply baseline correction
        df["Dissipation"] = df["Dissipation"] - base_d
        df["Resonance_Frequency"] = -(df["Resonance_Frequency"] - base_rf)

        # Compute difference with regularized data
        df["Difference"] = FusionDataprocessor.compute_difference_curve(
            df, difference_factor=2)
        df["Difference"] = -(df["Difference"])
        needs_smoothing = True
        if needs_smoothing:
            for col in ["Dissipation", "Resonance_Frequency", "Difference"]:
                df[col] = FusionDataprocessor.weighted_smooth(df[col].values,
                                                              df["Relative_time"].values,
                                                              window_size=0.05)
        df['Dissipation_DoG'] = compute_DoG(
            df, col='Dissipation')
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['Dissipation_DoG_baseline'], df['Dissipation_DoG_shift'] = compute_rolling_baseline_and_shift(
            df['Dissipation_DoG'], baseline_window
        )
        df['Dissipation_DoG_SVM_Score'] = compute_ocsvm_score(
            df['Dissipation_DoG_shift'])

        # `Resonance_Frequency` DoG processing
        df['Resonance_Frequency_DoG'] = compute_DoG(
            df, col='Resonance_Frequency')
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['Resonance_Frequency_DoG_baseline'], df['Resonance_Frequency_DoG_shift'] = compute_rolling_baseline_and_shift(
            df['Resonance_Frequency_DoG'], baseline_window
        )
        df['Resonance_Frequency_DoG_SVM_Score'] = compute_ocsvm_score(
            df['Resonance_Frequency_DoG_shift'])

        # `Difference` DoG processing
        df['Difference_DoG'] = compute_DoG(
            df, col='Difference')
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['Difference_DoG_baseline'], df['Difference_DoG_shift'] = compute_rolling_baseline_and_shift(
            df['Difference_DoG'], baseline_window
        )
        df['Difference_DoG_SVM_Score'] = compute_ocsvm_score(
            df['Difference_DoG_shift'])
        return df


if __name__ == "__main__":
    data_dir = "content/train"
    num_datasets = 10
    file_pairs = FusionDataprocessor.load_content(
        data_dir, num_datasets=num_datasets)

    for data_file, _ in tqdm(file_pairs):
        df = pd.read_csv(data_file, engine="pyarrow")
        df = FusionDataprocessor.get_features(df)
        plt.figure()
        plt.plot(df["Difference"])
        plt.show()
