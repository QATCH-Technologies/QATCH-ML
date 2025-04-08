import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from typing import Union, Dict, List, Tuple
from sklearn.metrics import silhouette_score
from sklearn.svm import OneClassSVM
from scipy.signal import savgol_filter, butter, filtfilt, detrend, find_peaks
from scipy.ndimage import gaussian_filter1d


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
TARGET = "POI"
PLOTTING = False


class QDataProcessor:
    @staticmethod
    def load_content(data_dir: str, num_datasets: int = np.inf, column: str = 'Dissipation') -> list:
        logging.info(f"Loading content from {data_dir}")
        loaded_content = []

        for root, _, files in tqdm(os.walk(data_dir), desc='Loading files...'):
            for f in files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    poi_file = f.replace(".csv", "_poi.csv")
                    if not os.path.exists(os.path.join(root, poi_file)):
                        continue
                    poi_df = pd.read_csv(os.path.join(
                        root, poi_file), header=None)
                    poi_values = poi_df.values
                    if len(poi_values) != len(np.unique(poi_values)):
                        logging.warning(
                            f'POI file contains duplicate indices.')
                    else:
                        loaded_content.append(
                            (os.path.join(root, f), os.path.join(root, poi_file))
                        )
        random.shuffle(loaded_content)
        if num_datasets == np.inf:
            return loaded_content

        return loaded_content[:num_datasets]

    @staticmethod
    def load_balanced_content(data_dir: str, num_datasets: Union[int, float] = np.inf, clusters: int = 3, opt: bool = False) -> Union[Tuple[List[Tuple[str, str]], int], List]:
        """
        Load and balance content by clustering feature vectors extracted from CSV files.

        The function loads content using `load_content`, computes feature vectors for each valid file,
        and uses KMeans clustering to balance the datasets. Optionally, the optimal number of clusters is computed.

        Args:
            data_dir (str): Directory path to search for files.
            num_datasets (int or float, optional): Desired number of datasets. Defaults to np.inf.
            clusters (int, optional): Number of clusters to use if not using optimal clustering. Defaults to 3.
            opt (bool, optional): If True, compute the optimal number of clusters. Defaults to False.

        Returns:
            Union[Tuple[List[Tuple[str, str]], int], List]: A tuple containing the balanced sample and the number of clusters,
                                                              or an empty list if no content is loaded.
        """
        loaded_content = QDataProcessor.load_content(
            data_dir, num_datasets=np.inf)
        if not loaded_content:
            return []

        features = []
        valid_content = []
        for data_file, poi_file in loaded_content:
            try:
                df = pd.read_csv(data_file)
                poi_df = pd.read_csv(poi_file, header=None)
                if "Dissipation" not in df.columns or "Resonance_Frequency" not in df.columns:
                    continue
                poi_mean = np.mean(poi_df.values)
                poi_std = np.std(poi_df.values)
                mean_diss = df["Dissipation"].mean()
                std_diss = df["Dissipation"].std()
                mean_freq = df["Resonance_Frequency"].mean()
                std_freq = df["Resonance_Frequency"].std()
                max_r_time = df['Relative_time'].max(
                ) if 'Relative_time' in df.columns else 0
                feature_vec = [poi_mean, poi_std, mean_diss,
                               std_diss, mean_freq, std_freq, max_r_time]
                features.append(feature_vec)
                valid_content.append((data_file, poi_file))
            except Exception as e:
                logging.error(f"Could not read file {data_file}: {e}")
                continue

        if not features:
            return []
        features_arr = np.array(features)
        if opt:
            n_clusters, silhouette_scores = QDataProcessor.compute_optimal_clusters(
                features_arr, min_clusters=2, max_clusters=50)
            logging.info(
                f"Optimal clusters: {n_clusters} with score {silhouette_scores}.")
        else:
            n_clusters = min(clusters, len(features_arr))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_arr)

        cluster_dict: Dict[int, List[Tuple[str, str]]] = {}
        for label, file_pair in zip(labels, valid_content):
            cluster_dict.setdefault(label, []).append(file_pair)

        balanced_sample: List[Tuple[str, str]] = []
        if num_datasets == np.inf:
            min_count = min(len(files) for files in cluster_dict.values())
            for file_list in cluster_dict.values():
                balanced_sample.extend(random.sample(file_list, min_count))
        else:
            per_cluster = int(num_datasets) // n_clusters
            for file_list in cluster_dict.values():
                if len(file_list) <= per_cluster:
                    balanced_sample.extend(file_list)
                else:
                    balanced_sample.extend(
                        random.sample(file_list, per_cluster))
            remaining = int(num_datasets) - len(balanced_sample)
            if remaining > 0:
                extra_files = [fp for files in cluster_dict.values()
                               for fp in files if fp not in balanced_sample]
                if extra_files:
                    balanced_sample.extend(random.sample(
                        extra_files, min(remaining, len(extra_files))))
        random.shuffle(balanced_sample)

        if PLOTTING:
            plt.figure(figsize=(10, 6))
            plt.scatter(features_arr[:, 0], features_arr[:, 1],
                        c=labels, cmap='viridis', s=50, label='All files')
            selected_indices = [i for i, pair in enumerate(
                valid_content) if pair in balanced_sample]
            plt.scatter(features_arr[selected_indices, 0], features_arr[selected_indices, 1],
                        facecolors='none', edgecolors='red', s=100, label='Selected balanced sample')
            centers = kmeans.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], c='black',
                        s=200, marker='X', label='Cluster Centers')
            plt.xlabel("Mean Dissipation")
            plt.ylabel("Std. Dissipation")
            plt.title("Clustering of Data Files and Selected Balanced Sample")
            plt.legend()
            plt.grid(True)
            plt.show()

        return balanced_sample, n_clusters

    @staticmethod
    def compute_optimal_clusters(features: np.ndarray, min_clusters: int = 2, max_clusters: int = 10) -> Tuple[int, Dict[int, float]]:
        """
        Compute the optimal number of clusters for the given feature set using silhouette scores.

        Args:
            features (np.ndarray): Array of feature vectors.
            min_clusters (int, optional): Minimum number of clusters to consider. Defaults to 2.
            max_clusters (int, optional): Maximum number of clusters to consider. Defaults to 10.

        Returns:
            Tuple[int, Dict[int, float]]: The optimal number of clusters and a dictionary mapping cluster numbers to silhouette scores.

        Raises:
            ValueError: If features is empty.
        """
        if features.size == 0:
            raise ValueError("Features array is empty.")

        scores: Dict[int, float] = {}
        best_k: int = min_clusters
        best_score: float = -1.0
        max_possible = min(max_clusters, features.shape[0])
        for k in range(min_clusters, max_possible + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k

        if PLOTTING:
            plt.figure(figsize=(8, 4))
            plt.plot(list(scores.keys()), list(scores.values()), marker='o')
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Average Silhouette Score")
            plt.title("Silhouette Score vs. Number of Clusters")
            plt.grid(True)
            plt.show()

        return best_k, scores

    @staticmethod
    def find_sampling_shift(df: pd.DataFrame) -> int:
        """
        Find the index where a significant shift in sampling occurs.

        Computes the difference of the "Relative_time" column, compares it against a threshold,
        and returns the first index where the difference exceeds the threshold.

        Args:
            df (pd.DataFrame): Input DataFrame that must contain the "Relative_time" column.

        Returns:
            int: The index of the first significant change, or -1 if none is found.

        Raises:
            ValueError: If "Relative_time" column is missing.
        """
        if "Relative_time" not in df.columns:
            raise ValueError(
                "Input DataFrame must contain 'Relative_time' column.")

        time_df = pd.DataFrame()
        time_df["Delta"] = df["Relative_time"].diff()
        threshold = 0.032
        rolling_avg = time_df["Delta"].expanding(min_periods=2).mean()
        time_df["Significant_change"] = (
            time_df["Delta"] - rolling_avg).abs() > threshold
        change_indices = time_df.index[time_df["Significant_change"]].tolist()
        return change_indices[0] if change_indices else -1

    @staticmethod
    def compute_ocsvm_score(shift_series: pd.Series) -> np.ndarray:
        """
        Compute anomaly scores using One-Class SVM on a shifted series.

        Args:
            shift_series (pd.Series): The series on which to compute the anomaly score.

        Returns:
            np.ndarray: Array of anomaly scores from the One-Class SVM.

        Raises:
            ValueError: If shift_series is empty.
        """
        if shift_series.empty:
            raise ValueError("shift_series is empty.")
        X = shift_series.values.reshape(-1, 1)
        ocsvm = OneClassSVM(nu=0.05, kernel='rbf',
                            gamma='scale', shrinking=False)
        ocsvm.fit(X)
        return ocsvm.decision_function(X)

    @staticmethod
    def compute_DoG(df: pd.DataFrame, col: str, sigma: float = 2) -> pd.Series:
        """
        Compute the Difference of Gaussians (DoG) for a specified column in the DataFrame.

        Applies a Gaussian filter to compute the first derivative of the specified column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            col (str): Column name on which to compute the DoG.
            sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to 2.

        Returns:
            pd.Series: Series containing the computed DoG.

        Raises:
            ValueError: If the column is not in the DataFrame or if sigma is non-positive.
        """
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if sigma <= 0:
            raise ValueError("sigma must be a positive number.")
        result = gaussian_filter1d(df[col], sigma=sigma, order=1)
        return pd.Series(result, index=df.index)

    @staticmethod
    def compute_difference_curve(df: pd.DataFrame, difference_factor: int = 2) -> pd.Series:
        """
        Compute the difference curve from the DataFrame.

        Searches for indices in 'Relative_time' where the time exceeds given thresholds,
        then computes a difference curve using 'Resonance_Frequency' and 'Dissipation'.

        Args:
            df (pd.DataFrame): Input DataFrame that must contain 'Relative_time',
                               'Resonance_Frequency', and 'Dissipation' columns.

        Returns:
            pd.Series: The computed difference curve.

        Raises:
            ValueError: If required columns are missing or if thresholds cannot be determined.
        """
        required_cols = ["Relative_time", "Resonance_Frequency", "Dissipation"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' is missing from DataFrame.")

        xs = df["Relative_time"]
        i = next((x for x, t in enumerate(xs) if t > 0.5), None)
        j = next((x for x, t in enumerate(xs) if t > 2.5), None)

        avg_res_freq = df["Resonance_Frequency"].iloc[i:j].mean()
        avg_diss = df["Dissipation"].iloc[i:j].mean()
        ys_diss = (df["Dissipation"] - avg_diss) * avg_res_freq / 2
        ys_freq = avg_res_freq - df["Resonance_Frequency"]
        return ys_freq - difference_factor * ys_diss

    @staticmethod
    def compute_rolling_baseline_and_shift(dog_series: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
        """
        Compute the rolling baseline and shift of a DoG series.

        Args:
            dog_series (pd.Series): The input DoG series.
            window (int): Window size for the rolling median.

        Returns:
            Tuple[pd.Series, pd.Series]: A tuple containing the rolling baseline and the difference (shift).

        Raises:
            ValueError: If window is not a positive integer.
        """
        if window <= 0:
            raise ValueError("window must be a positive integer.")
        baseline = dog_series.rolling(
            window=window, center=True, min_periods=1).median()
        shift = dog_series - baseline
        return baseline, shift

    @staticmethod
    def compute_super_gradient(series: pd.Series) -> pd.Series:

        window = int(len(series) * 0.01)
        if window % 2 == 0:
            window += 1
        if window <= 1:
            window = 3
        smoothed_data = savgol_filter(
            x=series, window_length=window, polyorder=1, deriv=0
        )
        gradient = savgol_filter(
            x=smoothed_data, window_length=window, polyorder=1, deriv=1
        )
        return gradient

    @staticmethod
    def compute_cumulative_gradient(series_1: pd.Series, series_2: pd.Series):
        return savgol_filter(
            series_1.values + series_2.values,
            window_length=25,
            polyorder=1,
            deriv=1,
        )

    @staticmethod
    def compute_graident(series: pd.Series):
        return series.diff()

    @staticmethod
    def noise_filter(series: pd.Series):
        if series.size == 0 or series is None:
            logging.error("`series` cannot be an empty pd.Series.")
            raise ValueError(
                "`series` cannot be an empty pd.Series."
            )
        # Define Butterworth low-pass filter parameters
        fs = 300  # Sampling frequency
        normal_cutoff = 2 / (0.5 * fs)  # Normalize the cutoff frequency

        # Get the filter coefficients for a 2nd order low-pass Butterworth filter
        b, a = butter(2, Wn=normal_cutoff, btype="lowpass", analog=False)
        # Apply the filter to the data using filtfilt for zero-phase filtering
        filtered = filtfilt(b, a, series.values)
        filtered = np.maximum(filtered, 0)
        return filtered

    @staticmethod
    def detrend(series: pd.Series):
        return detrend(series.values)

    @staticmethod
    def generate_features(df: pd.DataFrame, live: bool = True) -> pd.DataFrame:
        if live:
            required_columns = ["Dissipation",
                                "Resonance_Frequency", "Relative_time"]
        else:
            required_columns = ["Dissipation",
                                "Resonance_Frequency", "Relative_time", "POI"]

        if df.empty or not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"Input DataFrame must contain the following columns: {required_columns}")

        df = df[required_columns].copy()
        if df.empty:
            raise ValueError(
                "DataFrame is empty after selecting required columns.")

        # Calculate difference curve
        df["Difference"] = QDataProcessor.compute_difference_curve(df)

        # Compute cummulative curve
        df["Cumulative"] = QDataProcessor.compute_cumulative_gradient(
            df["Dissipation"], df["Resonance_Frequency"])

        # Smooth `Difference`, `Dissipation`, and `Resonance_Frequency`
        smooth_win = int(
            0.005 * len(df["Relative_time"].values))
        if smooth_win % 2 == 0:
            smooth_win += 1
        if smooth_win <= 1:
            smooth_win = 2
        polyorder = 3
        df["Difference_smooth"] = savgol_filter(
            df["Difference"], smooth_win, polyorder)
        df["Dissipation_smooth"] = savgol_filter(
            df["Dissipation"], smooth_win, polyorder)
        df["Resonance_Frequency_smooth"] = savgol_filter(
            df["Resonance_Frequency"], smooth_win, polyorder)

        # `Dissipation` DoG processing
        df['Dissipation_DoG'] = QDataProcessor.compute_DoG(
            df, col='Dissipation')
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['Dissipation_DoG_baseline'], df['Dissipation_DoG_shift'] = QDataProcessor.compute_rolling_baseline_and_shift(
            df['Dissipation_DoG'], baseline_window
        )
        df['Dissipation_DoG_SVM_Score'] = QDataProcessor.compute_ocsvm_score(
            df['Dissipation_DoG_shift'])

        # `Resonance_Frequency` DoG processing
        df['Resonance_Frequency_DoG'] = QDataProcessor.compute_DoG(
            df, col='Resonance_Frequency')
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['Resonance_Frequency_DoG_baseline'], df['Resonance_Frequency_DoG_shift'] = QDataProcessor.compute_rolling_baseline_and_shift(
            df['Resonance_Frequency_DoG'], baseline_window
        )
        df['Resonance_Frequency_DoG_SVM_Score'] = QDataProcessor.compute_ocsvm_score(
            df['Resonance_Frequency_DoG_shift'])

        # `Difference` DoG processing
        df['Difference_DoG'] = QDataProcessor.compute_DoG(
            df, col='Difference')
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['Difference_DoG_baseline'], df['Difference_DoG_shift'] = QDataProcessor.compute_rolling_baseline_and_shift(
            df['Difference_DoG'], baseline_window
        )
        df['Difference_DoG_SVM_Score'] = QDataProcessor.compute_ocsvm_score(
            df['Difference_DoG_shift'])

        # Compute `super_gradient` column.
        df['Super_Dissipation'] = QDataProcessor.compute_super_gradient(
            df['Dissipation'])
        df['Super_Difference'] = QDataProcessor.compute_super_gradient(
            df['Difference'])
        df['Super_Resonance_Frequency'] = QDataProcessor.compute_super_gradient(
            df['Resonance_Frequency'])
        df['Super_Cumulative'] = QDataProcessor.compute_super_gradient(
            df['Cumulative'])

        # Compute gradients
        df["Gradient_Dissipation"] = QDataProcessor.compute_graident(
            df["Dissipation"])
        df["Gradient_Resonance_Frequency"] = QDataProcessor.compute_graident(
            df["Resonance_Frequency"])
        df["Gradient_Difference"] = QDataProcessor.compute_graident(
            df["Difference"])

        # Noise filter data
        df["Filtered_Dissipation"] = QDataProcessor.noise_filter(
            df["Gradient_Dissipation"])
        df["Filtered_Difference"] = QDataProcessor.noise_filter(
            df["Difference"])
        df["Filtered_Resonance_Frequency"] = QDataProcessor.noise_filter(
            df["Resonance_Frequency"])
        df["Filtered_Gradient_Dissipation"] = QDataProcessor.noise_filter(
            df["Gradient_Dissipation"])
        df["Filtered_Gradient_Resonance_Frequency"] = QDataProcessor.noise_filter(
            df["Gradient_Resonance_Frequency"])
        df["Filtered_Gradient_Difference"] = QDataProcessor.noise_filter(
            df["Gradient_Difference"])
        df["Filtered_Cumulative"] = QDataProcessor.noise_filter(
            df["Cumulative"])

        # Detrend data
        df["Detrend_Dissipation"] = QDataProcessor.detrend(df["Dissipation"])
        df["Detrend_Difference"] = QDataProcessor.detrend(df["Difference"])
        df["Detrend_Resonance_Frequency"] = QDataProcessor.detrend(
            df["Resonance_Frequency"])
        df["Detrend_Cumulative"] = QDataProcessor.detrend(df["Cumulative"])
        # # Re-baseline `Dissipation` and `Resonance_Frequency`
        # slice_loc = int(len(df) * 0.01)
        # if slice_loc == 0 or len(df) < 2 * slice_loc:
        #     logging.error(
        #         "DataFrame is too small to compute baseline adjustments.")
        #     raise ValueError(
        #         "DataFrame is too small to compute baseline adjustments.")
        # df["Dissipation"] -= np.average(
        #     df["Dissipation"].iloc[slice_loc:2 * slice_loc])
        # df["Resonance_Frequency"] -= np.average(
        #     df["Resonance_Frequency"].iloc[slice_loc:2 * slice_loc])
        # df["Resonance_Frequency"] = df["Resonance_Frequency"].values * -1

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        return df

    @staticmethod
    def process_poi(poi_file: str, length_df: int) -> pd.DataFrame:
        if not isinstance(poi_file, str) or not poi_file.strip():
            raise ValueError("poi_file must be a non-empty string.")
        if not os.path.exists(poi_file):
            raise ValueError(f"POI file '{poi_file}' does not exist.")
        if not isinstance(length_df, int) or length_df <= 0:
            raise ValueError("length_df must be a positive integer.")
        poi_df = pd.read_csv(poi_file, header=None)
        poi_positions = poi_df[0].astype(int).values

        if len(poi_positions) < 6:
            raise ValueError("POI file does not contain enough fill indices.")

        labels = np.zeros(length_df, dtype=int)

        for i, pos in enumerate(poi_positions[:6]):
            labels[pos] = i + 1

        return pd.DataFrame(labels)

    @staticmethod
    def process_data(df: pd.DataFrame, live: bool = True) -> pd.DataFrame:
        if live:
            required_cols = ["Relative_time",
                             "Dissipation", "Resonance_Frequency"]
        else:
            required_cols = ["Relative_time", "Dissipation",
                             "Resonance_Frequency", "POI"]
        if df.empty or not all(col in df.columns for col in required_cols):
            raise ValueError(
                "The dataset is empty or missing required columns.")
        df = df[required_cols].copy()
        if df.empty:
            raise ValueError(
                "DataFrame is empty after filtering required columns.")

        df = df.reset_index(drop=True)
        df = QDataProcessor.generate_features(
            df, live=live)
        df.reset_index(drop=True, inplace=True)
        if "Relative_time" in df.columns:
            df.drop(columns=['Relative_time'], inplace=True)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df
