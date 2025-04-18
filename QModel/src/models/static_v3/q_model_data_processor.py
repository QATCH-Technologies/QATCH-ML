#!/usr/bin/env python3
"""
q_data_processor.py

Module for loading and processing time-series data, generating features,
clustering for dataset balancing, and handling Points of Interest (POIs).

Author: Paul MacNichol (paul.macnichol@qatchtech.com)
Date: 04-18-2025
Version: QModel.Ver3.0
"""
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
from ModelData import ModelData
import math

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

TARGET = "POI"
PLOTTING = False

""" The percentage of the run data to ignore from the head of a difference curve. """
HEAD_TRIM_PERCENTAGE = 0.05
""" The percentage of the run data to ignore from the tail of a difference curve. """
TAIL_TRIM_PERCENTAGE = 0.5

##########################################
# CHANGE THESE
##########################################
# This is a percentage setback from the initial drop application.  Increasing this value in the range (0,1) will
# move the left side of the correction zone further from the initial application of the drop as a percentage of the
# index where the drop is applied.
INIT_DROP_SETBACK = 0.000

# Adjust the detected left-bound index by subtracting a proportion (BASE_OFFSET) of the data length,
# effectively shifting the boundary left to better capture the true start of the drop.
BASE_OFFSET = 0.005

# This is the detection sensitivity for the dissipation and RF drop effects.  Increasing these values should independently
# increase how large a delta needs to be in order to be counted as a drop effect essentially correcting fewer deltas resulting
# in a less aggressive correction.
STARTING_THRESHOLD_FACTOR = 50
##########################################
# CHANGE THESE
##########################################


class QDataProcessor:
    """
    Processor for loading and transforming time-series POI data, including feature generation,
    dataset balancing, and clustering utilities.
    """

    @staticmethod
    def load_content(data_dir: str, num_datasets: int = np.inf, column: str = 'Dissipation') -> List[Tuple[str, str]]:
        """
        Recursively load pairs of data and POI files from the specified directory.

        Skips files without matching POI or lower-bound files and
        shuffles the returned list of valid file pairs.

        Args:
            data_dir (str): Path to the directory containing CSV data files.
            num_datasets (int, optional): Maximum number of datasets to return; returns all if infinite. Defaults to np.inf.
            column (str, optional): Column name to check existence. Defaults to 'Dissipation'.

        Returns:
            List[Tuple[str, str]]: List of tuples, each containing paths to a data file and its corresponding POI file.
        """
        if not os.path.exists(data_dir):
            logging.error("Data directory does not exist.")
            return []
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
                        logging.warning('POI file contains duplicate indices.')
                    else:
                        loaded_content.append(
                            (os.path.join(root, f), os.path.join(root, poi_file)))
        random.shuffle(loaded_content)
        if num_datasets == np.inf:
            return loaded_content
        return loaded_content[:num_datasets]

    @staticmethod
    def load_balanced_content(
        data_dir: str,
        num_datasets: Union[int, float] = np.inf,
        clusters: int = 3,
        opt: bool = False
    ) -> Union[Tuple[List[Tuple[str, str]], int], List]:
        """
        Load content and balance it by clustering based on feature vectors.

        Features are computed from POI and time-series statistics, then KMeans clustering
        balances the sample across clusters. Optionally, computes an optimal cluster count.

        Args:
            data_dir (str): Root directory for CSV files.
            num_datasets (int or float, optional): Desired number of samples. Defaults to np.inf.
            clusters (int, optional): Number of clusters if optimal clustering not requested. Defaults to 3.
            opt (bool, optional): If True, determine optimal number of clusters automatically. Defaults to False.

        Returns:
            Union[Tuple[List[Tuple[str, str]], int], List]: A tuple of the selected balanced samples and cluster count,
            or an empty list if no content is found.
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
                poi_mean = poi_df.values.mean()
                poi_std = poi_df.values.std()
                mean_diss = df["Dissipation"].mean()
                std_diss = df["Dissipation"].std()
                mean_freq = df["Resonance_Frequency"].mean()
                std_freq = df["Resonance_Frequency"].std()
                max_r_time = df['Relative_time'].max(
                ) if 'Relative_time' in df.columns else 0
                features.append([poi_mean, poi_std, mean_diss,
                                std_diss, mean_freq, std_freq, max_r_time])
                valid_content.append((data_file, poi_file))
            except Exception as e:
                logging.error(f"Could not read file {data_file}: {e}")
                continue

        if not features:
            return []
        features_arr = np.array(features)
        if opt:
            n_clusters, scores = QDataProcessor.compute_optimal_clusters(
                features_arr, min_clusters=2, max_clusters=50)
            logging.info(
                f"Optimal clusters: {n_clusters} with score {scores[n_clusters]}")
        else:
            n_clusters = min(clusters, len(features_arr))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_arr)

        cluster_dict: Dict[int, List[Tuple[str, str]]] = {}
        for lbl, pair in zip(labels, valid_content):
            cluster_dict.setdefault(lbl, []).append(pair)

        balanced_sample: List[Tuple[str, str]] = []
        if num_datasets == np.inf:
            min_count = min(len(lst) for lst in cluster_dict.values())
            for lst in cluster_dict.values():
                balanced_sample.extend(random.sample(lst, min_count))
        else:
            per_cluster = int(num_datasets) // n_clusters
            for lst in cluster_dict.values():
                balanced_sample.extend(lst if len(
                    lst) <= per_cluster else random.sample(lst, per_cluster))
            remaining = int(num_datasets) - len(balanced_sample)
            if remaining > 0:
                extras = [fp for lst in cluster_dict.values()
                          for fp in lst if fp not in balanced_sample]
                balanced_sample.extend(random.sample(
                    extras, min(remaining, len(extras))))
        random.shuffle(balanced_sample)

        if PLOTTING:
            plt.figure(figsize=(10, 6))
            plt.scatter(features_arr[:, 0], features_arr[:,
                        1], c=labels, cmap='viridis', s=50)
            sel_idx = [i for i, pair in enumerate(
                valid_content) if pair in balanced_sample]
            plt.scatter(features_arr[sel_idx, 0], features_arr[sel_idx,
                        1], facecolors='none', edgecolors='red', s=100)
            centers = kmeans.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200)
            plt.xlabel("Mean Dissipation")
            plt.ylabel("Std Dissipation")
            plt.title("Cluster Balancing")
            plt.show()

        return (balanced_sample, n_clusters) if num_datasets != np.inf else balanced_sample

    @staticmethod
    def compute_optimal_clusters(
        features: np.ndarray,
        min_clusters: int = 2,
        max_clusters: int = 10
    ) -> Tuple[int, Dict[int, float]]:
        """
        Determine the optimal number of clusters via silhouette analysis.

        Args:
            features (np.ndarray): 2D array of feature vectors.
            min_clusters (int, optional): Starting cluster count. Defaults to 2.
            max_clusters (int, optional): Maximum cluster count to test. Defaults to 10.

        Returns:
            Tuple[int, Dict[int, float]]: Best cluster count and dict of silhouette scores.

        Raises:
            ValueError: If the features array is empty.
        """
        if features.size == 0:
            raise ValueError("Features array is empty.")

        scores: Dict[int, float] = {}
        best_k, best_score = min_clusters, -1.0
        max_k = min(max_clusters, features.shape[0])
        for k in range(min_clusters, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            scores[k] = score
            if score > best_score:
                best_k, best_score = k, score

        if PLOTTING:
            plt.figure(figsize=(8, 4))
            plt.plot(list(scores.keys()), list(scores.values()), marker='o')
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette vs Clusters")
            plt.show()

        return best_k, scores

    @staticmethod
    def find_sampling_shift(df: pd.DataFrame) -> int:
        """
        Identify index where sampling rate significantly changes in the time series.

        Args:
            df (pd.DataFrame): DataFrame with a 'Relative_time' column.

        Returns:
            int: Index of first significant sampling shift, or -1 if none found.

        Raises:
            ValueError: If 'Relative_time' is missing.
        """
        if "Relative_time" not in df.columns:
            raise ValueError(
                "Input DataFrame must contain 'Relative_time' column.")
        deltas = df["Relative_time"].diff()
        rolling_avg = deltas.expanding(min_periods=2).mean()
        significant = (deltas - rolling_avg).abs() > 0.032
        indices = significant[significant].index
        return int(indices[0]) if not indices.empty else -1

    @staticmethod
    def compute_ocsvm_score(shift_series: pd.Series) -> np.ndarray:
        """
        Compute anomaly scores using One-Class SVM.

        Args:
            shift_series (pd.Series): Input series of shifts/baseline deviations.

        Returns:
            np.ndarray: Decision function scores from the One-Class SVM.

        Raises:
            ValueError: If the input series is empty.
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
        Compute the first-order Difference of Gaussians (DoG) for a column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            col (str): Column name to process.
            sigma (float, optional): Gaussian kernel standard deviation. Defaults to 2.

        Returns:
            pd.Series: DoG series aligned with the original index.

        Raises:
            ValueError: If column missing or sigma non-positive.
        """
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")
        result = gaussian_filter1d(df[col], sigma=sigma, order=1)
        return pd.Series(result, index=df.index)

    @staticmethod
    def compute_difference_curve(df: pd.DataFrame, difference_factor: int = 2) -> pd.Series:
        """
        Compute a custom difference curve combining resonance and dissipation.

        Args:
            df (pd.DataFrame): DataFrame with 'Relative_time', 'Resonance_Frequency', 'Dissipation' columns.
            difference_factor (int, optional): Scaling factor for dissipation term. Defaults to 2.

        Returns:
            pd.Series: Difference curve values.

        Raises:
            ValueError: If required columns are missing.
        """
        for col in ["Relative_time", "Resonance_Frequency", "Dissipation"]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing.")
        xs = df["Relative_time"]
        i = next((i for i, t in enumerate(xs) if t > 0.5), 0)
        j = next((i for i, t in enumerate(xs) if t > 2.5), len(xs))
        avg_freq = df["Resonance_Frequency"].iloc[i:j].mean()
        avg_diss = df["Dissipation"].iloc[i:j].mean()
        ys_d = (df["Dissipation"] - avg_diss) * avg_freq / 2
        ys_f = avg_freq - df["Resonance_Frequency"]
        return ys_f - difference_factor * ys_d

    @staticmethod
    def compute_rolling_baseline_and_shift(
        dog_series: pd.Series,
        window: int
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute rolling median baseline and shift for DoG series.

        Args:
            dog_series (pd.Series): DoG values.
            window (int): Window size for rolling median.

        Returns:
            Tuple[pd.Series, pd.Series]: Baseline series and shift series.

        Raises:
            ValueError: If window non-positive.
        """
        if window <= 0:
            raise ValueError("window must be > 0.")
        baseline = dog_series.rolling(
            window=window, center=True, min_periods=1).median()
        shift = dog_series - baseline
        return baseline, shift

    @staticmethod
    def compute_super_gradient(series: pd.Series) -> pd.Series:
        """
        Compute a smoothed numerical derivative (super gradient) using Savitzky-Golay.

        Args:
            series (pd.Series): Input data series.

        Returns:
            pd.Series: First derivative of smoothed data.
        """
        window = max(3, int(len(series) * 0.01) | 1)
        smoothed = savgol_filter(
            series, window_length=window, polyorder=1, deriv=0)
        gradient = savgol_filter(
            smoothed, window_length=window, polyorder=1, deriv=1)
        return pd.Series(gradient, index=series.index)

    @staticmethod
    def compute_cumulative_gradient(series_1: pd.Series, series_2: pd.Series) -> pd.Series:
        """
        Compute derivative of the sum of two series.

        Args:
            series_1 (pd.Series): First series.
            series_2 (pd.Series): Second series.

        Returns:
            pd.Series: Derivative of combined series.
        """
        combined = series_1.values + series_2.values
        deriv = savgol_filter(combined, window_length=25, polyorder=1, deriv=1)
        return pd.Series(deriv, index=series_1.index)

    @staticmethod
    def compute_graident(series: pd.Series) -> pd.Series:
        """
        Compute simple discrete gradient (difference).

        Args:
            series (pd.Series): Input series.

        Returns:
            pd.Series: Discrete difference of series.
        """
        return series.diff()

    @staticmethod
    def noise_filter(series: pd.Series) -> np.ndarray:
        """
        Apply a 2nd-order Butterworth low-pass filter and remove negative values.

        Args:
            series (pd.Series): Time-series data.

        Returns:
            np.ndarray: Filtered non-negative data.

        Raises:
            ValueError: If series empty or None.
        """
        if series is None or series.empty:
            logging.error("`series` cannot be an empty pd.Series.")
            raise ValueError("`series` cannot be an empty pd.Series.")
        fs = 300
        normal_cutoff = 2 / (0.5 * fs)
        b, a = butter(2, Wn=normal_cutoff, btype="lowpass", analog=False)
        filtered = filtfilt(b, a, series.values)
        return np.maximum(filtered, 0)

    @staticmethod
    def detrend(series: pd.Series) -> np.ndarray:
        """
        Remove linear trend from data.

        Args:
            series (pd.Series): Input data.

        Returns:
            np.ndarray: Detrended values.
        """
        return detrend(series.values)

    @staticmethod
    def generate_features(df: pd.DataFrame, live: bool = True) -> pd.DataFrame:
        """
        Generate a suite of features for modeling from raw time-series data.

        Includes difference curves, smoothing, DoG processing, SVM scores,
        super gradients, noise filters, and detrending.

        Args:
            df (pd.DataFrame): DataFrame containing required columns.
            live (bool): If True, excludes 'POI' column requirement.

        Returns:
            pd.DataFrame: DataFrame augmented with feature columns.

        Raises:
            ValueError: If required columns missing or DataFrame empty.
        """
        req_cols = ["Dissipation", "Resonance_Frequency", "Relative_time"]
        if not live:
            req_cols.append("POI")
        if df.empty or not all(c in df.columns for c in req_cols):
            raise ValueError(
                f"Input DataFrame must contain columns: {req_cols}")
        df = df[req_cols].copy()
        if df.empty:
            raise ValueError(
                "DataFrame is empty after selecting required columns.")

        df["Difference"] = QDataProcessor.compute_difference_curve(df)
        df["Cumulative"] = QDataProcessor.compute_cumulative_gradient(
            df["Dissipation"], df["Resonance_Frequency"])

        smooth_win = max(3, int(0.005 * len(df)))
        if smooth_win % 2 == 0:
            smooth_win += 1
        df["Difference_smooth"] = savgol_filter(
            df["Difference"], smooth_win, 3)
        df["Dissipation_smooth"] = savgol_filter(
            df["Dissipation"], smooth_win, 3)
        df["Resonance_Frequency_smooth"] = savgol_filter(
            df["Resonance_Frequency"], smooth_win, 3)

        # DoG and baseline processing
        for col in ["Dissipation", "Resonance_Frequency", "Difference"]:
            dog = QDataProcessor.compute_DoG(df, col=col)
            win = max(3, int(np.ceil(0.05 * len(df))))
            base, shift = QDataProcessor.compute_rolling_baseline_and_shift(
                dog, win)
            df[f"{col}_DoG"] = dog
            df[f"{col}_DoG_baseline"] = base
            df[f"{col}_DoG_shift"] = shift
            df[f"{col}_DoG_SVM_Score"] = QDataProcessor.compute_ocsvm_score(
                shift)

        # Super gradients and discrete gradients
        df["Super_Dissipation"] = QDataProcessor.compute_super_gradient(
            df["Dissipation"])
        df["Super_Difference"] = QDataProcessor.compute_super_gradient(
            df["Difference"])
        df["Super_Resonance_Frequency"] = QDataProcessor.compute_super_gradient(
            df["Resonance_Frequency"])
        df["Super_Cumulative"] = QDataProcessor.compute_super_gradient(
            df["Cumulative"])

        df["Gradient_Dissipation"] = QDataProcessor.compute_graident(
            df["Dissipation"])
        df["Gradient_Resonance_Frequency"] = QDataProcessor.compute_graident(
            df["Resonance_Frequency"])
        df["Gradient_Difference"] = QDataProcessor.compute_graident(
            df["Difference"])

        # Noise filters
        for col in ["Gradient_Dissipation", "Difference", "Resonance_Frequency"]:
            df[f"Filtered_{col}"] = QDataProcessor.noise_filter(df[col])
        df["Filtered_Cumulative"] = QDataProcessor.noise_filter(
            df["Cumulative"])

        # Detrend
        for col in ["Dissipation", "Difference", "Resonance_Frequency", "Cumulative"]:
            df[f"Detrend_{col}"] = QDataProcessor.detrend(df[col])

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def find_initial_fill_region(
        difference: pd.Series,
        relative_time: pd.Series
    ) -> Tuple[Tuple[float, int], Tuple[float, int]]:
        """
        Determine approximate start and end times (and indices) for the fill region.

        Args:
            difference (pd.Series): Difference curve values.
            relative_time (pd.Series): Corresponding time values.

        Returns:
            Tuple[Tuple[float, int], Tuple[float, int]]: ((lb_time, lb_idx), (rb_time, rb_idx)).
        """
        def region_search(difference, relative_time, mode, rb_time=-1.0, rb_index=-1):
            head_trim = int(len(difference) * HEAD_TRIM_PERCENTAGE)
            tail_trim = int(len(difference) * TAIL_TRIM_PERCENTAGE)
            rt = relative_time.iloc[head_trim:tail_trim]
            diff = difference.iloc[head_trim:tail_trim]
            trend = np.linspace(0, diff.max(), len(diff))
            adj_diff = diff - trend
            if mode == "left":
                if not (0 <= rb_index <= len(rt)):
                    return (0, 0)
                rt_sub = rt.iloc[:rb_index]
                diff_sub = diff.iloc[:rb_index]
                trend_sub = np.linspace(0, diff_sub.max(), len(diff_sub))
                adj_sub = diff_sub - trend_sub
                idx = int(np.argmin(adj_sub) + len(rt_sub) * BASE_OFFSET)
                idx += math.floor(INIT_DROP_SETBACK * idx)
                idx = min(idx, len(rt_sub) - 1)
                return rt_sub.iloc[idx], idx + head_trim
            elif mode == "right":
                idx = int(np.argmax(adj_diff) * 1.1)
                idx = min(idx, len(rt) - 1)
                return rt.iloc[idx], idx + head_trim
            else:
                raise ValueError(f"Invalid search bound: {mode}")

        rb_time, rb_idx = region_search(difference, relative_time, "right")
        lb_time, lb_idx = region_search(
            difference, relative_time, "left", rb_time, rb_idx)
        return (lb_time, lb_idx), (rb_time, rb_idx)

    @staticmethod
    def get_model_data_predictions(file_buffer: Union[str, any], n: int) -> np.ndarray:
        """
        Obtain binary POI predictions from ModelData API.

        Args:
            file_buffer (str or file-like): Path or stream of input data.
            n (int): Expected length of output vector.

        Returns:
            np.ndarray: Binary vector marking predicted POI indices.
        """
        def reset_file_buffer(buf):
            if isinstance(buf, str):
                return buf
            if hasattr(buf, 'seekable') and buf.seekable():
                buf.seek(0)
                return buf
            raise IOError("Cannot reset stream for processing.")

        model = ModelData()
        if isinstance(file_buffer, str):
            preds = model.IdentifyPoints(file_buffer)
        else:
            buf = reset_file_buffer(file_buffer)
            header = next(buf)
            if isinstance(header, bytes):
                header = header.decode()
            csv_cols = (2, 4, 6, 7) if "Ambient" in header else (2, 3, 5, 6)
            data = np.loadtxt(buf.readlines(), delimiter=",", usecols=csv_cols)
            preds = model.IdentifyPoints(data_path="QModel Passthrough",
                                         times=data[:, 0],
                                         freq=data[:, 2],
                                         diss=data[:, 3])
        points = []
        if isinstance(preds, list):
            for pt in preds:
                if isinstance(pt, int):
                    points.append(pt)
                elif isinstance(pt, list) and pt:
                    points.append(max(pt, key=lambda x: x[1])[0])
        out = np.zeros(n, dtype=int)
        for idx in points:
            if 0 <= idx < n:
                out[idx] = 1
        return out

    @staticmethod
    def process_poi(poi_file: str, length_df: int) -> pd.DataFrame:
        """
        Read POI indices from file and generate a labeled DataFrame.

        Args:
            poi_file (str): Path to POI CSV file.
            length_df (int): Length of target DataFrame for labels.

        Returns:
            pd.DataFrame: Single-column DataFrame with POI labels (1-6).

        Raises:
            ValueError: For invalid inputs or insufficient POI indices.
        """
        if not isinstance(poi_file, str) or not poi_file.strip():
            raise ValueError("poi_file must be a non-empty string.")
        if not os.path.exists(poi_file):
            raise ValueError(f"POI file '{poi_file}' does not exist.")
        if not isinstance(length_df, int) or length_df <= 0:
            raise ValueError("length_df must be a positive integer.")
        poi_df = pd.read_csv(poi_file, header=None)
        positions = poi_df[0].astype(int).values
        if len(positions) < 6:
            raise ValueError("POI file does not contain enough fill indices.")
        labels = np.zeros(length_df, dtype=int)
        for i, pos in enumerate(positions[:6]):
            labels[pos] = i+1
        return pd.DataFrame(labels)

    @staticmethod
    def process_data(file_buffer: Union[str, pd.DataFrame], live: bool = True) -> pd.DataFrame:
        """
        Load and fully process a data file or DataFrame into feature space.

        Args:
            file_buffer (str or pd.DataFrame): Input data source.
            live (bool): If True, excludes POI from required columns.

        Returns:
            pd.DataFrame: Processed DataFrame ready for modeling.

        Raises:
            IOError: If input type unsupported.
            ValueError: For missing required columns or empty data.
        """
        if isinstance(file_buffer, pd.DataFrame):
            df = file_buffer.copy()
        elif isinstance(file_buffer, str):
            df = pd.read_csv(file_buffer)
        else:
            raise IOError(f"Error processing file `{file_buffer}`.")

        required = ["Relative_time", "Dissipation", "Resonance_Frequency"]
        if not live:
            required.append("POI")
        if df.empty or not all(c in df.columns for c in required):
            raise ValueError("Dataset missing required columns or empty.")
        df = df[required].copy().reset_index(drop=True)
        df = QDataProcessor.generate_features(df, live=live)
        df.reset_index(drop=True, inplace=True)
        if 'Relative_time' in df.columns:
            df.drop(columns=['Relative_time'], inplace=True)
        df['ModelData_Guesses'] = QDataProcessor.get_model_data_predictions(
            file_buffer=file_buffer, n=len(df))
        return df.replace([np.inf, -np.inf], np.nan).fillna(0)
