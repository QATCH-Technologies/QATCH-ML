
import os
import logging
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, peak_prominences, peak_widths
import matplotlib.pyplot as plt


class PFDataProcessor:
    @staticmethod
    def load_content(data_dir: str,
                     num_datasets: int = np.inf) -> List[tuple]:
        """
        Returns a list of (data_csv_path, poi_csv_path) tuples.
        """
        if not os.path.exists(data_dir):
            logging.error("Data directory does not exist.")
            return []

        loaded_content = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if not f.endswith(".csv") or f.endswith(("_poi.csv", "_lower.csv")):
                    continue
                data_file = os.path.join(root, f)
                poi_file = data_file.replace(".csv", "_poi.csv")
                if not os.path.exists(poi_file):
                    continue
                loaded_content.append((data_file, poi_file))

        random.shuffle(loaded_content)
        if num_datasets == np.inf:
            return loaded_content
        return loaded_content[:num_datasets]

    @staticmethod
    def _curve_stats(df: pd.DataFrame) -> pd.DataFrame:
        stats_results = pd.DataFrame()

        for col in df.columns:
            curve = df[col].values
            stats_results[f"{col}_mean"] = [float(np.mean(curve))]
            stats_results[f"{col}_std"] = [float(np.std(curve))]
            stats_results[f"{col}_min"] = [float(np.min(curve))]
            stats_results[f"{col}_max"] = [float(np.max(curve))]
            stats_results[f"{col}_skew"] = [float(skew(curve))]
            stats_results[f"{col}_kurtosis"] = [float(kurtosis(curve))]
        return stats_results

    @staticmethod
    def _peak_features(df: pd.DataFrame) -> pd.DataFrame:
        peak_results = pd.DataFrame()
        for col in df.columns:
            curve = df[col].values
            peaks, _ = find_peaks(curve)
            if not peaks.size:
                peak_results[f"{col}_peak_count"] = [0.0]
                peak_results[f"{col}_mean_prominence"] = [0.0]
                peak_results[f"{col}_mean_width"] = [0.0]
            else:
                prom, _, _ = peak_prominences(curve, peaks)
                widths, _, _, _ = peak_widths(curve, peaks, rel_height=0.5)
                mask = widths > 0
                peak_results[f"{col}_peak_count"] = [float(mask.sum())]
                peak_results[f"{col}_mean_prominence"] = [
                    float(prom[mask].mean()) if mask.any() else 0.0]
                peak_results[f"{col}_mean_width"] = [
                    float(widths[mask].mean()) if mask.any() else 0.0]
        return peak_results

    @staticmethod
    def _curve_dynamics(df: pd.DataFrame) -> pd.DataFrame:
        dynamics_results = pd.DataFrame()
        for col in df.columns:
            curve = df[col].values
            diffs = np.diff(curve)
            auc = float(np.trapz(curve))
            dynamics_results[f"{col}_slope_max"] = [
                float(diffs.max()) if diffs.size else 0.0]
            dynamics_results[f"{col}_slope_min"] = [
                float(diffs.min()) if diffs.size else 0.0]
            dynamics_results[f"{col}_slope_mean"] = [
                float(diffs.mean()) if diffs.size else 0.0]
            dynamics_results[f"{col}_auc"] = [auc]
        return dynamics_results

    @staticmethod
    def _fft_features(df: pd.DataFrame, sampling_rate: float = 1.0) -> pd.DataFrame:
        fft_results = pd.DataFrame()
        for col in df.columns:
            curve = df[col].values
            n = len(curve)
            if n < 2:
                fft_results[f"{col}_fft_dominant_freq"] = [0.0]
                fft_results[f"{col}_fft_spectral_centroid"] = [0.0]
            else:
                freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
                vals = np.fft.rfft(curve - curve.mean())
                mags = np.abs(vals)
                idx = mags.argmax()
                dom = float(freqs[idx])
                tot = mags.sum()
                cent = float((freqs * mags).sum() / tot) if tot > 0 else 0.0
                fft_results[f"{col}_fft_dominant_freq"] = [dom]
                fft_results[f"{col}_fft_spectral_centroid"] = [cent]
        return fft_results

    @staticmethod
    def _cross_corr_features(df: pd.DataFrame) -> pd.DataFrame:
        corr_results = pd.DataFrame()
        columns = df.columns
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:
                    curve1 = df[col1].values
                    curve2 = df[col2].values
                    c1, c2 = curve1 - curve1.mean(), curve2 - curve2.mean()
                    corr = np.correlate(c1, c2, mode='full')
                    corr_max = float(corr.max())
                    corrcoef = float(np.corrcoef(curve1, curve2)[0, 1])
                    corr_results[f"{col1}_vs_{col2}_corr_max"] = [corr_max]
                    corr_results[f"{col1}_vs_{col2}_corrcoef"] = [corrcoef]

        return corr_results

    @staticmethod
    def _sample_interval_features(df: pd.DataFrame,
                                  intervals: List[float] = None) -> pd.DataFrame:
        """
        Sample each column of `df` at the given fractional positions
        (e.g. 0.0, 0.1, ..., 1.0) and return one‐row DataFrame of those values.
        """
        if intervals is None:
            # 0%, 10%, 20%, …, 100%
            intervals = [i / 10 for i in range(11)]

        n = len(df)
        samples = {}
        for col in df.columns:
            for p in intervals:
                # index at fraction p of the way through (clamped to valid range)
                idx = min(int(round(p * (n - 1))), n - 1)
                samples[f"{col}_p{int(p * 100)}"] = float(df[col].iloc[idx])

        return pd.DataFrame([samples])

    @staticmethod
    def generate_features(dataframe: pd.DataFrame,
                          sampling_rate: float = 1.0,
                          detected_poi1=None) -> pd.DataFrame:
        feat_columns = ["Dissipation", "Resonance_Frequency", "Relative_time"]
        df = dataframe[feat_columns].copy()

        if detected_poi1 is None:
            idx = None
        else:
            flat = np.atleast_1d(detected_poi1).flat[0]
            idx = int(flat)

        meas_cols = ["Dissipation", "Resonance_Frequency"]
        if idx is not None and idx > 0:
            baseline = df[meas_cols].iloc[:idx].mean()
        else:
            baseline = df[meas_cols].mean()
        df[meas_cols] = df[meas_cols].subtract(baseline)
        time_scale = 30 * 60
        df["Relative_time"] = df["Relative_time"] / time_scale
        feats = pd.DataFrame({"run_length": [len(df)]})

        if idx is not None and 0 <= idx < len(df):
            last_t = df["Relative_time"].iloc[-1]   # pure scalar
            fill_t = df["Relative_time"].iloc[idx]  # pure scalar
            delta_norm = last_t - fill_t
        else:
            delta_norm = 0.0
        delta_norm = float(np.clip(delta_norm, 0.0, 1.0))
        feats["time_since_detected_fill"] = [delta_norm]
        feats = pd.concat([feats, PFDataProcessor._curve_stats(df)], axis=1)
        feats = pd.concat([feats, PFDataProcessor._peak_features(df)], axis=1)
        feats = pd.concat([feats, PFDataProcessor._curve_dynamics(df)], axis=1)
        feats = pd.concat(
            [feats, PFDataProcessor._fft_features(df, sampling_rate)], axis=1)
        feats = pd.concat(
            [feats, PFDataProcessor._cross_corr_features(df)], axis=1)
        feats = pd.concat([feats,
                           PFDataProcessor._sample_interval_features(df)],
                          axis=1)

        return feats


if __name__ == "__main__":
    content = PFDataProcessor.load_content(
        os.path.join('content', 'static', 'test'))
    for data_file, poi_file in content:
        dataframe = pd.read_csv(data_file)
        pois = pd.read_csv(poi_file, header=None).values
        features = PFDataProcessor.generate_features(
            dataframe, detected_poi1=pois[0])
