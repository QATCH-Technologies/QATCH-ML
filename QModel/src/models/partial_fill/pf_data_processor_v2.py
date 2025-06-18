
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
from sklearn.feature_selection import mutual_info_regression
from scipy.signal import savgol_filter


class PFDataProcessor:
    SAMPLE_FACTOR = 400

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
    def generate_features(dataframe: pd.DataFrame,
                          sampling_rate: float = 1.0,
                          detected_poi1: int = None) -> Dict[str, float]:

        feat_columns = ["Dissipation", "Resonance_Frequency"]
        dataframe = dataframe[feat_columns].copy()
        feats = pd.DataFrame()
        # feats["run_length"] = [len(dataframe)]

        n = len(dataframe)
        win = max(3, int(np.ceil(0.05 * n)))
        if win % 2 == 0:
            win += 1
        diss = dataframe["Dissipation"].values
        rf = dataframe["Resonance_Frequency"].values
        # diss = savgol_filter(
        #     diss, window_length=win, polyorder=2)
        # rf = savgol_filter(
        #     rf, window_length=win, polyorder=2)

        n_baseline = max(1, int(len(dataframe) * 0.05))
        baseline_diss = diss[:n_baseline].mean()
        baseline_rf = rf[:n_baseline].mean()
        diss_re = diss - baseline_diss
        rf_re = rf - baseline_rf
        rf_flipped = -rf_re
        num_points = PFDataProcessor.SAMPLE_FACTOR
        idxs = np.linspace(0, n-1, num=num_points, dtype=int)
        diss_sel = diss_re[idxs]
        rf_sel = rf_flipped[idxs]
        d_min, d_max = diss_sel.min(),    diss_sel.max()
        r_min, r_max = rf_sel.min(),      rf_sel.max()

        diss_scaled = (diss_sel - d_min) / (d_max - d_min)
        rf_scaled = (rf_sel - r_min) / (r_max - r_min)

        sampled_df = pd.DataFrame({
            "Dissipation":         diss_scaled,
            "Resonance_Frequency": rf_scaled
        })

        # plt.figure(figsize=(8, 4))
        # plt.plot(diss_scaled,
        #          linestyle='-', label="Diss [200 pts]")
        # plt.plot(rf_scaled,
        #          linestyle='-', label="RF   [200 pts]")
        # plt.legend(loc="upper right")
        # plt.xlabel("Sample # (evenly spaced)")
        # plt.ylabel("Scaled & rebaselined value")
        # plt.title("200-Point Representation of Each Curve")
        # plt.tight_layout()
        # plt.show()
        feats = pd.concat(
            [feats, PFDataProcessor._curve_stats(sampled_df)], axis=1)
        feats = pd.concat(
            [feats, PFDataProcessor._peak_features(sampled_df)], axis=1)
        feats = pd.concat(
            [feats, PFDataProcessor._curve_dynamics(sampled_df)], axis=1)
        feats = pd.concat([feats, PFDataProcessor._fft_features(
            sampled_df, sampling_rate)], axis=1)
        feats = pd.concat(
            [feats, PFDataProcessor._cross_corr_features(sampled_df)], axis=1)

        return feats


if __name__ == "__main__":
    content = PFDataProcessor.load_content(
        os.path.join('content', 'static', 'test'))
    for data_file, poi_file in content:
        dataframe = pd.read_csv(data_file)
        pois = pd.read_csv(poi_file, header=None).values
        features = PFDataProcessor.generate_features(
            dataframe, detected_poi1=pois[0])
