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


class PFDataProcessor:
    @staticmethod
    def load_content(data_dir: str,
                     num_datasets: int = np.inf,
                     column: str = 'Dissipation') -> List[tuple]:
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
    def _curve_stats(curve: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(curve)),
            "std": float(np.std(curve)),
            "min": float(np.min(curve)),
            "max": float(np.max(curve)),
            "skew": float(skew(curve)),
            "kurtosis": float(kurtosis(curve)),
        }

    @staticmethod
    def _peak_features(curve: np.ndarray) -> Dict[str, float]:
        peaks, _ = find_peaks(curve)
        if not peaks.size:
            return {"peak_count": 0.0, "mean_prominence": 0.0, "mean_width": 0.0}
        prom, _, _ = peak_prominences(curve, peaks)
        widths, _, _, _ = peak_widths(curve, peaks, rel_height=0.5)
        mask = widths > 0
        if not mask.any():
            return {"peak_count": 0.0, "mean_prominence": 0.0, "mean_width": 0.0}
        return {
            "peak_count": float(mask.sum()),
            "mean_prominence": float(prom[mask].mean()),
            "mean_width": float(widths[mask].mean()),
        }

    @staticmethod
    def _curve_dynamics(curve: np.ndarray) -> Dict[str, float]:
        diffs = np.diff(curve)
        auc = float(np.trapz(curve))
        return {
            "slope_max": float(diffs.max()) if diffs.size else 0.0,
            "slope_min": float(diffs.min()) if diffs.size else 0.0,
            "slope_mean": float(diffs.mean()) if diffs.size else 0.0,
            "auc": auc,
        }

    @staticmethod
    def _fft_features(curve: np.ndarray,
                      sampling_rate: float = 1.0) -> Dict[str, float]:
        n = len(curve)
        if n < 2:
            return {"fft_dominant_freq": 0.0, "fft_spectral_centroid": 0.0}
        freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
        vals = np.fft.rfft(curve - curve.mean())
        mags = np.abs(vals)
        idx = mags.argmax()
        dom = float(freqs[idx])
        tot = mags.sum()
        cent = float((freqs * mags).sum() / tot) if tot > 0 else 0.0
        return {"fft_dominant_freq": dom, "fft_spectral_centroid": cent}

    @staticmethod
    def _cross_corr_features(curve1: np.ndarray,
                             curve2: np.ndarray) -> Dict[str, float]:
        c1, c2 = curve1 - curve1.mean(), curve2 - curve2.mean()
        corr = np.correlate(c1, c2, mode='full')
        return {"corr_max": float(corr.max()),
                "corrcoef": float(np.corrcoef(curve1, curve2)[0, 1])}

    @staticmethod
    def _generate_features(df_slice: pd.DataFrame,
                           label: str,
                           sampling_rate: float = 1.0,
                           full_curve: np.ndarray = None,
                           detected_poi1: int = None) -> Dict[str, float]:
        curve = df_slice.values.flatten()
        feats = {"segment_label": label, "slice_length": len(curve)}
        if detected_poi1 is not None:
            feats["time_since_detected_fill"] = float(
                len(curve) - detected_poi1)
        else:
            feats["time_since_detected_fill"] = 0.0
        feats.update(PFDataProcessor._curve_stats(curve))
        feats.update(PFDataProcessor._peak_features(curve))
        feats.update(PFDataProcessor._curve_dynamics(curve))
        feats.update(PFDataProcessor._fft_features(curve, sampling_rate))
        # if full_curve is not None:
        #     feats.update(
        #         PFDataProcessor._cross_corr_features(curve, full_curve))
        return feats

    @staticmethod
    def _slice_and_summarize(df: pd.DataFrame,
                             poi_indices: np.ndarray,
                             sampling_rate: float = 1.0,
                             plot: bool = False) -> pd.DataFrame:
        """
        Slice df at random points between POI intervals and full length,
        simulate detection error on POI1, subtract baseline, compute features, and optionally plot.
        """
        full_curve = df.values.flatten()
        # POI boundaries
        pois = np.unique(poi_indices.astype(int))
        pois = pois[(pois > 0) & (pois < len(df))]

        # simulate detected POI1 within ±5 points
        detected_poi1 = pois[0] + random.randint(0, 5) if pois.size else 0

        # define slice intervals
        starts = [0] + pois.tolist()
        ends = pois.tolist() + [len(df)]
        # choose random cutpoint in each interval
        random_cuts = [random.randint(s + 1, e) if e - s > 1 else e
                       for s, e in zip(starts, ends)]

        # baseline from the first (no_fill) slice
        baseline_end = random_cuts[0]
        baseline_curve = full_curve[:baseline_end]
        baseline_mean = float(np.mean(baseline_curve))
        # subtract baseline from full curve
        full_curve_rel = full_curve - baseline_mean

        if plot:
            plt.figure(figsize=(8, 4))
            plt.plot(full_curve_rel, label='Baseline-subtracted full curve')
            for cut in random_cuts[:-1]:
                plt.axvline(cut, linestyle='--', color='gray')
            plt.title('Relativistic Dissipation Curve with Random Slice Points')
            plt.xlabel('Index')
            plt.ylabel('Dissipation (rel)')
            plt.tight_layout()
            plt.show()

        # generate features per slice
        records = []
        for i, cut in enumerate(random_cuts):
            label = ('no_fill' if i == 0 else
                     'full_fill' if i == len(pois) else
                     f'poi{i}_partial')
            df_slice = df.iloc[:cut]
            # subtract baseline
            df_slice_rel = df_slice - baseline_mean

            if plot:
                plt.figure(figsize=(6, 3))
                plt.plot(df_slice_rel.values.flatten(), label=label)
                plt.title(f'Slice {i}: {label} (cut @ {cut})')
                plt.xlabel('Index')
                plt.ylabel('Dissipation (rel)')
                plt.tight_layout()
                plt.show()

            feat = PFDataProcessor._generate_features(
                df_slice_rel,
                label,
                sampling_rate,
                full_curve_rel,
                detected_poi1
            )
            records.append(feat)

        return pd.DataFrame.from_records(records)

    @classmethod
    def build_feature_vectors(cls,
                              data_dir: str,
                              num_datasets: int = np.inf,
                              column: str = 'Dissipation',
                              sampling_rate: float = 1.0,
                              plot_slices: bool = False) -> pd.DataFrame:
        content = cls.load_content(data_dir, num_datasets, column)
        frames = []
        for data_path, poi_path in tqdm(content, desc="Processing curves"):
            df = pd.read_csv(data_path)
            poi_vals = pd.read_csv(poi_path, header=None).values.flatten()
            df_col = df[[column]]
            feats = cls._slice_and_summarize(
                df_col, poi_vals, sampling_rate, plot_slices)
            feats['dataset_id'] = os.path.basename(data_path)
            frames.append(feats)
        return pd.concat(frames, ignore_index=True)

    @classmethod
    def build_training_data(cls,
                            data_dir: str,
                            num_datasets: int = np.inf,
                            column: str = 'Dissipation',
                            sampling_rate: float = 1.0,
                            plot_slices: bool = False
                            ) -> Tuple[pd.DataFrame, pd.Series]:
        df_feats = cls.build_feature_vectors(
            data_dir, num_datasets, column, sampling_rate, plot_slices)
        labels = ['no_fill'] + \
            [f'poi{i}_partial' for i in range(1, 6)] + ['full_fill']
        mapping = {lbl: idx for idx, lbl in enumerate(labels)}
        df_feats['target'] = df_feats['segment_label'].map(mapping)
        y = df_feats['target']
        X = df_feats.drop(columns=['segment_label', 'target', 'dataset_id'])
        return X, y


if __name__ == "__main__":
    DATA_DIR = r"C:\Users\paulm\dev\QATCH-ML\content\static"
    # plot first dataset with simulated detection
    X_dummy, y_dummy = PFDataProcessor.build_training_data(
        DATA_DIR, num_datasets=1, plot_slices=True, sampling_rate=5.0)
    # full training data
    X, y = PFDataProcessor.build_training_data(DATA_DIR)

    # Correlation per class
    y_onehot = pd.get_dummies(y, prefix='class')
    classes = sorted(y.unique())
    n = len(classes)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    for idx, cls in enumerate(classes):
        ax = axes.flat[idx]
        corr_cls = X.corrwith(y_onehot[f'class_{cls}'])
        corr_cls.sort_values(ascending=False).plot.bar(ax=ax)
        ax.set_title(f'Feature Corr with class {cls}')
        ax.set_ylabel('Pearson r')
    # remove extra axes
    for j in range(n, rows * cols):
        fig.delaxes(axes.flat[j])
    fig.tight_layout()
    plt.show()

    # Mutual information overall
    mi = mutual_info_regression(X.fillna(0), y)
    mi_series = pd.Series(mi, index=X.columns)
    plt.figure(figsize=(10, 6))
    mi_series.sort_values(ascending=False).plot.bar()
    plt.title("Mutual Information of Features with Target")
    plt.ylabel("Mutual Information")
    plt.tight_layout()
    plt.show()
