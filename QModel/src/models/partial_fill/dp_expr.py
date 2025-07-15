import os
import random
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdp import rdp
from tqdm import tqdm


class DissipationFeaturePipeline:

    def __init__(self, exact_n: int = 10, min_duration: float = 120.0):
        """
        Args:
            exact_n:    Number of RDP key‐points per curve.
            min_duration: Minimum Relative_time span (in seconds) to include.
        """
        self.exact_n = exact_n
        self.min_duration = min_duration

    @staticmethod
    def load_content(data_dir: str,
                     num_datasets: int = np.inf) -> List[Tuple[str, str]]:
        if not os.path.exists(data_dir):
            logging.error("Data directory does not exist: %s", data_dir)
            return []

        loaded = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if not f.endswith(".csv") or f.endswith(("_poi.csv", "_lower.csv")):
                    continue
                data_path = os.path.join(root, f)
                df = pd.read_csv(data_path)
                if df['Relative_time'].max() > 0:
                    poi_path = data_path.replace(".csv", "_poi.csv")
                    if os.path.exists(poi_path):
                        loaded.append((data_path, poi_path))

        random.shuffle(loaded)
        if num_datasets == np.inf:
            return loaded
        return loaded[:int(num_datasets)]

    # ── RDP & feature helpers ─────────────────────────────────────────────────

    def _compute_endpoint_distances(self, time: np.ndarray, diss: np.ndarray) -> np.ndarray:
        pts = np.column_stack((time, diss))
        start, end = pts[0], pts[-1]
        vec = end - start
        norm2 = np.dot(vec, vec)
        if norm2 == 0:
            return np.zeros(len(pts))
        rel = pts - start
        proj = (rel @ vec) / norm2
        proj_pts = start + np.outer(proj, vec)
        return np.linalg.norm(pts - proj_pts, axis=1)

    def _find_epsilon_for_exact_n(self,
                                  time: np.ndarray,
                                  diss: np.ndarray,
                                  exact_n: int,
                                  tol: float = 1e-3,
                                  max_iters: int = 50) -> float:
        dists = self._compute_endpoint_distances(time, diss)
        lo, hi = 0.0, float(dists.max())
        best = {"diff": float("inf"), "eps": None}

        for _ in range(max_iters):
            mid = (lo + hi) / 2
            n_pts = len(rdp(np.column_stack((time, diss)), epsilon=mid))
            diff = abs(n_pts - exact_n)
            if diff < best["diff"]:
                best.update(diff=diff, eps=mid)
            if n_pts > exact_n:
                lo = mid
            elif n_pts < exact_n:
                hi = mid
            else:
                break
            if hi - lo < tol:
                break

        return best["eps"]

    def _general_shape_rdp_exact(self,
                                 time: np.ndarray,
                                 diss: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return exactly `self.exact_n` points (or as close as possible), padding/truncating if needed.
        """
        eps = self._find_epsilon_for_exact_n(time, diss, self.exact_n)
        pts = np.column_stack((time, diss))
        simplified = rdp(pts, epsilon=eps)
        t_key, d_key = simplified[:, 0], simplified[:, 1]

        # enforce exact length
        if len(t_key) < self.exact_n:
            pad_n = self.exact_n - len(t_key)
            t_key = np.concatenate([t_key, np.full(pad_n, t_key[-1])])
            d_key = np.concatenate([d_key, np.full(pad_n, d_key[-1])])
        elif len(t_key) > self.exact_n:
            t_key, d_key = t_key[:self.exact_n], d_key[:self.exact_n]

        return t_key, d_key

    def _extract_features(self,
                          data_path: str) -> pd.Series:
        """
        Read `data_path` (expects columns 'time' & 'dissipation'),
        normalize & scale, summarize with RDP→10 points,
        and return a Series of length 20.
        """
        df = pd.read_csv(data_path)
        t = df['Relative_time'].values
        d = df['Dissipation'].values

        # --- min–max normalize time to [0,1] by run length ---
        duration = t[-1] - t[0] if len(t) > 1 else 1.0
        t_norm = (t - t[0]) / duration

        # --- min–max scale dissipation to [0,1] ---
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            d_scaled = (d - d_min) / (d_max - d_min)
        else:
            d_scaled = np.zeros_like(d)  # flat curve → all zeros

        # --- RDP summarization on the normalized curve ---
        t_key, d_key = self._general_shape_rdp_exact(t_norm, d_scaled)
        feat = np.concatenate([t_key, d_key])

        idx = [f"time_{i+1}" for i in range(self.exact_n)] + \
              [f"diss_{i+1}" for i in range(self.exact_n)]
        return pd.Series(feat, index=idx, name=Path(data_path).stem)
    # ── Public API ────────────────────────────────────────────────────────────

    def build_feature_df(self,
                         data_dir: str,
                         num_datasets: int = np.inf) -> pd.DataFrame:
        """
        1) Load up to `num_datasets` (data,poi) tuples,
        2) Skip any whose Relative_time span < min_duration,
        3) Extract features from each remaining data CSV,
        4) Return a DataFrame (rows=datasets, cols=features).
        """
        files = self.load_content(data_dir, num_datasets)
        if not files:
            return pd.DataFrame()

        series_list = []
        for data_path, poi_path in tqdm(files, desc="Generating Features"):
            df = pd.read_csv(data_path)
            # compute true duration in seconds
            if len(df) > 1:
                duration = df['Relative_time'].iat[-1] - \
                    df['Relative_time'].iat[0]
            else:
                duration = 0.0

            if duration < self.min_duration:
                logging.info(
                    f"Skipping {data_path!r}: duration {duration:.1f}s < {self.min_duration}s")
                continue

            # now safe to extract
            series_list.append(self._extract_features(data_path))

        if not series_list:
            return pd.DataFrame()

        return pd.DataFrame(series_list)


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = DissipationFeaturePipeline(exact_n=10)
    df_feats = pipeline.build_feature_df(
        "content/dropbox_dump", num_datasets=5)
    print(df_feats.head())
    df_feats.to_csv("dissipation_features.csv")

    # ── Plot a few raw+RDP summaries ────────────────────────────────────────────
    import matplotlib.pyplot as plt

    # pick 5 random examples from the same set
    sample_pairs = pipeline.load_content(
        "content/dropbox_dump", num_datasets=50)
    random.seed(42)
    sample_pairs = random.sample(sample_pairs, k=min(5, len(sample_pairs)))

    fig, axes = plt.subplots(1, len(sample_pairs),
                             figsize=(4*len(sample_pairs), 4))
    if len(sample_pairs) == 1:
        axes = [axes]

    for ax, (data_path, _) in zip(axes, sample_pairs):
        # load & normalize
        df = pd.read_csv(data_path)
        t = df['Relative_time'].values
        d = df['Dissipation'].values
        t_norm = (t - t[0])/(t[-1]-t[0]) if len(t) > 1 else t*0
        d_min, d_max = d.min(), d.max()
        d_scaled = (d - d_min) / \
            (d_max-d_min) if d_max > d_min else np.zeros_like(d)

        # get key-points
        t_key, d_key = pipeline._general_shape_rdp_exact(t_norm, d_scaled)

        # plot
        ax.plot(t_norm, d_scaled, label="normalized curve", alpha=0.4)
        ax.plot(t_key,    d_key,    "-o", label="10-pt RDP")
        ax.set_title(Path(data_path).stem, fontsize=10)
        ax.set_xlabel("time (scaled)")
        ax.set_ylabel("dissipation (scaled)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.show()
