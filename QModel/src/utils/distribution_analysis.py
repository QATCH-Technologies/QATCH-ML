import os
import glob
import json
import pandas as pd
import numpy as np


def build_poi_summary(base_dir, interval_size=10):
    """
    Walks each numeric subfolder under base_dir, reads the data and *_poi.csv,
    shifts POI times so POI1 → 0, and computes per-interval summary stats.
    Returns a dict keyed by the interval’s lower bound (0, 10, 20, …).
    """
    raw = {}
    for sub in os.listdir(base_dir):
        subdir = os.path.join(base_dir, sub)
        if not os.path.isdir(subdir):
            continue

        all_csv = glob.glob(os.path.join(subdir, "*.csv"))
        pois = [f for f in all_csv if f.lower().endswith("_poi.csv")]
        data = [f for f in all_csv if f not in pois]
        if not data or not pois:
            continue

        df = pd.read_csv(data[0])
        times = df["Relative_time"].values
        lower = int(times.max() // interval_size) * interval_size

        # load POI indices
        poi_df = pd.read_csv(pois[0], header=None)
        if poi_df.shape[1] == 1:
            idx = poi_df.iloc[:, 0].astype(int).values
        else:
            idx = poi_df.iloc[0].astype(int).values

        abs_poi = times[idx]
        shifted = abs_poi - abs_poi[0]  # POI1→0

        raw.setdefault(lower, []).append(shifted)

    # compute stats
    summary = {}
    for lower, runs in raw.items():
        mat = np.vstack(runs)  # shape (n_runs, 6)
        poi_stats = {}
        for i in range(mat.shape[1]):
            arr = mat[:, i]
            poi_stats[i+1] = {
                "min":    float(arr.min()),
                "median": float(np.median(arr)),
                "mean":   float(arr.mean()),
                "std":    float(arr.std()),
                "max":    float(arr.max()),
            }
        summary[lower] = {
            "interval": f"{lower}-{lower+interval_size}",
            "n_runs":   mat.shape[0],
            "poi":      poi_stats
        }
    return summary


def get_poi_stats_for_time(summary, t, interval_size=10):
    """Lookup the stats bucket for relative_time t."""
    lower = int(t // interval_size) * interval_size
    return summary.get(lower)


if __name__ == "__main__":
    base_dir = r"C:\Users\paulm\dev\QATCH-ML\content\dropbox_dump"
    summary = build_poi_summary(base_dir)

    # Write out JSON
    out_path = "poi_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary for {len(summary)} intervals to {out_path}")

    # Example load-back
    # with open(out_path, "r") as f:
    #     loaded_summary = json.load(f)
    # stats_for_123 = get_poi_stats_for_time(loaded_summary, 123)
    # print(stats_for_123)
