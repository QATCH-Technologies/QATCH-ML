import argparse
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DryingDetection:
    """
    Detect when both Resonance_Frequency and Dissipation
    have flattened out (i.e. sensor is dry), using a rolling window.
    """

    def __init__(self,
                 window_size: int = 30,
                 sigma_stable_freq: float = 0.02,
                 sigma_stable_diss: float = 0.02,
                 flat_slope_eps:    float = 0.005):
        """
        Args:
            window_size:        Number of samples in rolling window
            sigma_stable_freq:  Max stddev (on normalized RF) to consider "stable"
            sigma_stable_diss:  Max stddev (on normalized Diss) to consider "stable"
            flat_slope_eps:     Max |slope| (on normalized data) to consider "flat"
        """
        self.win_n = int(window_size)
        self.freq_w = deque(maxlen=self.win_n)
        self.diss_w = deque(maxlen=self.win_n)
        self.sigma_stable_freq = float(sigma_stable_freq)
        self.sigma_stable_diss = float(sigma_stable_diss)
        self.flat_eps = float(flat_slope_eps)
        self._dried = False

    def reset(self):
        """Clear all history and allow a new drying detection."""
        self.freq_w.clear()
        self.diss_w.clear()
        self._dried = False

    @property
    def is_dry(self) -> bool:
        """True once the dry condition has been met."""
        return self._dried

    def _compute_slope(self, arr: np.ndarray) -> float:
        """Return slope of best‐fit line through (0…N-1, arr)."""
        if arr.size < 2:
            return 0.0
        x = np.arange(arr.size)
        # polyfit will handle constant arrays by returning zero slope
        m, _ = np.polyfit(x, arr, 1)
        return float(m)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Min–max normalize, or zeros if constant."""
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def update(self, resonance_freq: float, dissipation: float) -> bool:
        """
        Append one new sample. Returns True exactly once when:
          - stddev(norm_rf) < sigma_stable_freq
          - stddev(norm_diss) < sigma_stable_diss
          - |slope(norm_rf)| < flat_slope_eps
          - |slope(norm_diss)| < flat_slope_eps

        After True is returned once, further calls stay False until reset().
        """
        if self._dried:
            return False

        # guard against NaNs
        if not np.isfinite(resonance_freq) or not np.isfinite(dissipation):
            return False

        # collect
        self.freq_w.append(resonance_freq)
        self.diss_w.append(dissipation)

        # not enough data yet
        if len(self.freq_w) < self.win_n:
            return False

        # to numpy
        raw_f = np.array(self.freq_w, dtype=float)
        raw_d = np.array(self.diss_w, dtype=float)

        # normalize
        nf = self._normalize(raw_f)
        nd = self._normalize(raw_d)

        # compute metrics
        sigma_f = float(np.nanstd(nf))
        sigma_d = float(np.nanstd(nd))
        slope_f = self._compute_slope(nf)
        slope_d = self._compute_slope(nd)

        # check all 4 conditions
        if (sigma_f < self.sigma_stable_freq and
            sigma_d < self.sigma_stable_diss and
            abs(slope_f) < self.flat_eps and
                abs(slope_d) < self.flat_eps):
            self._dried = True
            return True

        return False


def main(csv_path: str, interval: float = 0.1):
    df = pd.read_csv(csv_path)
    # 2) define your total drop
    # total_drop_rf = 5.0   # RF will decrease by 5 units over the whole run
    # total_drop_diss = 2.0   # Dissipation will decrease by 2 units

    # # 3) build a linear trend
    # n = len(df)
    # trend_rf = np.linspace(0, total_drop_rf,   n)
    # trend_diss = np.linspace(0, total_drop_diss, n)

    # # 4) subtract the trend (so the data goes downward)
    # df['Resonance_Frequency'] = df['Resonance_Frequency'] - trend_rf
    # df['Dissipation'] = df['Dissipation'] - trend_diss
    if not {'Resonance_Frequency', 'Dissipation'}.issubset(df.columns):
        raise ValueError(
            "CSV must have 'Resonance_Frequency' and 'Dissipation' columns")

    det = SensorDryingDetector(
        window_size=50,
        sigma_stable_freq=0.15,
        sigma_stable_diss=0.15,
        flat_slope_eps=0.005
    )

    plt.ion()
    fig, (ax_f, ax_d, ax_s) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    ax_f.set_ylabel("Resonance_Frequency")
    ax_d.set_ylabel("Dissipation")
    ax_s.set_ylabel("State")
    ax_s.set_yticks([0, 1])
    ax_s.set_yticklabels(["DRYING", "DRY"])
    ax_s.set_xlabel("Sample #")

    xdata, fdata, ddata, stdata = [], [], [], []

    for idx, row in df.iterrows():
        xdata.append(idx)
        f = row['Resonance_Frequency']
        d = row['Dissipation']
        fdata.append(f)
        ddata.append(d)

        fired, state = det.update(f, d)
        stdata.append(0 if state == "DRYING" else 1)

        # redraw
        ax_f.clear()
        ax_f.plot(xdata, fdata, '-o', markersize=3)
        ax_d.clear()
        ax_d.plot(xdata, ddata, '-o', markersize=3)
        ax_s.clear()
        ax_s.plot(xdata, stdata, '-o', markersize=3)
        ax_s.set_yticks([0, 1])
        ax_s.set_yticklabels(["DRYING", "DRY"])
        fig.suptitle(f"Sample {idx}: State={state}" +
                     (" ← DRY EVENT!" if fired else ""))
        plt.pause(interval)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    data_file = r"content/dropbox_dump/02480/MM240625Y4_PBS_1_3rd.csv"
    interval = 0.01

    main(data_file, interval)
