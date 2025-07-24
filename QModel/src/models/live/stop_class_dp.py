from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.preprocessing import RobustScaler, StandardScaler
import tensorflow as tf
from tqdm import tqdm
# ─── parameters & mappings ─────────────────────────────────────────────────────

WIN_SIZE = 32
DS_FACTOR = 5
TARGET_POIS = {
    # 1,4, 5,
    6}
# map original label → contiguous class id (0 = background)
LABEL_MAP = {0: 0,
             #  1: 0, 4: 0, 5: 0,
             6: 1}

# ─── existing preprocessing funcs (unchanged) ───────────────────────────────────


def downsample_and_rebaseline(
    df: pd.DataFrame,
    factor: int = 5,
    window_size: int = WIN_SIZE,
    diss_col: str = "Dissipation",
    rf_col: str = "Resonance_Frequency"
) -> pd.DataFrame:
    df_ds = df.iloc[::factor].reset_index(drop=True)
    base_d = df_ds[diss_col].iloc[:window_size].mean()
    base_rf = df_ds[rf_col].iloc[:window_size].mean()
    df_ds[diss_col] = df_ds[diss_col] - base_d
    # flip RF so it increases
    df_ds[rf_col] = -(df_ds[rf_col] - base_rf)
    return df_ds


def make_windows(df: pd.DataFrame, window_size: int = 50) -> List[pd.DataFrame]:
    windows = []
    for start in range(0, len(df) - window_size + 1, window_size):
        win = df.iloc[start:start + window_size].reset_index(drop=True)
        windows.append(win)
    return windows

# ─── modified labeling (only keep POIs 1,4,5,6) ─────────────────────────────────


def label_windows(
    num_windows: int,
    poi_indices: np.ndarray,
    factor: int,
    window_size: int,
    target_pois: set
) -> List[int]:
    """
    Return a list of length `num_windows`:
      - if a window contains POI i∈target_pois, label = i
      - else label = 0
    """
    poi_ds = poi_indices // factor
    labels = []
    for w in range(num_windows):
        start = w * window_size
        end = start + window_size
        # only consider POIs in {1,4,5,6}
        hits = [
            i for i, ds in enumerate(poi_ds, start=1)
            if start <= ds < end and i in target_pois
        ]
        labels.append(hits[0] if hits else 0)
    return labels

# ─── feature extractor (unchanged) ──────────────────────────────────────────────


class FeatureExtractor:
    def __init__(self,
                 window_size: int = 100,
                 scaler_cls=StandardScaler):
        self.window_size = window_size
        self.scalers = {
            "Relative_time":        scaler_cls(),
            "Dissipation":          scaler_cls(),
            "Resonance_Frequency":  scaler_cls(),
        }

    def fit_scalers(self,
                    csv_paths: List[Path],
                    downsample_factor: int = 5):
        all_t, all_d, all_rf = [], [], []
        for path in csv_paths:
            df = pd.read_csv(path)
            df_ds = downsample_and_rebaseline(
                df, factor=downsample_factor,
                window_size=self.window_size
            )
            all_t.append(df_ds[["Relative_time"]])
            all_d.append(df_ds[["Dissipation"]])
            all_rf.append(df_ds[["Resonance_Frequency"]])
        big_t = pd.concat(all_t, axis=0)
        big_d = pd.concat(all_d, axis=0)
        big_rf = pd.concat(all_rf, axis=0)
        self.scalers["Relative_time"].fit(big_t)
        self.scalers["Dissipation"].fit(big_d)
        self.scalers["Resonance_Frequency"].fit(big_rf)

    def transform_window(
        self,
        win: pd.DataFrame
    ) -> np.ndarray:
        df = win.copy()
        for col in ["Relative_time", "Dissipation", "Resonance_Frequency"]:
            df[col] = self.scalers[col].transform(df[[col]]).ravel()

        # simple interaction features
        df["dissipation_change"] = df["Dissipation"].diff().fillna(0)
        df["rf_change"] = df["Resonance_Frequency"].diff().fillna(0)
        df["diss_x_rf"] = df["Dissipation"] * df["Resonance_Frequency"]
        df["change_prod"] = df["dissipation_change"] * df["rf_change"]

        cols = [
            "Relative_time", "Dissipation", "Resonance_Frequency",
            "dissipation_change", "rf_change",
            "diss_x_rf", "change_prod"
        ]
        return df[cols].to_numpy()

# ─── dataset builder ────────────────────────────────────────────────────────────


def build_dataset(
    csv_paths: List[Path],
    window_size: int = WIN_SIZE,
    factor: int = DS_FACTOR
) -> Tuple[np.ndarray, np.ndarray]:
    fe = FeatureExtractor(window_size=window_size)
    fe.fit_scalers(csv_paths, downsample_factor=factor)

    X_list, y_list = [], []
    for csv_path in tqdm(csv_paths):
        # locate the corresponding POI file
        poi_path = csv_path.with_name(f"{csv_path.stem}_poi.csv")
        if not poi_path.exists():
            print("POI file not found, skipping run: %s", poi_path)
            continue

        # attempt to load the POI indices
        try:
            poi_indices = np.loadtxt(poi_path, dtype=int, ndmin=1)
        except Exception as e:
            print(
                "Could not load POI file %s (%s), skipping", poi_path, e)
            continue

        # load & preprocess the run
        df = pd.read_csv(csv_path)
        df_ds = downsample_and_rebaseline(
            df, factor=factor, window_size=window_size)
        windows = make_windows(df_ds, window_size=window_size)
        labels = label_windows(len(windows), poi_indices,
                               factor, window_size, TARGET_POIS)

        # extract features & map labels
        for win, lbl in zip(windows, labels):
            X_list.append(fe.transform_window(win))
            y_list.append(LABEL_MAP[lbl])

    if not X_list:
        raise ValueError(
            "No valid runs found (all POI files missing or unreadable)")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X, y

# ─── split, tf.data pipeline ───────────────────────────────────────────────────


all_csvs = [
    p for p in Path("content/live/train").rglob("*.csv")
    if not p.name.endswith(("_poi.csv", "_lower.csv"))
]
train_paths, val_paths = train_test_split(
    all_csvs, test_size=0.2, random_state=42)

X_train, y_train = build_dataset(train_paths)
# Separate background (class 0) from the POI classes
bg_mask = (y_train == 0)
X_bg,   y_bg = X_train[bg_mask],    y_train[bg_mask]
X_non0, y_non0 = X_train[~bg_mask],   y_train[~bg_mask]

# find how many background windows you have
bg_count = len(y_bg)

# upsample each POI class to match the background count
X_upsampled, y_upsampled = [], []
for cls in np.unique(y_non0):
    Xc = X_non0[y_non0 == cls]
    yc = y_non0[y_non0 == cls]
    Xr, yr = resample(
        Xc, yc,
        replace=True,
        n_samples=bg_count,
        random_state=42
    )
    X_upsampled.append(Xr)
    y_upsampled.append(yr)

# concatenate everything back together
X_pois = np.concatenate(X_upsampled, axis=0)
y_pois = np.concatenate(y_upsampled, axis=0)

X_train_bal = np.concatenate([X_bg, X_pois], axis=0)
y_train_bal = np.concatenate([y_bg, y_pois], axis=0)

# shuffle
perm = np.random.permutation(len(X_train_bal))
X_train, y_train = X_train_bal[perm], y_train_bal[perm]

# print counts to verify
unique, counts = np.unique(y_train, return_counts=True)
print("Post‑balance class counts:")
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls}: {cnt}")

X_val,   y_val = build_dataset(val_paths)

batch_size = 64
train_ds = (
    tf.data.Dataset
      .from_tensor_slices((X_train, y_train))
      .shuffle(len(X_train))
      .batch(batch_size)
      .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset
      .from_tensor_slices((X_val, y_val))
      .batch(batch_size)
      .prefetch(tf.data.AUTOTUNE)
)

# ─── model definition ───────────────────────────────────────────────────────────

num_classes = len(LABEL_MAP)  # 5

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WIN_SIZE, 7)),
    tf.keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ─── training ──────────────────────────────────────────────────────────────────

epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)


# ─── live simulation on a held‑out validation run ────────────────────────────

# inverse mapping back to original POI labels
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
num_classes = len(LABEL_MAP)

# collect all runs in the live/valid directory
live_dir = Path("content/live/valid")
val_paths = sorted(
    p for p in live_dir.rglob("*.csv")
    if not p.name.endswith("_poi.csv")
)

# re‑fit your extractor on TRAINING paths once
fe_live = FeatureExtractor(window_size=WIN_SIZE)
fe_live.fit_scalers(train_paths, downsample_factor=DS_FACTOR)

for sample_path in val_paths:
    print(f"\n=== Live sim on: {sample_path} ===")

    # derive the matching POI file
    poi_path = str(sample_path).replace(".csv", "_poi.csv")
    orig_pois = np.loadtxt(poi_path, delimiter=",", dtype=int, ndmin=1)
    orig_stop_idx = int(orig_pois[-1])

    # load & preprocess the sample run
    df_sample = pd.read_csv(sample_path)
    df_ds = downsample_and_rebaseline(
        df_sample, factor=DS_FACTOR, window_size=WIN_SIZE
    )
    windows = make_windows(df_ds, window_size=WIN_SIZE)

    # set up a fresh live plot for this run
    plt.ion()
    fig, (ax_d, ax_rf, ax_p) = plt.subplots(3, 1, figsize=(8, 10))
    # … (same plotting setup for line_d, line_rf, bar_containers, poi_txt, scatter_preds) …

    buffer_t = []
    buffer_d = []
    buffer_rf = []
    # ─── time‑series lines ────────────────────────────────────
    line_d, = ax_d.plot([], [], linestyle='-', label="Dissipation")
    line_rf, = ax_rf.plot([], [], linestyle='-', label="Resonance Freq")
    for ax in (ax_d, ax_rf):
        ax.set_ylabel("Scaled Value")
        ax.legend()
        ax.grid(True)
    ax_rf.set_xlabel("Relative Time (scaled)")

    # ─── probability bar chart ───────────────────────────────
    bar_containers = ax_p.bar(
        range(num_classes),
        [0] * num_classes,
    )
    ax_p.set_ylim(0, 1)
    ax_p.set_xticks(range(num_classes))
    ax_p.set_xticklabels([INV_LABEL_MAP[i] for i in range(num_classes)])
    ax_p.set_ylabel("Probability")
    ax_p.set_title("Class Probabilities")

    # ─── POI text annotation ─────────────────────────────────
    poi_txt = ax_d.text(
        0.05, 0.9, "", transform=ax_d.transAxes,
        fontsize=14, color='red',
        bbox=dict(facecolor='white', alpha=0.7),
        visible=False
    )

    # ─── scatter for predicted POIs ──────────────────────────
    scatter_preds = ax_d.scatter(
        [], [], marker='X', s=100, color='red', label="Predicted POI"
    )
    ax_d.legend()
    for i, win in enumerate(windows, start=1):
        feats = fe_live.transform_window(win)               # (WIN_SIZE, 3)
        x, y_d, y_rf = feats[:, 0], feats[:, 1], feats[:, 2]

        # predict
        preds = model.predict(feats[np.newaxis, ...], verbose=0)[0]
        cls = np.argmax(preds)
        lbl = INV_LABEL_MAP[cls]

        # append to buffers and update the time‑series lines
        buffer_t.extend(x)
        buffer_d.extend(y_d)
        buffer_rf.extend(y_rf)
        line_d.set_data(buffer_t, buffer_d)
        line_rf.set_data(buffer_t, buffer_rf)
        for ax in (ax_d, ax_rf):
            ax.relim()
            ax.autoscale_view()

        # scatter a POI hit if any
        if lbl != 0:
            pred_time, pred_val = x[-1], y_d[-1]
            old = scatter_preds.get_offsets()
            new = np.vstack([old, [pred_time, pred_val]]) if old.size else np.array(
                [[pred_time, pred_val]])
            scatter_preds.set_offsets(new)

        # update probability bars
        for bar, p in zip(bar_containers, preds):
            bar.set_height(p)

        # title & annotation
        fig.suptitle(
            f"Run {sample_path.name} — Window {i}/{len(windows)} — Predicted POI: {lbl}")
        if lbl != 0:
            poi_txt.set_text(f"POI {lbl}")
            poi_txt.set_visible(True)
        else:
            poi_txt.set_visible(False)

        fig.canvas.draw_idle()
        plt.pause(0.1)

    plt.ioff()
    plt.show()
    plt.close(fig)
