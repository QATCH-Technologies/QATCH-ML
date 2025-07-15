
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau, ModelCheckpoint)
from keras.optimizers import Adam
from keras.layers import Dense, Lambda, BatchNormalization, Dropout
from keras import Input, Model
import matplotlib.pyplot as plt
from keras.losses import Huber
from keras import layers, models, callbacks
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import random
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from rdp import rdp
from tqdm import tqdm
import math


class DissipationFeaturePipeline:

    def __init__(self, exact_n: int = 10):
        self.exact_n = exact_n

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
                poi_path = data_path.replace(".csv", "_poi.csv")
                if os.path.exists(poi_path):
                    loaded.append((data_path, poi_path))

        random.shuffle(loaded)
        if num_datasets == np.inf:
            return loaded
        return loaded[:int(num_datasets)]

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

    def _extract_features(self, data_path: str) -> pd.Series:
        """
        Read `data_path` (expects columns 'Relative_time' & 'Dissipation'),
        normalize each to [0,1], summarize with RDP→exact_n points,
        and return a Series of length 2*self.exact_n.
        """
        df = pd.read_csv(data_path)
        t = df['Relative_time'].to_numpy()
        d = df['Dissipation'].to_numpy()
        t0, tN = t.min(), t.max()
        if tN == t0:
            t_norm = np.zeros_like(t)
        else:
            t_norm = (t - t0) / (tN - t0)

        dmin, dmax = d.min(), d.max()
        if dmax == dmin:
            d_norm = np.zeros_like(d)
        else:
            d_norm = (d - dmin) / (dmax - dmin)
        t_key, d_key = self._general_shape_rdp_exact(t_norm, d_norm)
        feat = np.concatenate([t_key, d_key])
        idx = [f"time_{i+1}" for i in range(self.exact_n)] + \
            [f"diss_{i+1}" for i in range(self.exact_n)]
        return pd.Series(feat, index=idx, name=Path(data_path).stem)

    def _extract_targets(self, data_path: str, poi_path: str) -> pd.Series:
        """
        Read your POI CSV (one row of absolute times),
        convert to fractions of the run length ([0,1]),
        and return a Series of length k (k = number of poi_i in the file).
        """
        poi_sec = pd.read_csv(poi_path, header=None).to_numpy().flatten()
        df = pd.read_csv(data_path)
        t0, tN = df['Relative_time'].min(), df['Relative_time'].max()
        if tN == t0:
            poi_rel = np.zeros_like(poi_sec)
        else:
            poi_rel = (poi_sec - t0) / (tN - t0)

        # 4) build index
        idx = [f"poi_{i+1}" for i in range(len(poi_rel))]
        return pd.Series(poi_rel, index=idx, name=Path(poi_path).stem)

    def build_feature_df(self,
                         data_dir: str,
                         num_datasets: int = np.inf) -> pd.DataFrame:
        """
        1) Load up to `num_datasets` (data, poi) tuples,
        2) Skip any whose Relative_time span < min_duration,
        3) Extract features from each data CSV,
        4) Extract POI targets from each poi CSV,
        5) Return a DataFrame with feature columns + poi_1…poi_k columns.
        """
        files = self.load_content(data_dir, num_datasets)
        if not files:
            return pd.DataFrame()

        feats, targets = [], []
        for data_path, poi_path in tqdm(files, desc="Generating Features"):

            feats.append(self._extract_features(data_path))
            targets.append(self._extract_targets(data_path, poi_path))

        if not feats:
            return pd.DataFrame()

        feature_df = pd.DataFrame(feats)
        target_df = pd.DataFrame(targets, index=feature_df.index)
        return pd.concat([feature_df, target_df], axis=1)


# ── Build the DataFrame ───────────────────────────────────────────────────────
pipeline = DissipationFeaturePipeline(exact_n=20)
df = pipeline.build_feature_df("content/static/train", num_datasets=200)

feature_cols = [c for c in df.columns if c.startswith(
    "time_") or c.startswith("diss_")]
target_cols = [c for c in df.columns if c.startswith("poi_")]

X = df[feature_cols] .values.astype(np.float32)
# now in [0,1] if you used relative POIs
y = df[target_cols].values.astype(np.float32)

# ── Train/test split & scaling ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

x_scaler = MinMaxScaler()   # [0,1] input features
y_scaler = MinMaxScaler()   # [0,1] targets (harmless if already in [0,1])

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

n_features = X_train_scaled.shape[1]
n_pois = y_train_scaled.shape[1]

# ── Model builder ────────────────────────────────────────────────────────────


def build_monotonic_poi_model(n_features: int, n_pois: int) -> Model:
    inp = Input(shape=(n_features,), name="RDP_features")
    x = Dense(256, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    deltas = Dense(n_pois, activation="softplus", name="deltas")(x)
    pois = Lambda(lambda z: tf.math.cumsum(z, axis=1), name="pois")(deltas)

    return Model(inputs=inp, outputs=pois, name="monotonic_poi_model")


model = build_monotonic_poi_model(n_features, n_pois)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=Huber(delta=1.0),
    metrics=["mae"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ModelCheckpoint("best_monotonic_poi.h5",
                    save_best_only=True, monitor="val_loss", verbose=1),
]

# ── Train ────────────────────────────────────────────────────────────────────
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.2,
    epochs=300, batch_size=32,
    callbacks=callbacks, verbose=2,
)

# ── Plot loss curves ─────────────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Huber loss")
plt.legend()
plt.tight_layout()
plt.show()

# ── Evaluate on test set ────────────────────────────────────────────────────
model.load_weights("best_monotonic_poi.h5")

y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

test_mse = mean_squared_error(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)
print(f"Test MSE: {test_mse:.3f}, Test MAE: {test_mae:.3f}\n")

for i, col in enumerate(target_cols):
    mse = mean_squared_error(y_test[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    print(f"{col:>6} → MSE {mse:.1f}, MAE {mae:.1f}")

# ── Per‑POI scatter plots (dynamic grid) ─────────────────────────────────────
n = len(target_cols)
ncols = min(n, 3)
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(
    4*ncols, 4*nrows), squeeze=False)
axes = axes.flatten()

for i, col in enumerate(target_cols):
    ax = axes[i]
    ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
    mn, mx = min(y_test[:, i].min(), y_pred[:, i].min()), max(
        y_test[:, i].max(), y_pred[:, i].max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2)
    ax.set_xlabel(f"True {col}")
    ax.set_ylabel(f"Pred {col}")
    ax.set_title(col)

# remove any unused subplots
for j in range(len(target_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ── Validate on new curves ───────────────────────────────────────────────────
examples = pipeline.load_content("content/static/valid/")

for data_path, poi_path in examples:
    raw_df = pd.read_csv(data_path)
    t_new = raw_df["Relative_time"].to_numpy()
    d_new = raw_df["Dissipation"].to_numpy()

    # extract & scale
    feat = pipeline._extract_features(
        data_path).values.reshape(1, -1).astype(np.float32)
    feat_scl = x_scaler.transform(feat)
    y_rel_pred_scaled = model.predict(feat_scl)
    y_rel_pred = y_scaler.inverse_transform(y_rel_pred_scaled).flatten()

    # convert relative [0,1] → sample index
    idx_pred = np.round(y_rel_pred * (len(t_new)-1)).astype(int)
    idx_pred = np.clip(idx_pred, 0, len(t_new)-1)

    t_poi = t_new[idx_pred]
    d_poi = d_new[idx_pred]

    # plot
    plt.figure(figsize=(8, 4))
    plt.plot(t_new, d_new, label="Dissipation curve", lw=1.5)
    plt.scatter(t_poi, d_poi, c="red", marker="x",
                s=100, label="Predicted POIs")
    for j, (tt, dd) in enumerate(zip(t_poi, d_poi), start=1):
        plt.annotate(f"POI{j}", (tt, dd),
                     textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Relative_time")
    plt.ylabel("Dissipation")
    plt.title(f"Sample: {Path(data_path).stem}")
    plt.legend()
    plt.tight_layout()
    plt.show()
