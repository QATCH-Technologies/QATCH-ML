from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.cm as cm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import ruptures as rpt
from scipy.signal import find_peaks
from keras.layers import GlobalAveragePooling1D, Dense, Conv1D, Activation
from keras.layers import GlobalAveragePooling1D, Dense
from sklearn.manifold import TSNE
from keras import Model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model, backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from q_model_data_processor import QDataProcessor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
# ── 1) Data loading & labeling ────────────────────────────────────────────────


def load_data_and_labels(load_dir, max_datasets=1000):
    content = QDataProcessor.load_content(load_dir, num_datasets=max_datasets)
    X_list, y_list = [], []
    for data_path, poi_path in content:
        df = pd.read_csv(data_path)
        X = df[['Dissipation']].values.astype(np.float32)
        X = MinMaxScaler((0, 1)).fit_transform(X)

        poi_idxs = np.loadtxt(poi_path, int)
        T = X.shape[0]
        y = np.zeros((T,), np.int32)
        poi_idxs = poi_idxs[(poi_idxs >= 0) & (poi_idxs < T)]
        y[poi_idxs] = 1

        X_list.append(X)
        y_list.append(y)
    return X_list, y_list


LOAD_DIRECTORY = os.path.join("content", "dropbox_dump")
X_list, y_list = load_data_and_labels(LOAD_DIRECTORY)

# ── 2) Padding & split ────────────────────────────────────────────────────────

X_pad = pad_sequences(X_list, padding='post', dtype='float32')
y_pad = pad_sequences(y_list, padding='post', dtype='int32')
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_pad, y_pad, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42)


# ── 3) Build Conv1D-VAE ────────────────────────────────────────────────────────


def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=K.shape(mu))
    return mu + K.exp(0.5 * log_var) * eps


def build_fc_conv1d_vae(channels=1, latent_channels=16):
    inp = layers.Input(shape=(None, channels), name='enc_in')

    # mask away our pad-value (-1)
    x = layers.Masking(mask_value=-1.0)(inp)

    # Encoder
    x = layers.Conv1D(16, 5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1D(32, 5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1D(64, 5, strides=2, padding='same', activation='relu')(x)

    # Bottleneck
    mu = layers.Conv1D(latent_channels, 1, padding='same', name='mu')(x)
    log_var = layers.Conv1D(
        latent_channels, 1, padding='same', name='log_var')(x)
    z = layers.Lambda(sampling, name='z')([mu, log_var])

    # Decoder
    x = layers.Conv1DTranspose(
        64, 5, strides=2, padding='same', activation='relu')(z)
    x = layers.Conv1DTranspose(
        32, 5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1DTranspose(
        16, 5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1D(channels, 1, padding='same', activation='sigmoid')(x)

    # crop back to input length
    outputs = layers.Lambda(
        lambda tensors: tensors[0][:, : tf.shape(tensors[1])[1], :],
        name='dynamic_crop'
    )([x, inp])

    vae = Model(inp, outputs, name='fc_conv1d_vae')
    recon = tf.reduce_mean(tf.keras.losses.mse(inp, outputs))
    kl = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    vae.add_loss(recon + kl)
    vae.compile(optimizer='adam')
    return vae


# ── after load_data_and_labels, before padding ────────────────────────────────
# record the original (unpadded) lengths
seq_lengths = np.array([len(x) for x in X_list])
# helper: compute how many time-steps remain after 3 stride-2 convs (ceil-div each time)


def compute_reduced_lengths(lengths, strides=3):
    l = lengths.astype(float)
    for _ in range(strides):
        l = np.ceil(l / 2)
    return l.astype(int)


pad_len = X_pad.shape[1]
red_pad = compute_reduced_lengths(np.array([pad_len]))[0]
print("Fixed reduced length:", red_pad)  # should match model output dim


# ── 2) Padding & split (now including lengths) ───────────────────────────────
X_pad = pad_sequences(X_list, padding='post', dtype='float32')
y_pad = pad_sequences(y_list, padding='post', dtype='int32')

# split *and* carry along the true lengths
X_train, X_tmp, y_train, y_tmp, lens_train, lens_tmp = train_test_split(
    X_pad, y_pad, seq_lengths, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test, lens_val, lens_test = train_test_split(
    X_tmp, y_tmp, lens_tmp, test_size=0.5, random_state=42
)


# reduced lengths for each split
red_train = compute_reduced_lengths(lens_train)
red_val = compute_reduced_lengths(lens_val)
red_test = compute_reduced_lengths(lens_test)

vae = build_fc_conv1d_vae(channels=1, latent_channels=16)
vae.summary()

vae.fit(
    X_train, None,
    validation_data=(X_val, None),
    epochs=50,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True)]
)

# ── 4) Build encoder & get latents ───────────────────────────────────────────
encoder = Model(
    inputs=vae.get_layer('enc_in').input,
    outputs=vae.get_layer('mu').output,
    name='encoder'
)
z_seq = encoder.predict(X_test, batch_size=32)   # (N, T', L)
z_vec = np.mean(z_seq, axis=1)                   # (N, L)

# compute “first POI” for coloring
first_poi = [
    np.where(y[:ℓ] == 1)[0][0] if np.any(y[:ℓ] == 1) else np.nan
    for y, ℓ in zip(y_test, lens_test)
]
first_poi = np.array(first_poi)
# ── 4) Build encoder & get latents ───────────────────────────────────────────
encoder = Model(
    inputs=vae.get_layer('enc_in').input,
    outputs=vae.get_layer('mu').output,
    name='encoder'
)

# Predict all latents at once
z_train_seq = encoder.predict(X_train, batch_size=32)  # (N_train, T', L)
z_test_seq = encoder.predict(X_test,  batch_size=32)  # (N_test,  T', L)

# Helper: build rich features from z_seq


def build_features(z_seq):
    # z_seq: (N, T', L)
    mean_feat = np.mean(z_seq, axis=1)          # (N, L)
    max_feat = np.max(z_seq,  axis=1)          # (N, L)
    var_feat = np.var(z_seq,  axis=1)          # (N, L)
    # time-of-max index for each channel, normalized to [0,1]
    idx_max = np.argmax(z_seq, axis=1)        # (N, L)
    tmax_frac = idx_max / z_seq.shape[1]        # (N, L)
    # concatenate into one feature vector per sample
    return np.concatenate([mean_feat, max_feat, var_feat, tmax_frac], axis=1)


# Build feature matrices
Xf_train = build_features(z_train_seq)
Xf_test = build_features(z_test_seq)

# Compute fractional-POI targets in [0,1]
poi_train = np.array([
    np.where(y[:L] == 1)[0][0] if np.any(y[:L] == 1) else np.nan
    for y, L in zip(y_train, lens_train)
], dtype=float)
frac_train = poi_train / lens_train

# Fit regressor
reg = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    random_state=0
)
reg.fit(Xf_train[~np.isnan(frac_train)], frac_train[~np.isnan(frac_train)])

# Predict fractional POI → convert back to time-step
frac_pred = reg.predict(Xf_test)
rf_preds = (frac_pred * lens_test).astype(int)

# ── 5) (Optional) Plot one example trace ─────────────────────────────────────


def plot_poi_trace(sample_idx):
    L = lens_test[sample_idx]
    series = X_test[sample_idx, :L, 0]
    actual = first_poi[sample_idx]
    pred = rf_preds[sample_idx]

    plt.figure()
    plt.plot(series)
    plt.axvline(actual, label='Actual POI', linestyle='-')
    plt.axvline(pred,   label='Predicted POI', linestyle='--')
    plt.title(f"Sample {sample_idx}: Actual vs Predicted POI")
    plt.xlabel("Time step")
    plt.ylabel("Normalized Dissipation")
    plt.legend()
    plt.show()

# ── 6) Scatter & Error‐dist plots ────────────────────────────────────────────
# (reuse the plotting code we already wrote)


# e.g. inspect the first 3 samples
for i in range(3):
    plot_poi_trace(i)


# 2) Scatter plot of actual vs. predicted across all samples
plt.figure()
plt.scatter(first_poi, rf_preds)
# add y=x line for reference
min_v = np.nanmin(np.concatenate([first_poi, rf_preds]))
max_v = np.nanmax(np.concatenate([first_poi, rf_preds]))
plt.plot([min_v, max_v], [min_v, max_v], linestyle='--', label='Ideal')
plt.title("Actual vs. Predicted POI (all samples)")
plt.xlabel("Actual POI (time step)")
plt.ylabel("Predicted POI (time step)")
plt.legend()
plt.show()
# drop any nan pairs
mask = ~np.isnan(first_poi) & ~np.isnan(rf_preds)
y_true = first_poi[mask]
y_pred = rf_preds[mask]

print("MAE: ", mean_absolute_error(y_true, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
print("R^2:  ", r2_score(y_true, y_pred))
errors = y_pred - y_true

plt.figure()
plt.hist(errors, bins=30, edgecolor='k')
plt.axvline(0, color='k', linestyle='--')
plt.title("Prediction Error Distribution")
plt.xlabel("Predicted POI − Actual POI (time steps)")
plt.ylabel("Count")
plt.show()

lengths = lens_test[mask]
norm = plt.Normalize(lengths.min(), lengths.max())
cmap = cm.viridis

plt.figure()
sc = plt.scatter(y_true, y_pred, c=lengths, cmap=cmap, norm=norm)
plt.plot([min_v, max_v], [min_v, max_v], '--', color='gray')
plt.colorbar(sc, label="Original seq length")
plt.title("Actual vs Predicted POI (colored by length)")
plt.xlabel("Actual POI")
plt.ylabel("Predicted POI")
plt.show()
# compute abs errors and sort
abs_err = np.abs(errors)
idx_sort = np.argsort(abs_err)

# 3 best
for i in idx_sort[:3]:
    plot_poi_trace(i)

# 3 worst
for i in idx_sort[-3:]:
    plot_poi_trace(i)
means = 0.5 * (y_true + y_pred)
diffs = y_pred - y_true

plt.figure()
plt.scatter(means, diffs, alpha=0.6)
plt.axhline(0, linestyle='--', color='k')
plt.title("Bland–Altman: Prediction Error vs Mean POI")
plt.xlabel("Mean of Actual & Predicted POI")
plt.ylabel("Predicted − Actual POI")
plt.show()
