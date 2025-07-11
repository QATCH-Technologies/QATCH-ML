from scipy.ndimage import uniform_filter1d
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.cm as cm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import ruptures as rpt
from scipy.signal import find_peaks
from sklearn.manifold import TSNE
from keras import Model
from sklearn.metrics import precision_recall_fscore_support

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, Model, backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from q_model_data_processor import QDataProcessor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
# ── 1) Data loading & labeling ────────────────────────────────────────────────


def load_data_and_labels(load_dir, max_datasets=100):
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
    epochs=10,
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

# ── 5) Reconstruction‐error anomaly detection ────────────────────────────────

# 5.1) Get reconstructions for the test set
recons = vae.predict(X_test, batch_size=32)     # (N_test, T_pad, 1)

# 5.2) Compute per‐timestep MSE error
errors = np.mean((X_test - recons)**2, axis=2)  # (N_test, T_pad)

# 5.3) For each sample, pick the POI as the time‐index of max error
y_pred = np.zeros_like(y_test)
for i, (err_seq, L) in enumerate(zip(errors, lens_test)):
    err = err_seq[:L]
    # smooth out spikey noise
    err_s = uniform_filter1d(err, size=5)
    # baseline = first 20% of your trace
    baseline = err_s[: int(0.2 * L)]
    thr = baseline.mean() + 3 * baseline.std()  # e.g. 3σ above baseline
    # find first crossing
    crosses = np.where(err_s > thr)[0]
    if crosses.size:
        poi = crosses[0]
    else:
        poi = int(np.argmax(err_s))
    y_pred[i, poi] = 1
# ── 6) Evaluate ───────────────────────────────────────────────────────────────
y_true_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()

prec, rec, f1, _ = precision_recall_fscore_support(
    y_true_flat, y_pred_flat, average='binary'
)
print(
    f"AE Anomaly Detection — Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

# ── 7) Sample plot to visualize an actual prediction ─────────────────────────

# choose a sample index to visualize
sample_idx = 0

# time axis up to the true length
t = np.arange(lens_test[sample_idx])

# the true dissipation curve (unpadded)
curve = X_test[sample_idx, :lens_test[sample_idx], 0]

# locate true and predicted POIs
true_pois = np.where(y_test[sample_idx, :lens_test[sample_idx]] == 1)[0]
pred_pois = np.where(y_pred[sample_idx, :lens_test[sample_idx]] == 1)[0]
print(pred_pois)
# pick first POI if multiple
true_poi = int(true_pois[0]) if true_pois.size else None
pred_poi = int(pred_pois[0]) if pred_pois.size else None

plt.figure()
plt.plot(t, curve, label='Dissipation')
if true_poi is not None:
    plt.scatter([true_poi], [curve[true_poi]], marker='o', label='True POI')
if pred_poi is not None:
    plt.scatter([pred_poi], [curve[pred_poi]],
                marker='x', label='Predicted POI')
plt.xlabel('Time step')
plt.ylabel('Normalized Dissipation')
plt.title(f'Sample POI Detection (sample {sample_idx})')
plt.legend()
plt.tight_layout()
plt.show()
