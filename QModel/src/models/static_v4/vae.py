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


# ── 5) PCA & t-SNE (as before) ───────────────────────────────────────────────
z_pca = PCA(n_components=2).fit_transform(z_vec)
perplex = min(30, z_vec.shape[0]-1)
z_tsne = TSNE(n_components=2, perplexity=perplex, random_state=42)\
    .fit_transform(z_vec)

# plot PCA colored by first-POI
plt.figure(figsize=(6, 5))
sc = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=first_poi, cmap='viridis', s=30)
plt.colorbar(sc, label='first POI index')
plt.title('PCA: colored by true first POI')
plt.tight_layout()
plt.show()

# plot t-SNE colored by first-POI
plt.figure(figsize=(6, 5))
sc2 = plt.scatter(z_tsne[:, 0], z_tsne[:, 1],
                  c=first_poi, cmap='viridis', s=30)
plt.colorbar(sc2, label='first POI index')
plt.title(f't-SNE (perplex={perplex})')
plt.tight_layout()
plt.show()


# ── 6) Automatic event detection & remapping to original time ────────────────
# pick the top-k most responsive channels (average amplitude across all test samples)
amp = np.max(z_seq, axis=1) - np.min(z_seq, axis=1)  # shape (N, L)
mean_amp = np.mean(amp, axis=0)                     # shape (L,)
top_k = 3
top_ch = np.argsort(mean_amp)[-top_k:][::-1]
print("Top channels by avg amplitude:", top_ch)

# detect on the single strongest channel:
ch = top_ch[0]
event_times = []
for seq, orig_len, red_len in zip(z_seq, lens_test, red_test):
    # only real region
    s = seq[:red_len, ch]
    # early 20% as “baseline”
    baseline = s[: int(red_len*0.2)].mean()
    peak = s.max()
    thresh = baseline + 0.5*(peak - baseline)        # halfway
    crossings = np.where(s > thresh)[0]
    if crossings.size:
        # reduced index → orig index
        evt_red = crossings[0]
        evt_orig = int(evt_red / red_len * orig_len)
    else:
        evt_orig = np.nan
    event_times.append(evt_orig)
event_times = np.array(event_times)

# plot PCA colored by detected time
plt.figure(figsize=(6, 5))
sc3 = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=event_times, cmap='plasma', s=30)
plt.colorbar(sc3, label='detected event time')
plt.title('PCA: colored by auto-detected POI')
plt.tight_layout()
plt.show()


# ── 7) Visualize heatmap & 1D trace (masked) for sample 0 ──────────────────
sample_idx = 0
Treal = red_test[sample_idx]
sample_z = z_seq[sample_idx, :Treal, :]

plt.figure(figsize=(10, 4))
plt.imshow(sample_z.T, aspect='auto', origin='lower')
plt.colorbar(label='μ activation')
plt.xlabel('Reduced time step')
plt.ylabel('Latent channel')
plt.title(f'Latent activations (sample {sample_idx})')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(np.arange(Treal), sample_z[:, ch])
plt.xlabel('Reduced time step')
plt.ylabel(f'μ (channel {ch})')
plt.title(f'Channel {ch} over time (sample {sample_idx})')
plt.grid(True)
plt.tight_layout()
plt.show()

# ─── 1) Define all four detection routines ───────────────────────────────────


def detect_peak_amp(seq, top_k=3, rel_height=0.5, min_distance=5):
    """Unsupervised amplitude‐peak detection in reduced space."""
    amp = seq.max(axis=0) - seq.min(axis=0)
    top_ch = np.argsort(amp)[-top_k:]
    # <-- use np.abs(...) here
    agg = np.abs(seq[:, top_ch]).sum(axis=1)
    peaks, _ = find_peaks(
        agg,
        height=agg.min() + rel_height*(agg.max() - agg.min()),
        distance=min_distance
    )
    return peaks[0] if peaks.size else np.nan


def detect_slope(seq, top_k=3, slope_factor=3):
    """First‐derivative (slope)‐based detection."""
    amp = seq.max(axis=0) - seq.min(axis=0)
    top_ch = np.argsort(amp)[-top_k:]
    agg = np.abs(seq[:, top_ch]).sum(axis=1)
    deriv = np.diff(agg, prepend=agg[0])
    base_std = deriv[: int(0.2 * len(deriv))].std()
    thresh = slope_factor * base_std
    idx = np.where(deriv > thresh)[0]
    return idx[0] if idx.size else np.nan


def detect_recon_error(vae, x_pad, seq_red_len):
    """
    Reconstruction‐error peak detection on the single output channel.
    - x_pad: 1D array of length T' (the reduced length), i.e. X_test[sample_idx,:,0]
    - seq_red_len: the true reduced length for this sample
    """
    # 1) reshape to (1, T', 1)
    inp = x_pad[np.newaxis, :seq_red_len, None]
    # 2) get reconstruction: shape (1, T', 1)
    recon = vae.predict(inp)[0, :seq_red_len, 0]
    # 3) per‐step MSE
    err = (x_pad[:seq_red_len] - recon)**2
    # 4) find the first big error‐peak
    peaks, _ = find_peaks(err, height=err.mean() + err.std())
    return peaks[0] if peaks.size else np.nan


def detect_changepoint(seq, top_k=3, pen=3):
    """Change‐point detection on aggregated channels via PELT."""
    amp = seq.max(axis=0) - seq.min(axis=0)
    top_ch = np.argsort(amp)[-top_k:]
    agg = np.abs(seq[:, top_ch]).sum(axis=1)
    algo = rpt.Pelt(model="rbf").fit(agg)
    bkpts = algo.predict(pen=pen)
    return bkpts[0] if bkpts else np.nan


# ── Supervised heads ────────────────────────────────────────────────────────
head_epochs = 20
batch_size = 32


def build_seq_head(encoder_model, max_red_len):
    mu = encoder_model.output
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(mu)
    logits = layers.Conv1D(1, 1, padding='same')(x)
    probs = layers.Activation('sigmoid', name='poi_prob')(logits)
    model = Model(encoder_model.input, probs, name='seq_head')
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['Precision', 'Recall'])
    return model


def build_reg_head(encoder_model):
    mu = encoder_model.output
    gap = layers.GlobalAveragePooling1D()(mu)
    rel = layers.Dense(1, activation='sigmoid', name='poi_rel')(gap)
    model = Model(encoder_model.input, rel, name='reg_head')
    model.compile(optimizer='adam', loss='mse')
    return model

# prepare seq labels


def make_fixed_seq_labels(y, orig_lens, red_lens, red_pad):
    N = len(y)
    out = np.zeros((N, red_pad, 1), dtype=float)
    for i, (yi, L, RL) in enumerate(zip(y, orig_lens, red_lens)):
        true_idxs = np.where(yi[:L] == 1)[0]
        if true_idxs.size:
            t_red = int(true_idxs[0] / L * RL)
            out[i, t_red, 0] = 1
    return out


y_train_seq = make_fixed_seq_labels(y_train, lens_train, red_train, red_pad)
y_val_seq = make_fixed_seq_labels(y_val,   lens_val,   red_val,   red_pad)


y_train_rel = np.array([np.where(y[:L] == 1)[
                       0][0]/L if y[:L].any() else 0 for y, L in zip(y_train, lens_train)])
y_val_rel = np.array([np.where(y[:L] == 1)[0][0] /
                     L if y[:L].any() else 0 for y, L in zip(y_val, lens_val)])

seq_model = build_seq_head(encoder, red_train.max())
reg_model = build_reg_head(encoder)

print("y_train_seq shape:", y_train_seq.shape)  # (n_train, red_pad, 1)
print("y_val_seq   shape:", y_val_seq.shape)    # (n_val,   red_pad, 1)

seq_model.fit(
    X_train, y_train_seq,
    validation_data=(X_val, y_val_seq),
    epochs=head_epochs, batch_size=batch_size
)

reg_model.fit(X_train, y_train_rel, validation_data=(
    X_val, y_val_rel), epochs=head_epochs, batch_size=batch_size)

# ── Prediction + Plotting ──────────────────────────────────────────────────


def predict_and_plot(sample_idx=0):
    Xs = X_test[sample_idx]
    L = lens_test[sample_idx]
    RL = red_test[sample_idx]
    y = y_test[sample_idx, :L]
    z = z_seq[sample_idx, :RL, :]
    t = np.arange(RL)
    ch0 = np.argmax(z.ptp(axis=0))

    uns = [
        detect_peak_amp(z),
        detect_slope(z),
        detect_recon_error(vae, Xs.squeeze(), RL),
        detect_changepoint(z)
    ]
    ens = np.nanmedian(uns)

    p_seq = seq_model.predict(Xs[None])[0, :RL, 0]
    p_reg = reg_model.predict(Xs[None])[0, 0]*RL
    pred_seq = np.where(p_seq > 0.5)[0]
    pred_seq = pred_seq[0] if pred_seq.size else np.nan

    true_time = np.where(y == 1)[0][0] if np.any(y == 1) else np.nan
    true = true_time / L * RL if not np.isnan(true_time) else np.nan

    plt.figure(figsize=(10, 4))
    plt.plot(t, z[:, ch0], label=f"latent ch{ch0}")
    plt.axvline(true, color='k', ls='--', label='True POI')
    labels = ['Amp', 'Slope', 'Recon', 'CP']
    for i, u in enumerate(uns):
        plt.axvline(u, color=f'C{i+1}', ls=':', label=labels[i])
    plt.axvline(ens, color='m', ls='--', label='Ensemble')
    plt.axvline(pred_seq, color='c', ls='-.', label='Seq-head')
    plt.axvline(p_reg, color='y', ls='-.', label='Reg-head')

    plt.title(f"Sample {sample_idx}")
    plt.xlabel('Reduced time-step')
    plt.ylabel('μ activation')
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()


for idx in [0, 5, 10]:
    predict_and_plot(idx)
