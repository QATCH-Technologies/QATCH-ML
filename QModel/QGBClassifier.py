import pandas as pd
import cupy as cp
from cupyx.scipy.signal import find_peaks
from cupyx.scipy.stats import entropy
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import entropy
from numpy.lib.stride_tricks import sliding_window_view
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from QModel import QModel
from sklearn.cluster import KMeans

import pywt
import seaborn as sns
import random

MEMPOOL = cp.get_default_memory_pool()
P_MEMPOOL = cp.get_default_pinned_memory_pool()

TRAINING = True
TESTING = True

GOOD_TRAIN_DIR = "content/good_runs/train"
GOOD_VALID_DIR = "content/good_runs/validate"

BAD_TRAIN_DIR = "content/bad_runs/train"
BAD_VALID_DIR = "content/bad_runs/validate"

FEATURES = [
    "Approx_Entropy",
    "Autocorrelation",
    "First_Derivative_Mean",
    "Max_Amp",
    "Max_Time",
    "Mean_Absolute_Deviation",
    "Min_Amp",
    "N_Peaks",
    "PTP_Jitter",
    "RMS_Jitter",
    "Second_Derivative_Mean",
    "Shannon_Entropy",
    "Signal_Energy",
    "Variance",
    "Wavelet_Energy",
    # "Zero_Crossing_Rate",
]
TARGET = "Class"
GOOD_LABEL = 0
BAD_LABEL = 1
BATCH_SIZE = np.inf


class SignalMetrics:
    def __init__(self, signal):
        self.__signal__ = self.normalize_data(cp.asarray(signal, dtype=cp.float16))

    def rms_jitter(self):
        jitter = cp.sqrt(cp.mean(cp.diff(self.__signal__) ** 2))
        return jitter

    def peak_to_peak_jitter(self):
        return cp.ptp(cp.diff(self.__signal__))

    def number_of_peaks(self):
        peaks, _ = find_peaks(cp.asnumpy(self.__signal__))
        return len(peaks)

    def signal_energy(self):
        return cp.sum(self.__signal__**2)

    def variance(self):
        return cp.var(self.__signal__)

    def mean_absolute_deviation(self):
        mean_signal = cp.mean(self.__signal__)
        return cp.mean(cp.abs(self.__signal__ - mean_signal))

    def zero_crossing_rate(self):
        zero_crossings = cp.where(cp.diff(cp.signbit(self.__signal__)))[0]
        return len(zero_crossings) / len(self.__signal__)

    def shannon_entropy(self):
        histogram, bin_edges = cp.histogram(self.__signal__, bins=10, density=True)
        return entropy(cp.asnumpy(histogram))

    def approximate_entropy(self, m=2, r=0.2):
        N = len(self.__signal__)
        signal_std = cp.std(self.__signal__)
        r *= signal_std
        X = sliding_window_view(cp.asnumpy(self.__signal__), m)
        X = cp.asarray(X)
        C = (cp.abs(X - X[:, None]) <= r).sum(axis=2).mean(axis=1)
        return -cp.log(C).mean()

    def autocorrelation(self, lag=1):
        n = len(self.__signal__)
        signal_mean = cp.mean(self.__signal__)
        c0 = cp.sum((self.__signal__ - signal_mean) ** 2) / n
        c_lag = (
            cp.sum(
                (self.__signal__[: n - lag] - signal_mean)
                * (self.__signal__[lag:] - signal_mean)
            )
            / n
        )
        return c_lag / c0

    def first_derivative_mean(self):
        deriviative = cp.diff(self.__signal__)
        mean = cp.mean(deriviative)
        return mean

    def second_derivative_mean(self):
        return cp.mean(cp.diff(self.__signal__, n=2))

    def wavelet_energy(self, wavelet="db1"):
        coeffs = pywt.wavedec(cp.asnumpy(self.__signal__), wavelet)
        return sum(cp.sum(c**2) for c in coeffs)

    def normalize_data(self, data):
        return (data - cp.min(data)) / (cp.max(data) - cp.min(data))


def extract_features(file_path):
    # Load CSV file
    df = pd.read_csv(file_path)
    df.drop(
        columns=[
            "Date",
            "Time",
            "Ambient",
            "Temperature",
            "Peak Magnitude (RAW)",
        ],
        inplace=True,
    )

    features = []
    dissipation = df[
        "Resonance_Frequency"
    ].values  # Convert to NumPy array for compatibility
    sm = SignalMetrics(dissipation)
    features.append(float(cp.asnumpy(sm.approximate_entropy())))
    features.append(float(cp.asnumpy(sm.autocorrelation())))
    features.append(float(cp.asnumpy(sm.first_derivative_mean())))
    features.append(dissipation.max())
    features.append(df["Relative_time"].max())
    features.append(float(cp.asnumpy(sm.mean_absolute_deviation())))
    features.append(dissipation.min())
    features.append(sm.number_of_peaks())
    features.append(float(cp.asnumpy(sm.peak_to_peak_jitter())))
    features.append(float(cp.asnumpy(sm.rms_jitter())))
    features.append(float(cp.asnumpy(sm.second_derivative_mean())))
    features.append(float(cp.asnumpy(sm.shannon_entropy())))
    features.append(float(cp.asnumpy(sm.signal_energy())))
    features.append(float(cp.asnumpy(sm.variance())))
    features.append(float(cp.asnumpy(sm.wavelet_energy())))
    features_df = pd.DataFrame([features], columns=FEATURES)
    # Cleanup
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return features_df


def tsne_view(X, y):
    print("[INFO] building t-SNE")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    X_tsne = tsne.fit_transform(X)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="viridis", edgecolor="k"
    )
    plt.title("t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(scatter, label="Target Value")
    plt.show()


def resample_df(data, target, droppable):
    print(f"[INFO] resampling target='{target}'")
    y = data[target].values
    X = data.drop(columns=droppable)
    over = SMOTE(sampling_strategy="auto")
    under = RandomUnderSampler(sampling_strategy="auto")
    steps = [
        ("o", over),
        ("u", under),
    ]
    pipeline = Pipeline(steps=steps)
    X, y = pipeline.fit_resample(X, y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    resampled_df = pd.DataFrame(X, columns=data.drop(columns=droppable).columns)
    resampled_df[target] = y
    return resampled_df


def load_content(data_dir, label):
    df = pd.DataFrame()
    total_files = min(
        sum([len(files) for _, _, files in os.walk(data_dir)]), BATCH_SIZE
    )
    count = 0
    with tqdm(total=total_files, desc="<<Processing Files>>") as pbar:
        for root, dirs, files in os.walk(data_dir):
            random.shuffle(files)
            for file_path in files:
                if count > BATCH_SIZE:
                    break
                if (
                    file_path.endswith(".csv")
                    and not file_path.endswith("_poi.csv")
                    and not file_path.endswith("_lower.csv")
                ):
                    extraction = extract_features(os.path.join(root, file_path))
                    df = pd.concat([df, extraction], ignore_index=True)
                    df[TARGET] = label
                    count += 1
                    pbar.update(1)
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
    return df


def get_representative_samples(labels, dataframes, n_clusters):
    representative_samples = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) > 0:
            representative_sample_index = cluster_indices[0]
            representative_samples.append(dataframes[representative_sample_index])
    return representative_samples


kmeans = None
labels = None
if __name__ == "__main__":
    if TRAINING:
        # Paths to files and their labels
        good_df = load_content(GOOD_TRAIN_DIR, label=GOOD_LABEL)
        bad_df = load_content(BAD_TRAIN_DIR, label=BAD_LABEL)
        # Merge the dataframes
        merged_df = pd.concat([good_df, bad_df], ignore_index=True)

        # Shuffle the rows
        shuffled_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
        corr_matrix = shuffled_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()
        dataset = resample_df(shuffled_df, TARGET, TARGET)
        tsne_view(dataset[FEATURES], dataset[TARGET])
        qm = QModel(dataset=dataset, predictors=FEATURES, target_features=TARGET)
        qm.tune()
        qm.train_model()
        qm.save_model("QGBClassifier")

    if TESTING:
        qmp = xgb.Booster()
        qmp.load_model("QModel/SavedModels/QGBClassifier.json")
        f_names = qmp.feature_names
        for root, dirs, files in os.walk(GOOD_VALID_DIR):
            for file_path in files:
                if (
                    file_path.endswith(".csv")
                    and not file_path.endswith("_poi.csv")
                    and not file_path.endswith("_lower.csv")
                ):
                    file_path = os.path.join(root, file_path)

                    df = extract_features(file_path)
                    d_data = xgb.DMatrix(df)
                    prediction = qmp.predict(d_data)
                    print(prediction)
