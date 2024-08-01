import pandas as pd
import numpy as np
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
from scipy.signal import find_peaks

TRAINING = True
TESTING = True

GOOD_TRAIN_DIR = "content/good_runs/train"
GOOD_VALID_DIR = "content/good_runs/validate"

BAD_TRAIN_DIR = "content/bad_runs/train"
BAD_VALID_DIR = "content/bad_runs/validate"

FEATURES = []
TARGET = "Class"
GOOD_LABEL = 0
BAD_LABEL = 1


def calculate_mean(signal):
    return np.mean(signal)


def calculate_standard_deviation(signal):
    return np.std(signal)


def calculate_peak_to_peak(signal):
    return np.ptp(signal)


def compute_timing_jitter(signal, threshold=0.5):
    peaks, _ = find_peaks(signal, height=threshold)
    intervals = np.diff(peaks)
    jitter = np.std(intervals)
    return jitter


def compute_rms_jitter(signal):
    mean_signal = np.mean(signal)
    deviations = signal - mean_signal
    rms_jitter = np.sqrt(np.mean(deviations**2))
    return rms_jitter


def compute_amplitude_jitter(signal):
    jitter = np.std(signal)
    return jitter


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


def extract_features(file_path):
    # Load CSV file
    df = pd.read_csv(file_path)
    df.drop(columns=["Date", "Time", "Ambient", "Temperature"], inplace=True)
    reduction = []
    for column in df.columns:
        # Assuming all columns are features
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(df[[column]])
        pca = PCA(n_components=1)
        principal_component = pca.fit_transform(data_standardized)
        summary_value = principal_component.mean()
        reduction.append(summary_value)
    df_arr = pd.DataFrame([reduction], columns=df.columns)
    return df_arr


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

    resampled_df = pd.DataFrame(X, columns=data.drop(columns=droppable).columns)
    resampled_df[target] = y
    return resampled_df


def load_content(data_dir, label):
    df = pd.DataFrame()
    for root, dirs, files in os.walk(data_dir):
        for file_path in tqdm(files, desc="<<Loading Data>>"):
            if (
                file_path.endswith(".csv")
                and not file_path.endswith("_poi.csv")
                and not file_path.endswith("_lower.csv")
            ):
                reduced_df = extract_features(os.path.join(root, file_path))
                df = pd.concat([df, reduced_df], ignore_index=True)
                df[TARGET] = label
    return df


if __name__ == "__main__":
    if TRAINING:
        # Paths to files and their labels
        good_df = load_content(GOOD_TRAIN_DIR, label=GOOD_LABEL)
        bad_df = load_content(BAD_TRAIN_DIR, label=BAD_LABEL)
        # Merge the dataframes
        merged_df = pd.concat([good_df, bad_df], ignore_index=True)

        # Shuffle the rows
        shuffled_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

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
        input(f_names)
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
