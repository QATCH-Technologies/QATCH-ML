import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump, load
from QDataPipline import QDataPipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from QMultiModel import QModel, QPredictor

PLOTTING = True
TRAINING = False
BATCH_SIZE = 0.8
TRAINING_PATH = "content/training_data_with_points"
TEST_PATH = "content/validation_datasets"
FEATURES = [
    "Relative_time",
    "Resonance_Frequency",
    "Dissipation",
    "Difference",
    "Cumulative",
    "Dissipation_super",
    "Difference_super",
    "Cumulative_super",
    "Resonance_Frequency_super",
    "Dissipation_gradient",
    "Difference_gradient",
    "Resonance_Frequency_gradient",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Resonance_Frequency_detrend",
    "Difference_detrend",
]
M_TARGET = "Class"


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def correlation_view(df):
    colormap = plt.cm.RdBu

    plt.figure(figsize=(14, 12))
    plt.title("Pearson Correlation of Features", y=1.05, size=15)
    sns.heatmap(
        df.astype(float).corr(),
        linewidths=0.1,
        vmax=1.0,
        square=True,
        cmap=colormap,
        linecolor="white",
        annot=True,
    )
    plt.show()


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


def pca_view(X, y):
    print("[INFO] building PCA")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolor="k")
    plt.title("PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Target Value")
    plt.show()


def resample_df(data, target, droppable, length):
    print(f"[INFO] resampling {length} target='{target}'")
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


def load_content(data_dir, size=0.5):
    print(f"[INFO] Loading content from {data_dir}")
    content = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if (
                file.endswith(".csv")
                and not file.endswith("_poi.csv")
                and not file.endswith("_lower.csv")
            ):
                content.append(os.path.join(root, file))
    train_content, test_content = train_test_split(
        content, test_size=size, random_state=42, shuffle=True
    )
    return train_content, test_content


def xgb_pipeline(train_content):
    print(f"[INFO] XGB Preprocessing on {len(train_content)} datasets")
    data_df_short = pd.DataFrame()
    data_df_long = pd.DataFrame()
    for filename in tqdm(train_content, desc="<<Processing XGB>>"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            qdp = QDataPipeline(data_file, multi_class=True)
            poi_file = filename.replace(".csv", "_poi.csv")
            time_delta = qdp.find_time_delta()
            qdp.preprocess(poi_file=poi_file)
            has_nan = qdp.__dataframe__.isna().any().any()
            if not has_nan:
                if time_delta == -1:
                    data_df_short = pd.concat([data_df_short, qdp.get_dataframe()])
                else:
                    data_df_long = pd.concat([data_df_long, qdp.get_dataframe()])
    resampled_df_short = resample_df(data_df_short, M_TARGET, M_TARGET, "short")
    resampled_df_long = resample_df(data_df_long, M_TARGET, M_TARGET, "long")
    return resampled_df_short, resampled_df_long


if __name__ == "__main__":
    print("[INFO] QTrainMulti.py script start")
    train_content, test_content = load_content(TRAINING_PATH, size=BATCH_SIZE)
    if TRAINING:
        short_set, long_set = xgb_pipeline(train_content)
        qmodel_short = QModel(
            dataset=short_set, predictors=FEATURES, target_features=M_TARGET
        )
        qmodel_long = QModel(
            dataset=long_set, predictors=FEATURES, target_features=M_TARGET
        )
        qmodel_short.tune()
        qmodel_short.train_model()
        qmodel_short.save_model("QMulti_S")
        qmodel_short.tune()
        qmodel_short.train_model()
        qmodel_short.save_model("QMulti_L")

    data_df = pd.DataFrame()
    content = []
    qmp = QPredictor("QModel/SavedModels/QMultiModel.json")

    for root, dirs, files in os.walk(TEST_PATH):
        for file in files:
            content.append(os.path.join(root, file))
    for filename in content:
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            poi_file = filename.replace(".csv", "_poi.csv")
            actual_indices = pd.read_csv(poi_file, header=None).values
            qdp = QDataPipeline(data_file)
            qdp.preprocess(poi_file=None)
            predictions = qmp.predict(data_file)
            if PLOTTING:
                palette = sns.color_palette("husl", 6)
                df = pd.read_csv(data_file)
                dissipation = normalize(df["Dissipation"])
                print("Actual, Predicted")
                for actual, predicted in zip(actual_indices, predictions):
                    print(f"{actual[0]}, {predicted}")
                plt.figure()

                plt.plot(
                    dissipation,
                    color="grey",
                    label="Dissipation",
                )
                for i, index in enumerate(predictions):
                    plt.axvline(
                        x=index, color=palette[i], label=f"Predicted POI {i + 1}"
                    )
                for i, index in enumerate(actual_indices):
                    plt.axvline(
                        x=index,
                        color=palette[i],
                        linestyle="dashed",
                        label=f"Actual POI {i + 1}",
                    )
                plot_name = data_file.replace(TRAINING_PATH, "")
                plt.xlabel("POIs")
                plt.ylabel("Dissipation")
                plt.title(f"Predicted/Actual POIs on Data: {plot_name}")

                plt.legend()
                plt.grid(True)
                plt.show()
