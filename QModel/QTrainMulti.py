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
TRAINING = True
PATH = "content/training_data_with_points"
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
    data_df = pd.DataFrame()

    for filename in tqdm(train_content, desc="<<Processing XGB>>"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            qdp = QDataPipeline(data_file, multi_class=True)
            time_delta = qdp.find_time_delta()
            if time_delta == -1:
                poi_file = filename.replace(".csv", "_poi.csv")
                qdp.preprocess(poi_file=poi_file)
                has_nan = qdp.__dataframe__.isna().any().any()
                if not has_nan:
                    data_df = pd.concat([data_df, qdp.get_dataframe()])
    resampled_df = resample_df(data_df, M_TARGET, M_TARGET)
    return resampled_df


train_content, test_content = load_content(PATH, size=0.8)
if TRAINING:
    training_set = xgb_pipeline(train_content)
    qmodel = QModel(dataset=training_set, predictors=FEATURES, target_features=M_TARGET)
    qmodel.tune()
    qmodel.train_model()
    qmodel.save_model()


PATH = "content/validation_datasets"
data_df = pd.DataFrame()
content = []
qmp = QPredictor("QModel/SavedModels/QMultiModel.json")

for root, dirs, files in os.walk(PATH):
    for file in files:
        content.append(os.path.join(root, file))
for filename in content:
    if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
        data_file = filename
        poi_file = filename.replace(".csv", "_poi.csv")
        actual_indices = pd.read_csv(poi_file, header=None).values
        qdp = QDataPipeline(data_file)
        qdp.preprocess(poi_file=None)
        # qdp.__dataframe__ = qdp.__dataframe__.drop(columns=["Class", "Pooling"])

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
            plt.axvline(
                x=predictions[0],
                color=palette[0],
                label="Predicted POI 0",
            )
            for i, index in enumerate(predictions):
                plt.axvline(x=index, color=palette[i], label=f"Predicted POI {i}")
            plt.axvline(
                x=actual_indices[0],
                color=palette[0],
                linestyle="dashed",
                label="Actual POI 0",
            )
            for i, index in enumerate(actual_indices):
                plt.axvline(
                    x=index,
                    color=palette[i],
                    linestyle="dashed",
                    label=f"Actual POI {i}",
                )
            plot_name = data_file.replace(PATH, "")
            plt.xlabel("POIs")
            plt.ylabel("Dissipation")
            plt.title(f"Predicted/Actual POIs on Data: {plot_name}")

            plt.legend()
            plt.grid(True)
            plt.show()
