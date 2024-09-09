import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump, load
from q_data_pipeline import QDataPipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import joblib
from QConstants import *
from QMultiModel import QMultiModel, QPredictor


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


def load_content(data_dir):
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
    return content


def xgb_pipeline(train_content):
    print(f"[INFO] XGB Preprocessing on {len(train_content)} datasets")
    data_df = pd.DataFrame()
    for filename in tqdm(train_content, desc="<<Processing XGB>>"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            qdp = QDataPipeline(data_file, multi_class=True)
            poi_file = filename.replace(".csv", "_poi.csv")
            if not os.path.exists(poi_file):
                continue
            qdp.preprocess(poi_filepath=poi_file)
            has_nan = qdp.__dataframe__.isna().any().any()
            if not has_nan:
                data_df = pd.concat([data_df, qdp.get_dataframe()])

    resampled_df = resample_df(data_df, M_TARGET, M_TARGET)
    return resampled_df


if __name__ == "__main__":
    print("[INFO] QTrainMulti.py script start")
    vals = [0, 1, 2]
    for t in vals:

        model_name = f"QMultiType_{t}"
        print(f"[INFO] Training {model_name}")
        TRAIN_PATH = f"content/label_{t}/train"

        if TRAINING:
            train_content = load_content(TRAIN_PATH)
            training_set = xgb_pipeline(train_content)
            print("[INFO] Building multi-target model")
            qmodel_short = QMultiModel(
                dataset=training_set, predictors=FEATURES, target_features=M_TARGET
            )
            qmodel_short.tune()
            qmodel_short.train_model()
            qmodel_short.save_model(model_name)
        if TESTING:
            cluster_model = joblib.load("QModel/SavedModels/cluster.joblib")
            data_df = pd.DataFrame()
            content = []
            qmp = QPredictor(f"QModel/SavedModels/{model_name}.json")
            TEST_PATH = f"content/label_{t}/test"
            for root, dirs, files in os.walk(TEST_PATH):
                for file in files:
                    content.append(os.path.join(root, file))

            for filename in tqdm(content, desc="<<Testing on files>>"):
                if (
                    filename.endswith(".csv")
                    and not filename.endswith("_poi.csv")
                    and not filename.endswith("_lower.csv")
                ):
                    data_file = filename
                    poi_file = filename.replace(".csv", "_poi.csv")
                    actual_indices = pd.read_csv(poi_file, header=None).values
                    qdp = QDataPipeline(data_file)
                    time_delta = qdp.find_time_delta()

                    qdp.preprocess(poi_filepath=None)
                    predictions = None
                    print("[INFO] Predicting using multi-target model")
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
                                x=index,
                                color=palette[i],
                                label=f"Predicted POI {i + 1}",
                            )
                        for i, index in enumerate(actual_indices):
                            plt.axvline(
                                x=index,
                                color=palette[i],
                                linestyle="dashed",
                                label=f"Actual POI {i + 1}",
                            )
                        plot_name = data_file.replace(TRAIN_PATH, "")
                        plt.xlabel("POIs")
                        plt.ylabel("Dissipation")
                        plt.title(f"Predicted/Actual POIs on Data: {plot_name}")

                        plt.legend()
                        plt.grid(True)
                        plt.show()
