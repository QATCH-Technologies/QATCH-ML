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

from QConstants import *
from QModel import QModel, QModelPredict


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
    print(f"[INFO] resampling df {target}")
    y = data[target].values
    X = data.drop(columns=droppable)

    over = SMOTE()
    under = RandomUnderSampler()
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
            content.append(os.path.join(root, file))

    train_content, test_content = train_test_split(
        content, test_size=0.95, random_state=42, shuffle=True
    )
    return train_content, test_content


def xgb_pipeline(train_content):
    print(f"[INFO] XGB Preprocessing on {len(train_content)} datasets")
    data_df_S = pd.DataFrame()
    data_df_L = pd.DataFrame()
    for filename in tqdm(train_content, desc="<<Processing XGB>>"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            qdp = QDataPipeline(data_file)
            time_delta = qdp.find_time_delta()
            poi_file = filename.replace(".csv", "_poi.csv")
            actual_indices = pd.read_csv(poi_file, header=None).values
            qdp.preprocess(poi_file=poi_file)
            has_nan = qdp.__dataframe__.isna().any().any()
            if not has_nan:
                if time_delta == -1:
                    data_df_S = pd.concat([data_df_S, qdp.get_dataframe()])
                else:
                    data_df_L = pd.concat([data_df_L, qdp.get_dataframe()])
    rds = []
    rdl = []
    for target in S_TARGETS:
        rds.append(resample_df(data_df_S, target, S_TARGETS))
        rdl.append(resample_df(data_df_L, target, S_TARGETS))
    return rds, rdl


train_content, test_content = load_content(T0_TRAIN_PATH_S)
if TRAINING:
    short_training, long_training = xgb_pipeline(train_content)
    qmodel_1_S = QModel(
        dataset=short_training[0], predictors=FEATURES, target_features=S_TARGETS[0]
    )
    qmodel_2_S = QModel(
        dataset=short_training[1], predictors=FEATURES, target_features=S_TARGETS[1]
    )
    qmodel_3_S = QModel(
        dataset=short_training[2], predictors=FEATURES, target_features=S_TARGETS[2]
    )
    qmodel_4_S = QModel(
        dataset=short_training[3], predictors=FEATURES, target_features=S_TARGETS[3]
    )
    qmodel_5_S = QModel(
        dataset=short_training[4], predictors=FEATURES, target_features=S_TARGETS[4]
    )
    qmodel_6_S = QModel(
        dataset=short_training[5], predictors=FEATURES, target_features=S_TARGETS[5]
    )
    qmodel_1_S.tune(250)
    qmodel_1_S.train_model()
    qmodel_1_S.save_model("QModel_1_T0_S")
    qmodel_2_S.tune(250)
    qmodel_2_S.train_model()
    qmodel_2_S.save_model("QModel_2_T0_S")
    qmodel_3_S.tune(250)
    qmodel_3_S.train_model()
    qmodel_3_S.save_model("QModel_3_T0_S")
    qmodel_4_S.tune(250)
    qmodel_4_S.train_model()
    qmodel_4_S.save_model("QModel_4_T0_S")
    qmodel_5_S.tune(250)
    qmodel_5_S.train_model()
    qmodel_5_S.save_model("QModel_5_T0_S")
    qmodel_6_S.tune(250)
    qmodel_6_S.train_model()
    qmodel_6_S.save_model("QModel_6_T0_S")

    ##########################
    # Long runs
    ##########################
    qmodel_1_L = QModel(
        dataset=long_training[0], predictors=FEATURES, target_features=S_TARGETS[0]
    )
    qmodel_2_L = QModel(
        dataset=long_training[1], predictors=FEATURES, target_features=S_TARGETS[1]
    )
    qmodel_3_L = QModel(
        dataset=long_training[2], predictors=FEATURES, target_features=S_TARGETS[2]
    )
    qmodel_4_L = QModel(
        dataset=long_training[3], predictors=FEATURES, target_features=S_TARGETS[3]
    )
    qmodel_5_L = QModel(
        dataset=long_training[4], predictors=FEATURES, target_features=S_TARGETS[4]
    )
    qmodel_6_L = QModel(
        dataset=long_training[5], predictors=FEATURES, target_features=S_TARGETS[5]
    )
    qmodel_1_L.tune(250)
    qmodel_1_L.train_model()
    qmodel_1_L.save_model("QModel_1_T0_L")
    qmodel_2_L.tune(250)
    qmodel_2_L.train_model()
    qmodel_2_L.save_model("QModel_2_T0_L")
    qmodel_3_L.tune(250)
    qmodel_3_L.train_model()
    qmodel_3_L.save_model("QModel_3_T0_L")
    qmodel_4_L.tune(250)
    qmodel_4_L.train_model()
    qmodel_4_L.save_model("QModel_4_T0_L")
    qmodel_5_L.tune(250)
    qmodel_5_L.train_model()
    qmodel_5_L.save_model("QModel_5_T0_L")
    qmodel_6_L.tune(250)
    qmodel_6_L.train_model()
    qmodel_6_L.save_model("QModel_6_T0_L")

    

data_df = pd.DataFrame()
content = []
qmp_S = QModelPredict(
    "QModel/SavedModels/QModel_1_T0_S.json",
    "QModel/SavedModels/QModel_2_T0_S.json",
    "QModel/SavedModels/QModel_3_T0_S.json",
    "QModel/SavedModels/QModel_4_T0_S.json",
    "QModel/SavedModels/QModel_5_T0_S.json",
    "QModel/SavedModels/QModel_6_T0_S.json",
)
qmp_L = QModelPredict(
    "QModel/SavedModels/QModel_1_T0_L.json",
    "QModel/SavedModels/QModel_2_T0_L.json",
    "QModel/SavedModels/QModel_3_T0_L.json",
    "QModel/SavedModels/QModel_4_T0_L.json",
    "QModel/SavedModels/QModel_5_T0_L.json",
    "QModel/SavedModels/QModel_6_T0_L.json",
)


for root, dirs, files in os.walk(T0_VALID_PATH_S):
    for file in files:
        content.append(os.path.join(root, file))
for filename in content:
    if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
        data_file = filename
        # model_data.run()
        poi_file = filename.replace(".csv", "_poi.csv")
        actual_indices = pd.read_csv(poi_file, header=None).values
        qdp = QDataPipeline(data_file)
        qdp.preprocess(poi_file=None)
        delta = qdp.find_time_delta()

        # qdp.__dataframe__ = qdp.__dataframe__.drop(columns=["Class", "Pooling"])
        if delta == -1:
            print('[INFO] Predicting using Short model')
            predictions = qmp_S.predict(data_file)
        else:
            print('[INFO] Predicting using Long model')
            predictions = qmp_L.predict(data_file)
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
            plot_name = data_file.replace(T0_VALID_PATH_S, "")
            plt.xlabel("POIs")
            plt.ylabel("Dissipation")
            plt.title(f"Predicted/Actual POIs on Data: {plot_name}")

            plt.legend()
            plt.grid(True)
            plt.show()
