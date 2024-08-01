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

from QModel import QModel, QModelPredict

PLOTTING = True
XGB = False
STACK = False
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
XGB_TARGETS = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6"]
META_TARGETS = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6"]
S_TARGETS = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6"]
M_TARGET = "Class"
META_FEATURE = [
    "EMP_1",
    "EMP_2",
    "EMP_3",
    "EMP_4",
    "EMP_5",
    "EMP_6",
    "XGB_1",
    "XGB_2",
    "XGB_3",
    "XGB_4",
    "XGB_5",
    "XGB_6",
]


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
    data_df = pd.DataFrame()
    for filename in tqdm(train_content, desc="<<Processing XGB>>"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            qdp = QDataPipeline(data_file)
            time_delta = qdp.find_time_delta()
            if time_delta == -1:
                poi_file = filename.replace(".csv", "_poi.csv")
                actual_indices = pd.read_csv(poi_file, header=None).values

                qdp.preprocess(poi_file=poi_file)
                df = qdp.__dataframe__.drop(columns=S_TARGETS)
                has_nan = qdp.__dataframe__.isna().any().any()
                if not has_nan:
                    data_df = pd.concat([data_df, qdp.get_dataframe()])
    for target in XGB_TARGETS:
        resampled_df = resample_df(data_df, target, S_TARGETS)
        yield resampled_df


train_content, test_content = load_content(PATH)
if XGB:
    (train_1, train_2, train_3, train_4, train_5, train_6) = xgb_pipeline(train_content)
    input()
    exit()
    qmodel_1 = QModel(
        dataset=train_1, predictors=FEATURES, target_features=S_TARGETS[0]
    )
    qmodel_2 = QModel(
        dataset=train_2, predictors=FEATURES, target_features=S_TARGETS[1]
    )
    qmodel_3 = QModel(
        dataset=train_3, predictors=FEATURES, target_features=S_TARGETS[2]
    )
    qmodel_4 = QModel(
        dataset=train_4, predictors=FEATURES, target_features=S_TARGETS[3]
    )
    qmodel_5 = QModel(
        dataset=train_5, predictors=FEATURES, target_features=S_TARGETS[4]
    )
    qmodel_6 = QModel(
        dataset=train_6, predictors=FEATURES, target_features=S_TARGETS[5]
    )

    qmodel_1.tune(15)
    qmodel_1.train_model()
    qmodel_1.save_model("QModel_1")
    qmodel_2.tune(15)
    qmodel_2.train_model()
    qmodel_2.save_model("QModel_2")
    qmodel_3.tune(15)
    qmodel_3.train_model()
    qmodel_3.save_model("QModel_3")
    qmodel_4.tune(15)
    qmodel_4.train_model()
    qmodel_4.save_model("QModel_4")
    qmodel_5.tune(15)
    qmodel_5.train_model()
    qmodel_5.save_model("QModel_5")
    qmodel_6.tune(15)
    qmodel_6.train_model()
    qmodel_6.save_model("QModel_6")


PATH = "content/validation_datasets"
data_df = pd.DataFrame()
content = []
qmp = QModelPredict(
    "QModel/SavedModels/QModel_1.json",
    "QModel/SavedModels/QModel_2.json",
    "QModel/SavedModels/QModel_3.json",
    "QModel/SavedModels/QModel_4.json",
    "QModel/SavedModels/QModel_5.json",
    "QModel/SavedModels/QModel_6.json",
)

for root, dirs, files in os.walk(PATH):
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
