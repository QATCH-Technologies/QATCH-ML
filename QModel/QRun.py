import os
import pandas as pd
from QModel import QModel, QModelPredict
from QDataPipline import QDataPipeline
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from joblib import dump, load
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


from ModelData import ModelData

# pd.set_option("display.max_rows", None)
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


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def tsne_view(X, y):
    print("[INFO] building t-SNE")
    tsne = TSNE(n_components=2, random_state=42)
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


def resample_arr(data, target):
    print(f"[INFO] resampling {target}")
    y = data[target].values
    X = data.drop(columns=META_TARGETS)
    over = SMOTE(sampling_strategy="auto")
    under = RandomUnderSampler(sampling_strategy="majority")
    steps = [
        # ("over_sample", over),
        ("under_sample", under),
    ]
    pipeline = Pipeline(steps=steps)

    X, y = pipeline.fit_resample(X, y)

    X = RobustScaler().fit_transform(X)

    tsne_view(X, y)
    pca_view(X, y)
    return X, y


def resample_df(data, target):
    print(f"[INFO] resampling df {target}")
    y = data[target].values
    X = data.drop(columns=XGB_TARGETS)

    over = SMOTE()
    under = RandomUnderSampler()
    steps = [("o", over), ("u", under)]
    pipeline = Pipeline(steps=steps)

    X, y = pipeline.fit_resample(X, y)

    resampled_df = pd.DataFrame(X, columns=data.drop(columns=XGB_TARGETS).columns)
    resampled_df[target] = y

    return resampled_df


def load_content(data_dir):
    content = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            content.append(os.path.join(root, file))

    train_content, test_content = train_test_split(
        content, test_size=0.5, random_state=42
    )
    return train_content, test_content


def xgb_pipeline(train_content):
    data_df = pd.DataFrame()
    for filename in tqdm(train_content, desc="<<Processing XGB>>"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            if max(pd.read_csv(data_file)["Relative_time"].values) < 90:
                poi_file = filename.replace(".csv", "_poi.csv")
                qdp = QDataPipeline(data_file)
                qdp.preprocess(poi_file=poi_file)

                has_nan = qdp.__dataframe__.isna().any().any()
                if not has_nan:
                    data_df = pd.concat([data_df, qdp.get_dataframe()])
    for target in XGB_TARGETS:
        resampled_df = resample_df(data_df, target)
        yield resampled_df


def meta_model_builder(dataset):
    data_df = pd.DataFrame()
    qpreditor_1 = QModelPredict(model_path="QModel/SavedModels/QModel_1.json")
    qpreditor_2 = QModelPredict(model_path="QModel/SavedModels/QModel_2.json")
    qpreditor_3 = QModelPredict(model_path="QModel/SavedModels/QModel_3.json")
    qpreditor_4 = QModelPredict(model_path="QModel/SavedModels/QModel_4.json")
    qpreditor_5 = QModelPredict(model_path="QModel/SavedModels/QModel_5.json")
    qpreditor_6 = QModelPredict(model_path="QModel/SavedModels/QModel_6.json")
    for filename in tqdm(dataset, desc="<<Processing Meta Model>>"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            if max(pd.read_csv(data_file)["Relative_time"].values) < 90:
                poi_file = filename.replace(".csv", "_poi.csv")
                indices = pd.read_csv(poi_file, header=None).values
                indices = indices.flatten().tolist()
                _, bound_1 = qpreditor_1.predict(data_file)
                _, bound_2 = qpreditor_2.predict(data_file)
                _, bound_3 = qpreditor_3.predict(data_file)
                _, bound_4 = qpreditor_4.predict(data_file)
                _, bound_5 = qpreditor_5.predict(data_file)
                _, bound_6 = qpreditor_6.predict(data_file)
                ranges = [
                    bound_1[0],
                    bound_2[0],
                    bound_3[0],
                    bound_4[0],
                    bound_5[0],
                    bound_6[0],
                ]

                emp_predictor = ModelData()
                emp_result = emp_predictor.IdentifyPoints(data_path=data_file)
                emp_result = [
                    item if isinstance(item, int) else [val[0] for val in item]
                    for item in emp_result
                    if isinstance(item, (int, list))
                ]
                emp_result = [
                    item
                    for sublist in emp_result
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                ]

                num_rows = max(
                    max(max(t) for t in ranges),
                    max(emp_result),
                    max(indices),
                )

                predictions = pd.DataFrame(index=range(int(num_rows) + 1))
                predictions["XGB"] = 0
                predictions["EMP"] = 0
                predictions["Class_1"] = 0
                predictions["Class_2"] = 0
                predictions["Class_3"] = 0
                predictions["Class_4"] = 0
                predictions["Class_5"] = 0
                predictions["Class_6"] = 0

                for poi, (i, j) in enumerate(ranges):
                    for idx in range(i, j):
                        predictions.loc[idx, "XGB"] = poi + 1

                for idx in emp_result:
                    predictions.loc[idx, "EMP"] = 1

                for poi, idx in enumerate(indices):
                    predictions.loc[idx, "Class_" + str(poi + 1)] = 1

                predictions = predictions.fillna(0)
                data_df = pd.concat([data_df, predictions])

    for target in META_TARGETS:
        meta_clf = RandomForestClassifier()
        meta_X, meta_y = resample_arr(data_df, target)
        meta_clf.fit(meta_X, meta_y)
        yield meta_clf


PLOTTING = True
XGB = True
PATH = "content/training_data_with_points"
if XGB:
    train_content, test_content = load_content(PATH)
    (train_1, train_2, train_3, train_4, train_5, train_6) = xgb_pipeline(train_content)

    # qmodel_1 = QModel(dataset=train_1, predictors=FEATURES, target_features=TARGETS[0])
    # qmodel_2 = QModel(dataset=train_2, predictors=FEATURES, target_features=TARGETS[1])
    # qmodel_3 = QModel(dataset=train_3, predictors=FEATURES, target_features=TARGETS[2])
    # qmodel_4 = QModel(dataset=train_4, predictors=FEATURES, target_features=TARGETS[3])
    # qmodel_5 = QModel(dataset=train_5, predictors=FEATURES, target_features=TARGETS[4])
    # qmodel_6 = QModel(dataset=train_6, predictors=FEATURES, target_features=TARGETS[5])

    # qmodel_1.tune(15)
    # qmodel_2.tune(15)
    # qmodel_3.tune(15)
    # qmodel_4.tune(15)
    # qmodel_5.tune(15)
    # qmodel_6.tune(15)

    # qmodel_1.train_model()
    # qmodel_2.train_model()
    # qmodel_3.train_model()
    # qmodel_4.train_model()
    # qmodel_5.train_model()
    # qmodel_6.train_model()

    # qmodel_1.save_model("QModel_1")
    # qmodel_2.save_model("QModel_2")
    # qmodel_3.save_model("QModel_3")
    # qmodel_4.save_model("QModel_4")
    # qmodel_5.save_model("QModel_5")
    # qmodel_6.save_model("QModel_6")
    (
        meta_model_1,
        meta_model_2,
        meta_model_3,
        meta_model_4,
        meta_model_5,
        meta_model_6,
    ) = meta_model_builder(test_content)
    dump(meta_model_1, "QModel/SavedModels/meta_model_1.joblib")
    dump(meta_model_2, "QModel/SavedModels/meta_model_2.joblib")
    dump(meta_model_3, "QModel/SavedModels/meta_model_3.joblib")
    dump(meta_model_4, "QModel/SavedModels/meta_model_4.joblib")
    dump(meta_model_5, "QModel/SavedModels/meta_model_5.joblib")
    dump(meta_model_6, "QModel/SavedModels/meta_model_6.joblib")
ERROR = 5
correct = 0
incorrect = 0


def compute_error(actual, predicted):
    return_flag = False
    if len(predicted) != len(actual):
        print(f"Actual and predicted are not the same length. found {len(predicted)}")
        return True
    for i in range(len(actual)):
        if actual[i] - predicted[i] > ERROR:
            return_flag = True
            print(
                f"Found error (pt {i + 1}, actual, predicted, difference): {actual[i]} - {predicted[i]} = {actual[i] - predicted[i]}"
            )

    return return_flag


meta_model_1 = load("QModel/SavedModels/meta_model_1.joblib")
meta_model_2 = load("QModel/SavedModels/meta_model_2.joblib")
meta_model_3 = load("QModel/SavedModels/meta_model_3.joblib")
meta_model_4 = load("QModel/SavedModels/meta_model_4.joblib")
meta_model_5 = load("QModel/SavedModels/meta_model_5.joblib")
meta_model_6 = load("QModel/SavedModels/meta_model_6.joblib")

qpreditor_1 = QModelPredict(model_path="QModel/SavedModels/QModel_1.json")
qpreditor_2 = QModelPredict(model_path="QModel/SavedModels/QModel_2.json")
qpreditor_3 = QModelPredict(model_path="QModel/SavedModels/QModel_3.json")
qpreditor_4 = QModelPredict(model_path="QModel/SavedModels/QModel_4.json")
qpreditor_5 = QModelPredict(model_path="QModel/SavedModels/QModel_5.json")
qpreditor_6 = QModelPredict(model_path="QModel/SavedModels/QModel_6.json")
PATH = "content/VOYAGER_PROD_DATA"
data_df = pd.DataFrame()
content = []
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
        qdp.__dataframe__ = qdp.__dataframe__.drop(columns=["Class", "Pooling"])

        results_1, bound_1 = qpreditor_1.predict(data_file)
        results_2, bound_2 = qpreditor_2.predict(data_file)
        results_3, bound_3 = qpreditor_3.predict(data_file)
        results_4, bound_4 = qpreditor_4.predict(data_file)
        results_5, bound_5 = qpreditor_5.predict(data_file)
        results_6, bound_6 = qpreditor_6.predict(data_file)

        ranges = [
            bound_1[0],
            bound_2[0],
            bound_3[0],
            bound_4[0],
            bound_5[0],
            bound_6[0],
        ]

        emp_predictor = ModelData()
        emp_result = emp_predictor.IdentifyPoints(data_path=data_file)
        emp_result = [
            item if isinstance(item, int) else [val[0] for val in item]
            for item in emp_result
            if isinstance(item, (int, list))
        ]
        emp_result = [
            item
            for sublist in emp_result
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]

        num_rows = max(
            max(max(t) for t in ranges),
            max(emp_result),
        )

        metadata = pd.DataFrame(index=range(int(num_rows) + 1))
        metadata["XGB"] = 0
        metadata["EMP"] = 0

        for i, j in ranges:
            for idx in range(i, j):
                metadata.loc[idx, "XGB"] = 1

        for idx in emp_result:
            metadata.loc[idx, "EMP"] = 1

        metadata = metadata.fillna(0)

        predictions_1 = meta_model_1.predict(metadata)
        predictions_2 = meta_model_2.predict(metadata)
        predictions_3 = meta_model_3.predict(metadata)
        predictions_4 = meta_model_4.predict(metadata)
        predictions_5 = meta_model_5.predict(metadata)
        predictions_6 = meta_model_6.predict(metadata)
        # for p in predictions:
        #     print(p)
        # input()
        if PLOTTING:
            df = pd.read_csv(data_file)
            dissipation = normalize(df["Dissipation"])
            plt.figure()
            plt.plot(predictions_1, label="Predictions_1")
            plt.plot(predictions_2, label="Predictions_2")
            plt.plot(predictions_3, label="Predictions_3")
            plt.plot(predictions_4, label="Predictions_4")
            plt.plot(predictions_5, label="Predictions_5")
            plt.plot(predictions_6, label="Predictions_6")
            # plt.plot(results_5)
            # for left, right in bound_1:
            #     plt.fill_between(
            #         np.arange(len(results_1))[left : right + 1],
            #         results_1[left : right + 1],
            #         alpha=0.5,
            #         color="red",
            #         label="1",
            #     )
            # for left, right in bound_2:
            #     plt.fill_between(
            #         np.arange(len(results_2))[left : right + 1],
            #         results_2[left : right + 1],
            #         color="orange",
            #         label="2",
            #     )
            # for left, right in bound_3:
            #     plt.fill_between(
            #         np.arange(len(results_3))[left : right + 1],
            #         results_2[left : right + 1],
            #         color="yellow",
            #         label="3",
            #     )
            # for left, right in bound_4:
            #     plt.fill_between(
            #         np.arange(len(results_4))[left : right + 1],
            #         results_4[left : right + 1],
            #         alpha=0.5,
            #         color="green",
            #         label="4",
            #     )
            # for left, right in bound_5:
            #     plt.fill_between(
            #         np.arange(len(results_5))[left : right + 1],
            #         results_5[left : right + 1],
            #         alpha=0.5,
            #         color="blue",
            #         label="5",
            #     )
            # for left, right in bound_6:
            #     plt.fill_between(
            #         np.arange(len(results_6))[left : right + 1],
            #         results_6[left : right + 1],
            #         alpha=0.5,
            #         color="purple",
            #         label="6",
            #     )
            plt.plot(
                dissipation,
                color="grey",
                label="Dissipation",
            )
            print(actual_indices)
            plt.axvline(
                x=actual_indices[0],
                color="black",
                linestyle="--",
                label="Actual",
            )
            for index in actual_indices:
                plt.axvline(
                    x=index,
                    color="black",
                    linestyle="--",
                )
            plot_name = data_file.replace(PATH, "")
            plt.xlabel("POIs")
            plt.ylabel("Dissipation")
            plt.title(f"Predicted/Actual POIs on Data: {plot_name}")

            plt.legend()
            plt.grid(True)
            plt.show()
