import os
import pandas as pd
from QModel import QModel, QModelPredict
from QDataPipline import QDataPipeline, QEncoder
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import xgboost as xgb
from tensorflow.keras.models import load_model

# pd.set_option("display.max_rows", None)
TARGET_1 = "Class_1"
TARGET_2 = "Class_2"
TARGET_3 = "Class_3"
TARGET_4 = "Class_4"
TARGET_5 = "Class_5"
TARGET_6 = "Class_6"
""" Training features for the pooling model. """
PREDICTORS_1 = [
    "Relative_time",
    "Dissipation",
    "Cumulative",
    "Difference",
    "Resonance_Frequency_gradient",
    "Difference_gradient",
    "Dissipation_gradient",
    "Difference_detrend",
    "Resonance_Frequency_detrend",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Difference_super",
    "Resonance_Frequency_super",
    "Cumulative_super",
    "Dissipation_super",
]
PREDICTORS_2 = [
    "Relative_time",
    "Dissipation",
    "Cumulative",
    "Difference",
    "Resonance_Frequency_gradient",
    "Difference_gradient",
    "Dissipation_gradient",
    "Difference_detrend",
    "Resonance_Frequency_detrend",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Difference_super",
    "Resonance_Frequency_super",
    "Cumulative_super",
    "Dissipation_super",
]
PREDICTORS_3 = [
    "Relative_time",
    "Dissipation",
    "Cumulative",
    "Difference",
    "Resonance_Frequency_gradient",
    "Difference_gradient",
    "Dissipation_gradient",
    "Difference_detrend",
    "Resonance_Frequency_detrend",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Difference_super",
    "Resonance_Frequency_super",
    "Cumulative_super",
    "Dissipation_super",
]
PREDICTORS_4 = [
    "Relative_time",
    "Dissipation",
    "Cumulative",
    "Difference",
    "Resonance_Frequency_gradient",
    "Difference_gradient",
    "Dissipation_gradient",
    "Difference_detrend",
    "Resonance_Frequency_detrend",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Difference_super",
    "Resonance_Frequency_super",
    "Cumulative_super",
    "Dissipation_super",
]

PREDICTORS_5 = [
    "Relative_time",
    "Dissipation",
    "Cumulative",
    "Difference",
    "Resonance_Frequency_gradient",
    "Difference_gradient",
    "Dissipation_gradient",
    "Difference_detrend",
    "Resonance_Frequency_detrend",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Difference_super",
    "Resonance_Frequency_super",
    "Cumulative_super",
    "Dissipation_super",
]
PREDICTORS_6 = [
    "Relative_time",
    "Dissipation",
    "Cumulative",
    "Difference",
    "Resonance_Frequency_gradient",
    "Difference_gradient",
    "Dissipation_gradient",
    "Difference_detrend",
    "Resonance_Frequency_detrend",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Difference_super",
    "Resonance_Frequency_super",
    "Cumulative_super",
    "Dissipation_super",
]


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


PLOTTING = True
TRAINING = False
if TRAINING:
    PATH = "content/training_data_with_points"
    data_df = pd.DataFrame()
    content = []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            content.append(os.path.join(root, file))

    for filename in tqdm(content, desc="Processing Files"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            if max(pd.read_csv(data_file)["Relative_time"].values) < 90:
                poi_file = filename.replace(".csv", "_poi.csv")
                qdp = QDataPipeline(data_file)
                qdp.preprocess(poi_file=poi_file)

                has_nan = qdp.__dataframe__.isna().any().any()
                if not has_nan:
                    data_df = pd.concat([data_df, qdp.get_dataframe()])
    encoder_df = data_df.drop(
        columns=[TARGET_1, TARGET_2, TARGET_3, TARGET_4, "Class", TARGET_6]
    )
    print(encoder_df.head())
    QEncoder(
        encoder_df,
        TARGET_5,
    )
    encoder = load_model("QModel/SavedModels/encoder.h5")
    for filename in tqdm(content, desc="Processing Files"):
        if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
            data_file = filename
            if max(pd.read_csv(data_file)["Relative_time"].values) < 90:
                poi_file = filename.replace(".csv", "_poi.csv")
                qdp = QDataPipeline(data_file)
                qdp.preprocess(poi_file=poi_file)

                # encode the train data
                encoding = encoder.predict(
                    qdp.__dataframe__.drop(
                        columns=[
                            TARGET_1,
                            TARGET_2,
                            TARGET_3,
                            TARGET_4,
                            TARGET_5,
                            "Class",
                            TARGET_6,
                        ]
                    )
                )
                print(encoding)
                input()
                has_nan = qdp.__dataframe__.isna().any().any()
                if not has_nan:
                    data_df = pd.concat([data_df, qdp.get_dataframe()])
    # Calculate the correlation matrix
    data_df.set_index("Relative_time")
    corr_matrix = data_df.corr()

    # Plot the heatmap
    plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.show()
    print("\rCreating training dataset...Done")

    # qmodel_all = QModel(
    #     dataset=data_df, predictors=PREDICTORS_1, target_features="Class"
    # )
    # qmodel_1 = QModel(
    #     dataset=data_df, predictors=PREDICTORS_1, target_features=TARGET_1
    # )
    # qmodel_2 = QModel(
    #     dataset=data_df, predictors=PREDICTORS_2, target_features=TARGET_2
    # )
    # qmodel_3 = QModel(
    #     dataset=data_df, predictors=PREDICTORS_3, target_features=TARGET_3
    # )
    # qmodel_4 = QModel(
    #     dataset=data_df, predictors=PREDICTORS_4, target_features=TARGET_4
    # )
    qmodel_5 = QModel(
        dataset=data_df, predictors=PREDICTORS_5, target_features=TARGET_5
    )
    # qmodel_6 = QModel(
    #     dataset=data_df, predictors=PREDICTORS_6, target_features=TARGET_6
    # )

    # qmodel_all.tune(15)
    # qmodel_1.tune(15)
    # qmodel_2.tune(15)
    # qmodel_3.tune(15)
    # qmodel_4.tune(15)
    qmodel_5.tune(15)
    # qmodel_6.tune(15)

    # qmodel_all.train_model()
    # qmodel_1.train_model()
    # qmodel_2.train_model()
    # qmodel_3.train_model()
    # qmodel_4.train_model()
    qmodel_5.train_model()
    # qmodel_6.train_model()
    # xgb.plot_importance(qmodel_all.__model__, importance_type="gain")
    # plt.show()
    # xgb.plot_importance(qmodel_1.__model__, importance_type="gain")
    # plt.show()
    # xgb.plot_importance(qmodel_2.__model__, importance_type="gain")
    # plt.show()
    # xgb.plot_importance(qmodel_3.__model__, importance_type="gain")
    # plt.show()
    # xgb.plot_importance(qmodel_4.__model__, importance_type="gain")
    # plt.show()
    xgb.plot_importance(qmodel_5.__model__, importance_type="weight")
    plt.show()
    # xgb.plot_importance(qmodel_6.__model__, importance_type="gain")
    # plt.show()
    # qmodel_all.save_model("QModel_all")
    # qmodel_1.save_model("QModel_1")
    # qmodel_2.save_model("QModel_2")
    # qmodel_3.save_model("QModel_3")
    # qmodel_4.save_model("QModel_4")
    qmodel_5.save_model("QModel_5")
    # qmodel_6.save_model("QModel_6")
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


qpreditor_all = QModelPredict(model_path="QModel/SavedModels/QModel_all.json")
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
        poi_file = filename.replace(".csv", "_poi.csv")
        results_all, _ = qpreditor_all.predict(data_file)
        results_1, bound_1 = qpreditor_1.predict(data_file)
        results_2, bound_2 = qpreditor_2.predict(data_file)
        results_3, bound_3 = qpreditor_3.predict(data_file)
        results_4, bound_4 = qpreditor_4.predict(data_file)
        results_5, bound_5 = qpreditor_5.predict(data_file)
        results_6, bound_6 = qpreditor_6.predict(data_file)
        actual_indices = pd.read_csv(poi_file, header=None).values
        # print(f"Predic: {peaks}")
        print(f"Actual: {actual_indices}")

        # if compute_error(actual_indices, peaks):
        #    incorrect += 1
        if PLOTTING:
            df = pd.read_csv(data_file)
            dissipation = normalize(df["Dissipation"])
            # difference = df["Difference"]
            # difference = np.abs(difference)
            # resonance_frequency = df["Resonance_Frequency"]
            # difference = normalize(difference)
            # resonance_frequency = normalize(resonance_frequency)
            plt.figure()
            plt.plot(
                results_all,
                color="black",
                label="Confidence_all",
            )
            for left, right in bound_1:
                plt.fill_between(
                    np.arange(len(results_1))[left : right + 1],
                    results_1[left : right + 1],
                    alpha=0.5,
                    color="orange",
                    label="1",
                )
            for left, right in bound_2:
                plt.fill_between(
                    np.arange(len(results_2))[left : right + 1],
                    results_2[left : right + 1],
                    alpha=0.5,
                    color="purple",
                    label="2",
                )
            for left, right in bound_3:
                plt.fill_between(
                    np.arange(len(results_3))[left : right + 1],
                    results_3[left : right + 1],
                    alpha=0.5,
                    color="y",
                    label="3",
                )
            for left, right in bound_4:
                plt.fill_between(
                    np.arange(len(results_4))[left : right + 1],
                    results_4[left : right + 1],
                    alpha=0.5,
                    color="b",
                    label="4",
                )
            for left, right in bound_5:
                plt.fill_between(
                    np.arange(len(results_5))[left : right + 1],
                    results_5[left : right + 1],
                    alpha=0.5,
                    color="g",
                    label="5",
                )
            for left, right in bound_6:
                plt.fill_between(
                    np.arange(len(results_6))[left : right + 1],
                    results_6[left : right + 1],
                    alpha=0.5,
                    color="r",
                    label="6",
                )
            plt.plot(
                dissipation,
                color="blue",
                label="Dissipation",
            )
            print(actual_indices)
            plt.axvline(
                x=actual_indices[0],
                color="dodgerblue",
                linestyle="--",
                label="Actual",
            )
            for index in actual_indices:
                plt.axvline(
                    x=index,
                    color="dodgerblue",
                    linestyle="--",
                )
            plot_name = data_file.replace(PATH, "")
            plt.xlabel("POIs")
            plt.ylabel("Dissipation")
            plt.title(f"Predicted/Actual POIs on Data: {plot_name}")

            plt.legend()
            plt.grid(True)
            plt.show()
