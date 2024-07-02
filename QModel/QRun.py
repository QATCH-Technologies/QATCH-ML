import os
import pandas as pd
from QModel import QModel, QModelPredict
from QDataPipline import QDataPipeline
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import xgboost as xgb

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
    "Resonance_Frequency_gradient",
    "Cumulative",
    "Difference",
    "Difference_gradient",
    "Dissipation_gradient",
]
PREDICTORS_2 = [
    "Relative_time",
    "Dissipation",
    "Resonance_Frequency_gradient",
    "Cumulative",
    "Difference",
    "Difference_gradient",
    "Dissipation_gradient",
]
PREDICTORS_3 = [
    "Relative_time",
    "Dissipation",
    "Resonance_Frequency_gradient",
    "Cumulative",
    "Difference",
    "Difference_gradient",
    "Dissipation_gradient",
]
PREDICTORS_4 = [
    "Relative_time",
    "Dissipation",
    "Resonance_Frequency_gradient",
    "Cumulative",
    "Difference",
    "Difference_gradient",
    "Dissipation_gradient",
]

PREDICTORS_5 = [
    "Relative_time",
    "Dissipation",
    "Resonance_Frequency_gradient",
    "Cumulative",
    "Difference",
    "Difference_gradient",
    "Dissipation_gradient",
]
PREDICTORS_6 = [
    "Relative_time",
    "Dissipation",
    "Resonance_Frequency_gradient",
    "Cumulative",
    "Difference",
    "Difference_gradient",
    "Dissipation_gradient",
]


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


PLOTTING = True
PATH = "content/training_data_with_points"
data_df = pd.DataFrame()
content = []
for root, dirs, files in os.walk(PATH):
    for file in files:
        content.append(os.path.join(root, file))

# i = 0
for filename in tqdm(content, desc="Processing Files"):
    # if i > 50:
    #     break
    # i += 1
    if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
        data_file = filename
        if max(pd.read_csv(data_file)["Relative_time"].values) < 90:
            poi_file = filename.replace(".csv", "_poi.csv")
            qdp = QDataPipeline(data_file)
            qdp.preprocess(poi_file=poi_file)

            indices = qdp.__dataframe__.index[
                qdp.__dataframe__["Class_1"] != 0
            ].values.tolist()
            indices.append(
                qdp.__dataframe__.index[
                    qdp.__dataframe__["Class_2"] != 0
                ].values.tolist()
            )
            # plt.figure()
            # plt.plot(qdp.__dataframe__["Dissipation_gradient"].values, color="g")
            # plt.plot(normalize(qdp.__dataframe__["Dissipation"].values), color="b")
            # plt.plot(normalize(qdp.__dataframe__["Cumulative"].values), color="r")
            # for index in indices:
            #     plt.axvline(x=index)
            # plt.show()
            data_df = pd.concat([data_df, qdp.get_dataframe()])
data_df.set_index("Relative_time")
print(data_df.head())

# Calculate the correlation matrix
corr_matrix = data_df.corr()

# Plot the heatmap
plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.show()
print("\rCreating training dataset...Done")

qmodel_all = QModel(dataset=data_df, predictors=PREDICTORS_1, target_features="Class")
qmodel_1 = QModel(dataset=data_df, predictors=PREDICTORS_1, target_features=TARGET_1)
qmodel_2 = QModel(dataset=data_df, predictors=PREDICTORS_2, target_features=TARGET_2)
qmodel_3 = QModel(dataset=data_df, predictors=PREDICTORS_3, target_features=TARGET_3)
qmodel_4 = QModel(dataset=data_df, predictors=PREDICTORS_4, target_features=TARGET_4)
qmodel_5 = QModel(dataset=data_df, predictors=PREDICTORS_5, target_features=TARGET_5)
qmodel_6 = QModel(dataset=data_df, predictors=PREDICTORS_6, target_features=TARGET_6)

qmodel_all.tune(15)
qmodel_1.tune(15)
qmodel_2.tune(15)
qmodel_3.tune(15)
qmodel_4.tune(15)
qmodel_5.tune(15)
qmodel_6.tune(15)

qmodel_all.train_model()
qmodel_1.train_model()
qmodel_2.train_model()
qmodel_3.train_model()
qmodel_4.train_model()
qmodel_5.train_model()
qmodel_6.train_model()
# xgb.plot_importance(qmodel_all.__model__, importance_type="gain")
# xgb.plot_importance(qmodel_1.__model__, importance_type="gain")
# xgb.plot_importance(qmodel_2.__model__, importance_type="gain")
# xgb.plot_importance(qmodel_3.__model__, importance_type="gain")
# xgb.plot_importance(qmodel_4.__model__, importance_type="gain")
# xgb.plot_importance(qmodel_5.__model__, importance_type="gain")
# xgb.plot_importance(qmodel_6.__model__, importance_type="gain")
plt.show()
qmodel_all.save_model("QModel_all")
qmodel_1.save_model("QModel_1")
qmodel_2.save_model("QModel_2")
qmodel_3.save_model("QModel_3")
qmodel_4.save_model("QModel_4")
qmodel_5.save_model("QModel_5")
qmodel_6.save_model("QModel_6")
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
        results_all = normalize(qpreditor_all.predict(data_file))
        results_1 = normalize(qpreditor_1.predict(data_file))
        results_2 = normalize(qpreditor_2.predict(data_file))
        results_3 = normalize(qpreditor_3.predict(data_file))
        # results_4 = normalize(qpreditor_4.predict(data_file))
        # results_5 = normalize(qpreditor_5.predict(data_file))
        # results_6 = normalize(qpreditor_6.predict(data_file))
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
                results_1,
                color="black",
                label="Confidence_1",
            )
            plt.plot(
                results_1,
                color="darkred",
                label="Confidence_1",
            )
            plt.plot(
                results_2,
                color="red",
                label="Confidence_2",
            )
            plt.plot(
                results_3,
                color="yellow",
                label="Confidence_3",
            )
            # plt.plot(
            #     results_4,
            #     color="firebrick",
            #     label="Confidence_4",
            # )
            # plt.plot(
            #     results_5,
            #     color="red",
            #     label="Confidence_5",
            # )
            # plt.plot(
            #     results_6,
            #     color="lightcoral",
            #     label="Confidence_6",
            # )
            plt.plot(
                dissipation,
                color="blue",
                label="Dissipation",
            )
            # plt.plot(
            #     difference,
            #     color="darkorange",
            #     label="Difference",
            # )
            # plt.plot(
            #     resonance_frequency,
            #     color="deeppink",
            #     label="Resonance Frequency",
            # )

            # plt.scatter(
            #     peaks,
            #     difference[peaks],
            #     marker="o",
            #     color="darkviolet",
            # )
            # plt.scatter(
            #     peaks,
            #     resonance_frequency[peaks],
            #     marker="o",
            #     color="darkviolet",
            # )
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
