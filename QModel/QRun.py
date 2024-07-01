import os
import pandas as pd
from QModel import QModel, QModelPredict
from QDataPipline import QDataPipeline
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import xgboost as xgb


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


PLOTTING = True
PATH = "content/training_data_with_points"
data_df = pd.DataFrame()
content = []
for root, dirs, files in os.walk(PATH):
    for file in files:
        content.append(os.path.join(root, file))

i = 0
for filename in tqdm(content, desc="Processing Files"):
    if i > 50:
        break
    i += 1
    if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
        data_file = filename
        poi_file = filename.replace(".csv", "_poi.csv")
        qdp = QDataPipeline(data_file)
        qdp.__dataframe__.drop(
            columns=[
                "Date",
                "Time",
                "Ambient",
                "Temperature",
                "Peak Magnitude (RAW)",
            ],
            inplace=True,
        )

        poi_vals = pd.read_csv(poi_file, header=None)
        qdp.fill_nan()
        qdp.add_class(poi_file)
        head_idx = qdp.trim_head()
        if head_idx > min(poi_vals.values):
            print(f"head: {head_idx}, min poi: {min(poi_vals.values)}")
            raise ValueError("Less than 6 POIs; too agressive trim")
        qdp.interpolate()
        qdp.compute_difference()
        qdp.noise_filter("Cumulative")
        # qdp.normalize_df()
        qdp.standardize("Cumulative")
        # qdp.standardize("Dissipation")
        qdp.compute_smooth(column="Dissipation", winsize=25, polyorder=1)
        qdp.compute_smooth(column="Difference", winsize=25, polyorder=1)
        qdp.compute_smooth(column="Resonance_Frequency", winsize=25, polyorder=1)

        qdp.compute_gradient(column="Dissipation")
        qdp.compute_gradient(column="Difference")
        qdp.compute_gradient(column="Resonance_Frequency")
        # print(normalize(qdp.__dataframe__["Dissipation_gradient"].values))
        indices = qdp.__dataframe__.index[qdp.__dataframe__["Class"] != 0]
        # plt.figure()
        # plt.plot(qdp.__dataframe__["Dissipation_gradient"].values, color="g")
        # plt.plot(normalize(qdp.__dataframe__["Dissipation"].values), color="b")
        # plt.plot(normalize(qdp.__dataframe__["Cumulative"].values), color="r")
        # for index in indices:
        #     plt.axvline(x=index)
        # plt.show()
        data_df = pd.concat([data_df, qdp.get_dataframe()])
# data_df.drop(columns=["Relative_time"], inplace=True)
print("\rCreating training dataset...Done")

# Calculate the correlation matrix
corr_matrix = data_df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap")
plt.show()

qmodel = QModel(data_df)
# qmodel.tune(15)
qmodel.train_model()
xgb.plot_importance(qmodel.__model__, importance_type="gain", max_num_features=10)
plt.show()
qmodel.save_model()
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


qpreditor = QModelPredict(model_path="QModel/SavedModels/QModel.json")

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
        results = normalize(qpreditor.predict(data_file))

        actual_indices = pd.read_csv(poi_file, header=None).values
        # print(f"Predic: {peaks}")
        print(f"Actual: {actual_indices}")

        # if compute_error(actual_indices, peaks):
        #    incorrect += 1
        if PLOTTING:
            df = pd.read_csv(data_file)
            dissipation = df["Dissipation"]
            # difference = df["Difference"]
            # difference = np.abs(difference)
            # resonance_frequency = df["Resonance_Frequency"]
            # difference = normalize(difference)
            # resonance_frequency = normalize(resonance_frequency)
            plt.figure(figsize=(8, 6))
            plt.plot(
                results,
                color="darkviolet",
                label="Confidence",
            )
            plt.plot(
                dissipation,
                color="gold",
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

# # For training_data_with_points directory
# i = 0
# for filename in os.listdir(PATH):
#     if i > 100:
#         break
#     if filename.endswith(".csv") and not filename.endswith("_poi.csv"):
#         print(f"Concatenating {filename}...")
#         file_data = pd.read_csv(os.path.join(PATH, filename))
#         data_df = pd.concat([data_df, file_data])
#     i = i + 1

# For VOYAGER_PROD_DATA direcotry
# for folder_name in os.listdir(PATH):
#     folder_path = os.path.join(PATH, folder_name)

#     if os.path.isdir(folder_path):
#         # Traverse through the second level directories
#         for sub_folder_name in os.listdir(folder_path):
#             sub_folder_path = os.path.join(folder_path, sub_folder_name)

#             if os.path.isdir(sub_folder_path):
#                 if os.path.isdir(sub_folder_path):
#                     for sub_sub_folder_name in os.listdir(sub_folder_path):
#                         for filename in os.listdir(sub_folder_path):
#                             for sub_sub_folder_name in os.listdir(sub_folder_path):
#                                 sub_sub_folder_path = os.path.join(
#                                     sub_folder_path, sub_sub_folder_name
#                                 )
#                                 if sub_sub_folder_path.endswith(
#                                     ".csv"
#                                 ) and not sub_sub_folder_path.endswith("_poi.csv"):
#                                     print(f"Concatenating {sub_sub_folder_path}...")
#                                     file_data = pd.read_csv(sub_sub_folder_path)
#                                     data_df = pd.concat([data_df, file_data])
# #
# qModel = QModel(data_df)
# # qModel.tune("p", 10)
# qModel.tune("d", 10)
# # qModel.train_pooler()
# qModel.train_discriminator()
# # qModel.save_pooler()
# qModel.save_discriminator()


# Extract columns from the DataFrame


# for folder_name in os.listdir(PATH):
#     folder_path = os.path.join(PATH, folder_name)

#     if os.path.isdir(folder_path):
#         # Traverse through the second level directories
#         for sub_folder_name in os.listdir(folder_path):
#             sub_folder_path = os.path.join(folder_path, sub_folder_name)

#             if os.path.isdir(sub_folder_path):
#                 if os.path.isdir(sub_folder_path):
#                     for filename in os.listdir(sub_folder_path):
#                         filename = os.path.join(sub_folder_path, filename)
#                         if filename.endswith("_poi.csv"):
#                             poi_file = filename
#                         elif filename.endswith(".csv") and not filename.endswith(
#                             "_poi.csv"
#                         ):
#                             data_file = filename
#                     print(poi_file)
#                     actual_indices = pd.read_csv(poi_file, header=None).values
#                     actual_indices = [item[0] for item in actual_indices]
#                     print(f"Analyzing {data_file}")
#                     (
#                         peaks,
#                         pooler_results,
#                         discriminator_results,
#                         s_bound,
#                         t_bound,
#                     ) = qpreditor.predict(data_file)

#                     print(f"Predic: {peaks}")
#                     print(f"Actual: {actual_indices}")

#                     if compute_error(actual_indices, peaks):
#                         incorrect += 1
#                         if PLOTTING:
#                             df = pd.read_csv(data_file)
#                             dissipation = df["Dissipation"]
#                             difference = df["Difference"]
#                             difference = np.abs(difference)
#                             resonance_frequency = df["Resonance_Frequency"]
#                             difference = normalize(difference)
#                             resonance_frequency = normalize(resonance_frequency)
#                             dissipation = normalize(dissipation)
#                             plt.figure(figsize=(8, 6))
#                             plt.plot(
#                                 normalize(pooler_results),
#                                 color="darkviolet",
#                                 label="Pooler Confidence",
#                             )
#                             plt.plot(
#                                 normalize(discriminator_results),
#                                 color="lime",
#                                 label="Discriminator Confidence",
#                             )
#                             plt.plot(
#                                 dissipation,
#                                 color="gold",
#                                 label="Dissipation",
#                             )
#                             plt.plot(
#                                 difference,
#                                 color="darkorange",
#                                 label="Difference",
#                             )
#                             # plt.plot(
#                             #     resonance_frequency,
#                             #     color="deeppink",
#                             #     label="Resonance Frequency",
#                             # )

#                             # plt.scatter(
#                             #     peaks,
#                             #     difference[peaks],
#                             #     marker="o",
#                             #     color="darkviolet",
#                             # )
#                             # plt.scatter(
#                             #     peaks,
#                             #     resonance_frequency[peaks],
#                             #     marker="o",
#                             #     color="darkviolet",
#                             # )
#                             print(actual_indices)
#                             plt.axvline(
#                                 x=actual_indices[0],
#                                 color="dodgerblue",
#                                 linestyle="--",
#                                 label="Actual",
#                             )
#                             for index in actual_indices:
#                                 plt.axvline(
#                                     x=index,
#                                     color="dodgerblue",
#                                     linestyle="--",
#                                 )
#                             plt.axvline(
#                                 x=s_bound,
#                                 color="red",
#                             )
#                             plt.axvline(
#                                 x=t_bound,
#                                 color="black",
#                             )
#                             plt.scatter(
#                                 peaks,
#                                 dissipation[peaks],
#                                 label="Predicted",
#                                 marker="o",
#                                 color="red",
#                             )
#                             plot_name = data_file.replace(PATH, "")
#                             plt.xlabel("POIs")
#                             plt.ylabel("Dissipation")
#                             plt.title(f"Predicted/Actual POIs on Data: {plot_name}")

#                             plt.legend()
#                             plt.grid(True)
#                             plt.show()

#                     else:
#                         correct += 1
#                     print(
#                         f"Current error on:\n\t# Correct={correct}\n\t# Incorrect={incorrect}\n\tAccuracy={correct/(incorrect+correct)}"
#                     )

# print(
#     f"Total error on dataset:\n\t# Correct={correct}\n\t# Incorrect={incorrect}\n\tAccuracy={correct/(incorrect+correct)}"
# )
