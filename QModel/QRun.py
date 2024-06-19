import os
import pandas as pd
from QModel import QModel, QModelPredict
from QDataPipline import QDataPipeline
import QDataVisualizer as qdv
import matplotlib.pyplot as plt
import numpy as np

PATH = "content/VOYAGER_PROD_DATA"
print(os.listdir(PATH))
data_df = pd.DataFrame()
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
###
# qModel = QModel(data_df)
# qModel.tune("p", 10)
# qModel.tune("d", 10)
# qModel.train_pooler()
# qModel.train_discriminator()
# qModel.save_pooler()
# qModel.save_discriminator()
# pipe = QDataPipeline("VOYAGER_models/W10+QV1862_EL5_L8_3rd.csv")
# pipe.compute_difference()
# pipe.save_dataframe()
qpreditor = QModelPredict(
    pooler_path="QModel/SavedModels/QModelPooler.json",
    discriminator_path="QModel/SavedModels/QModelDiscriminator.json",
)
pooler_results, discriminator_results, peaks = qpreditor.predict(
    "content/VOYAGER_PROD_DATA/VOYAGER_3RD BATCH_W7/MM231106W7_FV699_25X_D11/MM231106W7_FV699_25X_D11_3rd.csv"
)
print(peaks)
df = pd.read_csv(
    "content/VOYAGER_PROD_DATA/VOYAGER_3RD BATCH_W7/MM231106W7_FV699_25X_D11/MM231106W7_FV699_25X_D11_3rd.csv"
)


# Extract columns from the DataFrame
def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


dissipation = df["Dissipation"]
difference = df["Difference"]
difference = np.abs(difference)
resonance_frequency = df["Resonance_Frequency"]
difference = normalize(difference)
resonance_frequency = normalize(resonance_frequency)
dissipation = normalize(dissipation)
actual_indices = pd.read_csv(
    "content/VOYAGER_PROD_DATA/VOYAGER_3RD BATCH_W7/MM231106W7_FV699_25X_D11/MM231106W7_FV699_25X_D11_3rd_poi.csv",
    header=None,
).values
actual_indices = [item[0] for item in actual_indices]
plt.figure(figsize=(8, 6))
# plt.plot(normalize(pooler_results), color="darkviolet", label="Pooler Confidence")
plt.plot(
    normalize(discriminator_results), color="lime", label="Discriminator Confidence"
)
plt.plot(dissipation, color="gold", label="Dissipation")
# plt.plot(difference, color="darkorange", label="Difference")
# plt.plot(resonance_frequency, color="deeppink", label="Resonance Frequency")
plt.scatter(
    peaks,
    dissipation[peaks],
    label="Predicted",
    marker="o",
    color="darkviolet",
)
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
plt.axvline(x=actual_indices[0], color="dodgerblue", linestyle="--", label="Actual")
for index in actual_indices:
    plt.axvline(x=index, color="dodgerblue", linestyle="--")

plt.xlabel("POIs")
plt.ylabel("Dissipation")
plt.title("Predicted/Actual POIs on Data")
plt.legend()
plt.grid(True)
plt.show()
