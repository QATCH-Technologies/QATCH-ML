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
#                                     # print(f"Concatenating {sub_sub_folder_path}...")
#                                     file_data = pd.read_csv(sub_sub_folder_path)
#                                     data_df = pd.concat([data_df, file_data])
###
# qmodel = QModel(data_df)
# qmodel.tune_hyperopt(25)
# qmodel.train()
# qmodel.save()
actual = [2611, 2620, 2624, 2758, 3341, 3976]
pipe = QDataPipeline("VOYAGER_models/W10+QV1862_EL5_L8_3rd.csv")
pipe.compute_difference()
pipe.save_dataframe()
qpreditor = QModelPredict("QModel/SavedModels/QModel_20240618_074619.json")
predictions_raw, peaks = qpreditor.predict("VOYAGER_models/W10+QV1862_EL5_L8_3rd.csv")
print(peaks)
df = pd.read_csv("VOYAGER_models/W10+QV1862_EL5_L8_3rd.csv")


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
    "VOYAGER_models/W10+QV1862_EL5_L8_3rd_poi.csv", header=None
).values
actual_indices = [item[0] for item in actual_indices]
plt.figure(figsize=(8, 6))
plt.plot(normalize(predictions_raw), color="lime", label="Model Confidence")
plt.plot(dissipation, color="gold", label="Dissipation")
plt.plot(difference, color="darkorange", label="Difference")
plt.plot(resonance_frequency, color="deeppink", label="Resonance Frequency")
plt.scatter(
    peaks,
    dissipation[peaks],
    label="Predicted",
    marker="o",
    color="darkviolet",
)
plt.scatter(
    peaks,
    difference[peaks],
    marker="o",
    color="darkviolet",
)
plt.scatter(
    peaks,
    resonance_frequency[peaks],
    marker="o",
    color="darkviolet",
)
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
