import os
import pandas as pd
from QModel import QModel, QModelPredict

PATH = "content/VOYAGER_PROD_DATA"
print(os.listdir(PATH))
data_df = pd.DataFrame()
# For training_data_with_points directory
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
for folder_name in os.listdir(PATH):
    folder_path = os.path.join(PATH, folder_name)

    if os.path.isdir(folder_path):
        # Traverse through the second level directories
        for sub_folder_name in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder_name)

            if os.path.isdir(sub_folder_path):
                if os.path.isdir(sub_folder_path):
                    for sub_sub_folder_name in os.listdir(sub_folder_path):
                        for filename in os.listdir(sub_folder_path):
                            for sub_sub_folder_name in os.listdir(sub_folder_path):
                                sub_sub_folder_path = os.path.join(
                                    sub_folder_path, sub_sub_folder_name
                                )
                                if sub_sub_folder_path.endswith(
                                    ".csv"
                                ) and not sub_sub_folder_path.endswith("_poi.csv"):
                                    print(f"Concatenating {sub_sub_folder_path}...")
                                    file_data = pd.read_csv(sub_sub_folder_path)
                                    data_df = pd.concat([data_df, file_data])
print(
    "POI Detection data -  rows:",
    data_df.shape[0],
    " columns:",
    data_df.shape[1],
)

print(data_df.head())
print(data_df.describe())

total = data_df.isnull().sum().sort_values(ascending=False)
percent = (data_df.isnull().sum() / data_df.isnull().count() * 100).sort_values(
    ascending=False
)
pd.concat([total, percent], axis=1, keys=["Total", "Percent"]).transpose()

temp = data_df["Class"].value_counts()
df = pd.DataFrame({"Class": temp.index, "values": temp.values})
####
qmodel = QModel(data_df)
qmodel.train()
qmodel.tune_params()
