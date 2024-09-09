import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qmodel.q_data_pipeline import QDataPipeline
from tqdm import tqdm
import pickle


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


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def pipeline(train_content):
    print(f"[INFO] Preprocessing on {len(train_content)} datasets")
    dataset = []
    for filename in tqdm(train_content, desc="<<Processing Datasets>>"):
        if (
            filename.endswith(".csv")
            and not filename.endswith("_poi.csv")
            and not filename.endswith("_lower.csv")
        ):
            data_file = filename
            qdp = QDataPipeline(data_file, multi_class=True)
            poi_file = filename.replace(".csv", "_poi.csv")
            if not os.path.exists(poi_file):
                continue
            qdp.preprocess(poi_filepath=poi_file)
            has_nan = qdp.__dataframe__.isna().any().any()
            if not has_nan:
                i = int(qdp.__dataframe__.index[qdp.__dataframe__["Class"] == 1][0]) - 1
                j = int(qdp.__dataframe__.index[qdp.__dataframe__["Class"] == 6][0]) + 1
                qdp.__dataframe__ = qdp.__dataframe__[i:j]
                qdp.__dataframe__["Relative_time"] = normalize(
                    qdp.__dataframe__["Relative_time"]
                )
                dataset.append(qdp.get_dataframe())
    return dataset


def get_band_gap(dataset, bandgap_percentage=0.95):
    # Combine all dataframes into one
    combined_df = pd.concat(dataset)

    # Filter rows where 'Class' is 1 through 6
    class_dfs = [combined_df[combined_df["Class"] == i] for i in range(1, 7)]

    # Initialize a dictionary to hold quartile information
    bandgap_info = {}

    plt.figure(figsize=(12, 8))

    # Plot histogram and bandgaps for each class
    for i, class_df in enumerate(class_dfs, start=1):
        if class_df.empty:
            print(f"No data for Class_{i}. Skipping.")
            bandgap_info[f"Class_{i}"] = {
                "lower_quantile": None,
                "upper_quantile": None,
            }
            continue

        counts, bin_edges = np.histogram(class_df["Relative_time"], bins="auto")

        # plt.hist(
        #     class_df["Relative_time"],
        #     bins=bin_edges,
        #     alpha=0.5,
        #     label=f"Class_{i} Histogram",
        # )

        lower_quantile = np.percentile(
            class_df["Relative_time"], (1 - bandgap_percentage) / 2 * 100
        )
        upper_quantile = np.percentile(
            class_df["Relative_time"], (1 + bandgap_percentage) / 2 * 100
        )

        # plt.fill_betweenx(
        #     [0, max(counts) if len(counts) > 0 else 1],
        #     lower_quantile,
        #     upper_quantile,
        #     color=f"C{i-1}",
        #     alpha=0.3,
        #     label=f"Class_{i} Bandgap ({int(bandgap_percentage*100)}%)",
        # )

        bandgap_info[f"Class_{i}"] = {
            "lq": lower_quantile,
            "uq": upper_quantile,
        }

    # plt.xlabel("Time")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Class Occurrences Over Time with Bandgaps")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    return bandgap_info


if __name__ == "__main__":
    content = load_content("content/label_0")
    dataset = pipeline(content)
    gap = get_band_gap(dataset)
    with open("QModel/SavedModels/label_0.pkl", "wb") as file:
        pickle.dump(gap, file)

    content = load_content("content/label_1")
    dataset = pipeline(content)
    gap = get_band_gap(dataset)
    with open("QModel/SavedModels/label_1.pkl", "wb") as file:
        pickle.dump(gap, file)

    content = load_content("content/label_2")
    dataset = pipeline(content)
    gap = get_band_gap(dataset)
    with open("QModel/SavedModels/label_2.pkl", "wb") as file:
        pickle.dump(gap, file)
