import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ModelData import ModelData
from QDataPipline import QDataPipeline
from tqdm import tqdm

from QModel import QModelPredict

TEST_BATCH_SIZE = 0.2
VALIDATION_DATASETS_PATH = "QModel/validation_datasets"
PREDICTOR = QModelPredict(
    "QModel/SavedModels/QModel_1.json",
    "QModel/SavedModels/QModel_2.json",
    "QModel/SavedModels/QModel_3.json",
    "QModel/SavedModels/QModel_4.json",
    "QModel/SavedModels/QModel_5.json",
    "QModel/SavedModels/QModel_6.json",
)


def load_test_dataset(path, test_size):
    content = []
    for root, dirs, files in os.walk(path):
        for file in files:
            content.append(os.path.join(root, file))

    num_files_to_select = int(len(content) * test_size)

    if num_files_to_select == 0 and len(content) > 0:
        num_files_to_select = 1

    if num_files_to_select > len(content):
        return content

    return random.sample(content, num_files_to_select)


def test_md_on_file(filename, act_poi):
    md_predictor = ModelData()
    md_result = md_predictor.IdentifyPoints(data_path=filename)
    predictions = []
    for item in md_result:
        if isinstance(item, list):
            predictions.append(item[0][0])
        else:
            predictions.append(item)
    return list(zip(predictions, act_poi))


def test_qmp_on_file(filename, act_poi):
    qdp = QDataPipeline(filename)
    qdp.preprocess(poi_file=None)

    predictions = PREDICTOR.predict(filename)
    return list(zip(predictions, act_poi))


def compute_deltas(results):
    deltas = [abs((actual - prediction)) for prediction, actual in results]
    return deltas


def accuracy_scatter_view(results, name):
    # Define colors for each index
    colormap = plt.get_cmap("viridis")

    # Create a figure and axis

    # Collect all data for each index position
    index_data = {i: {"predicted": [], "actual": []} for i in range(6)}

    # Gather data from all datasets
    for dataset in results:
        for i, (pred, actual) in enumerate(dataset):
            index_data[i]["predicted"].append(pred)
            index_data[i]["actual"].append(actual)

    # Plot the predicted data with the corresponding color for each index
    for i in range(6):
        color = colormap(i / 5)  # Normalize index to range [0, 1]
        if index_data[i]["predicted"]:
            fig, ax = plt.subplots()
            ax.scatter(
                index_data[i]["predicted"],
                index_data[i]["actual"],
                color=color,
                label=f"POI {i+1} Predicted",
                marker="o",
                alpha=0.7,
            )
            # Add a line representing perfect predictions (y = x)
            lims = [
                min(min(pred for pred, _ in dataset) for dataset in results),
                max(max(pred for pred, _ in dataset) for dataset in results),
            ]

            ax.plot(
                lims,
                lims,
                linestyle="dotted",
                label="Perfect Predictions",
                color="grey",
            )  # 'k--' for black dashed line

            # Add labels and title
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Actual Values")
            ax.set_title(f"{name} Predicted/Actual Values POI={i + 1}")
            ax.legend()

            # Show the plot
            plt.show()
    # Calculate average deviance for each index
    average_deviance = {}
    for i in range(6):
        if index_data[i]["predicted"]:
            deviance = [
                abs(pred - actual)
                for pred, actual in zip(
                    index_data[i]["predicted"], index_data[i]["actual"]
                )
            ]
            average_deviance[i] = np.mean(deviance)

    # Print average deviance
    print("Average Deviance for each index:")
    for i, deviance in average_deviance.items():
        print(f"Index {i}: {deviance:.2f}")


def delta_distribution_view(deltas, name):
    deltas_np = np.array(deltas)
    point_counts = np.zeros((len(deltas), deltas_np.shape[1]))

    for i in range(deltas_np.shape[1]):
        unique, counts = np.unique(deltas_np[:, i], return_counts=True)
        count_dict = dict(zip(unique, counts))
        point_counts[:, i] = [count_dict.get(x, 0) for x in deltas_np[:, i]]

    fig, ax = plt.subplots(figsize=(10, 7))

    bottom_values = np.zeros(len(deltas))
    for i in range(deltas_np.shape[1]):
        ax.bar(
            range(len(deltas)),
            point_counts[:, i],
            bottom=bottom_values,
            label=f"Point {i}",
        )
        bottom_values += point_counts[:, i]

    ax.set_xlabel("Delta from actual")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of prediction delta from actual {name}")
    ax.legend(title="Points")

    plt.tight_layout()
    plt.show()


def run():
    qmp_deltas, md_deltas = [], []
    qmp_list, md_list = [], []
    content = load_test_dataset(VALIDATION_DATASETS_PATH, TEST_BATCH_SIZE)
    for filename in tqdm(content, desc="<<Running Tests>>"):
        if (
            filename.endswith(".csv")
            and not filename.endswith("_poi.csv")
            and not filename.endswith("_lower.csv")
        ):
            test_file = filename
            poi_file = filename.replace(".csv", "_poi.csv")
            act_poi = pd.read_csv(poi_file, header=None).values
            act_poi = [int(x[0]) for x in act_poi]
            qmp_results = test_qmp_on_file(test_file, act_poi)
            md_results = test_md_on_file(test_file, act_poi)

            qmp_list.append(qmp_results)
            md_list.append(md_results)
            qmp_deltas.append(compute_deltas(qmp_results))
            md_deltas.append(compute_deltas(md_results))
    accuracy_scatter_view(qmp_list, "QModel")
    accuracy_scatter_view(md_list, "ModelData")
    # delta_distribution_view(qmp_deltas, "QModel")
    # delta_distribution_view(md_deltas, "ModelData")


if __name__ == "__main__":
    run()
