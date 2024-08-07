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

TEST_BATCH_SIZE = 0.95
VALIDATION_DATASETS_PATH = "content/bad_runs/validate"
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
    qdp = QDataPipeline(filename)
    time_delta = qdp.find_time_delta()
    if time_delta == -1:
        md_predictor = ModelData()
        md_result = md_predictor.IdentifyPoints(data_path=filename)
        predictions = []
        for item in md_result:
            if isinstance(item, list):
                predictions.append(item[0][0])
            else:
                predictions.append(item)
        return list(zip(predictions, act_poi))
    else:
        print("[INFO] MD Skipping due to time delta")


def test_qmp_on_file(filename, act_poi):
    qdp = QDataPipeline(filename)
    time_delta = qdp.find_time_delta()
    if time_delta == -1:
        qdp.preprocess(poi_file=None)

        predictions = PREDICTOR.predict(filename)
        return list(zip(predictions, act_poi))
    else:
        print("[INFO] QM Skipping due to time delta")


def compute_deltas(results):
    deltas = []
    if results is not None:
        for prediction, actual in results:
            deltas.append(abs((actual - prediction)))
    return deltas


def extract_results(results):
    # Collect all data for each index position
    index_data = {i: {"predicted": [], "actual": []} for i in range(6)}

    # Gather data from all datasets
    for dataset in results:
        if dataset is not None:
            for i, (pred, actual) in enumerate(dataset):
                index_data[i]["predicted"].append(pred)
                index_data[i]["actual"].append(actual)
    return index_data


def accuracy_scatter_view(results, name):
    # Define colors for each index
    colormap = plt.get_cmap("viridis")

    # Create a figure and axis
    index_data = extract_results(results)
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
                min(
                    min(pred for pred, _ in dataset)
                    for dataset in results
                    if dataset is not None
                ),
                max(
                    max(pred for pred, _ in dataset)
                    for dataset in results
                    if dataset is not None
                ),
            ]

            ax.plot(
                lims,
                lims,
                linestyle="dotted",
                label="Perfect Predictions",
                color="grey",
            )

            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Actual Values")
            ax.set_title(f"{name} Predicted/Actual Values POI={i + 1}")
            ax.legend()
            plt.show()
    # Calculate average deviance for each index
    average_deviance = {}
    std = {}
    rng = {}
    for i in range(6):
        if index_data[i]["predicted"]:
            deviance = [
                (abs(pred - actual))
                for pred, actual in zip(
                    index_data[i]["predicted"],
                    index_data[i]["actual"],
                )
            ]
            average_deviance[i] = np.mean(deviance)
            std[i] = np.std(deviance)
            rng[i] = max(deviance) - min(deviance)

    # Print average deviance
    print(f"{name} Average Deviance/STD for each index:")
    for deviance, standard_deviation, gaps in zip(
        average_deviance.items(), std.items(), rng.items()
    ):
        print(f"POI {deviance[0]}:")
        print(f" - Average % Error {float(deviance[1])}")
        print(f" - STD {float(standard_deviation[1])}")
        print(f" - Maximum Range {gaps[1]}")
        # print(
        #     f"POI {1}:\n\t - Deviance={deviance:.2f}\n\t - STD={standard_deviation}\n\t - Range={gaps}"
        # )
    print()


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


def metrics_view(qmp_metric, md_metric, name, note):
    points = np.arange(1, 7)

    # Width of each bar
    bar_width = 0.35

    # Set the positions of the bars
    r1 = points - bar_width / 2
    r2 = points + bar_width / 2
    plt.figure(figsize=(12, 7))
    bars1 = plt.bar(
        r1,
        qmp_metric,
        width=bar_width,
        label=f"QModel {name}",
        color="blue",
        edgecolor="black",
    )
    bars2 = plt.bar(
        r2,
        md_metric,
        width=bar_width,
        label=f"ModelData {name}",
        color="red",
        edgecolor="black",
    )

    plt.title(f"Comparison of {name} Scores for QModel and ModelData")
    plt.xlabel("POI #")
    plt.ylabel(f"{name} Score")
    plt.xticks(points)  # Set x-ticks to be the point indices
    plt.legend()
    plt.grid(axis="y")
    for bar in bars1:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

    for bar in bars2:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )
    plt.annotate(
        note,
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
    )
    plt.show()


def mean_absolute_error(predictions, actual):
    return sum(abs(p - a) for p, a in zip(predictions, actual)) / len(actual)


def mean_squared_error(predictions, actual):
    return sum((p - a) ** 2 for p, a in zip(predictions, actual)) / len(actual)


def root_mean_squared_error(predictions, actual):
    return (sum((p - a) ** 2 for p, a in zip(predictions, actual)) / len(actual)) ** 0.5


def r_squared(predictions, actual):
    mean_actual = sum(actual) / len(actual)
    ss_total = sum((a - mean_actual) ** 2 for a in actual)
    ss_residual = sum((a - p) ** 2 for p, a in zip(predictions, actual))
    return 1 - (ss_residual / ss_total)


def mean_absolute_percentage_error(predictions, actual):
    return (
        sum(abs((a - p) / a) for p, a in zip(predictions, actual) if a != 0)
        / len(actual)
        * 100
    )


def poi_k_metrics(predictions, actual, k, verbose):
    mae = mean_absolute_error(predictions, actual)
    mse = mean_squared_error(predictions, actual)
    rmse = root_mean_squared_error(predictions, actual)
    r2 = r_squared(predictions, actual)
    mape = mean_absolute_percentage_error(predictions, actual)
    if verbose:
        print(f"\n<<POI {k} METRICS>>")
        print("MAE:", mae)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R2:", r2)
        print("MAPE:", mape)
    return mae, mse, rmse, r2, mape


def run():
    VERBOSE = False
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
    ############################################################
    # MD
    ############################################################
    qmp_ppr = extract_results(qmp_list)
    md_ppr = extract_results(md_list)
    qmp_mae, qmp_mse, qmp_rmse, qmp_r2, qmp_mape = [], [], [], [], []
    md_mae, md_mse, md_rmse, md_r2, md_mape = [], [], [], [], []
    mae, mse, rmse, r2, mape = poi_k_metrics(
        qmp_ppr[0]["predicted"], qmp_ppr[0]["actual"], 1, VERBOSE
    )
    qmp_mae.append(mae)
    qmp_mse.append(mse)
    qmp_rmse.append(rmse)
    qmp_r2.append(r2)
    qmp_mape.append(mape)
    mae, mse, rmse, r2, mape = poi_k_metrics(
        qmp_ppr[1]["predicted"], qmp_ppr[1]["actual"], 2, VERBOSE
    )
    qmp_mae.append(mae)
    qmp_mse.append(mse)
    qmp_rmse.append(rmse)
    qmp_r2.append(r2)
    qmp_mape.append(mape)
    mae, mse, rmse, r2, mape = poi_k_metrics(
        qmp_ppr[2]["predicted"], qmp_ppr[2]["actual"], 3, VERBOSE
    )
    qmp_mae.append(mae)
    qmp_mse.append(mse)
    qmp_rmse.append(rmse)
    qmp_r2.append(r2)
    qmp_mape.append(mape)
    mae, mse, rmse, r2, mape = poi_k_metrics(
        qmp_ppr[3]["predicted"], qmp_ppr[3]["actual"], 4, VERBOSE
    )
    qmp_mae.append(mae)
    qmp_mse.append(mse)
    qmp_rmse.append(rmse)
    qmp_r2.append(r2)
    qmp_mape.append(mape)
    mae, mse, rmse, r2, mape = poi_k_metrics(
        qmp_ppr[4]["predicted"], qmp_ppr[4]["actual"], 5, VERBOSE
    )
    qmp_mae.append(mae)
    qmp_mse.append(mse)
    qmp_rmse.append(rmse)
    qmp_r2.append(r2)
    qmp_mape.append(mape)
    mae, mse, rmse, r2, mape = poi_k_metrics(
        qmp_ppr[5]["predicted"], qmp_ppr[5]["actual"], 6, VERBOSE
    )
    qmp_mae.append(mae)
    qmp_mse.append(mse)
    qmp_rmse.append(rmse)
    qmp_r2.append(r2)
    qmp_mape.append(mape)
    ############################################################
    # MD
    ############################################################
    mae, mse, rmse, r2, mape = poi_k_metrics(
        md_ppr[0]["predicted"], md_ppr[0]["actual"], 1, VERBOSE
    )
    md_mae.append(mae)
    md_mse.append(mse)
    md_rmse.append(rmse)
    md_r2.append(r2)
    md_mape.append(mape)
    mae, mse, rmse, r2, mape = poi_k_metrics(
        md_ppr[1]["predicted"], md_ppr[1]["actual"], 2, VERBOSE
    )
    md_mae.append(mae)
    md_mse.append(mse)
    md_rmse.append(rmse)
    md_r2.append(r2)
    md_mape.append(mape)
    mae, mse, rmse, r2, mape = poi_k_metrics(
        md_ppr[2]["predicted"], md_ppr[2]["actual"], 3, VERBOSE
    )
    md_mae.append(mae)
    md_mse.append(mse)
    md_rmse.append(rmse)
    md_r2.append(r2)
    md_mape.append(mape)
    mae, mse, rmse, r2, mape = poi_k_metrics(
        md_ppr[2]["predicted"], md_ppr[3]["actual"], 4, VERBOSE
    )
    md_mae.append(mae)
    md_mse.append(mse)
    md_rmse.append(rmse)
    md_r2.append(r2)
    md_mape.append(mape)
    mae, mse, rmse, r2, mape = poi_k_metrics(
        md_ppr[4]["predicted"], md_ppr[4]["actual"], 5, VERBOSE
    )
    md_mae.append(mae)
    md_mse.append(mse)
    md_rmse.append(rmse)
    md_r2.append(r2)
    md_mape.append(mape)
    mae, mse, rmse, r2, mape = poi_k_metrics(
        md_ppr[5]["predicted"], md_ppr[5]["actual"], 6, VERBOSE
    )
    md_mae.append(mae)
    md_mse.append(mse)
    md_rmse.append(rmse)
    md_r2.append(r2)
    md_mape.append(mape)
    metrics_view(
        qmp_mae,
        md_mae,
        "MAE",
        "Average magnitude of errors between\npredicted/actual ignoring direction.\n(Lower is better)",
    )
    metrics_view(
        qmp_mse,
        md_mse,
        "MSE",
        "Average of squared differences between predicted/actual\nvalues emphasizes larger errors.\n(Lower is better)",
    )
    metrics_view(
        qmp_rmse,
        md_rmse,
        "RMSE",
        "Sqrt of MSE.\nUnits are the same as target variable.\n(Lower is better)",
    )
    metrics_view(
        qmp_r2, md_r2, "R^2", "How well the model 'fits' the data.\n(Bigger is better)"
    )
    metrics_view(
        qmp_mape,
        md_mape,
        "MAPE",
        "Average magnitude of errors as a percentage\nof the actual values.\n(Lower is better)",
    )
    # print(qmp_list)
    # print("MAE:", mean_absolute_error(qmp_list))
    # print("MSE:", mean_squared_error(qmp_list))
    # print("RMSE:", root_mean_squared_error(qmp_list))
    # print("R2:", r_squared(qmp_list))
    # print("MAPE:", mean_absolute_percentage_error(qmp_list))
    # print("----------")
    # print("MAE:", mean_absolute_error(md_list))
    # print("MSE:", mean_squared_error(md_list))
    # print("RMSE:", root_mean_squared_error(md_list))
    # print("R2:", r_squared(md_list))
    # print("MAPE:", mean_absolute_percentage_error(md_list))
    # accuracy_scatter_view(qmp_list, "QModel")
    # accuracy_scatter_view(md_list, "ModelData")
    # delta_distribution_view(qmp_deltas, "QModel")
    # delta_distribution_view(md_deltas, "ModelData")


if __name__ == "__main__":
    run()
