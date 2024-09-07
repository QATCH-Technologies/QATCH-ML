import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from ModelData import ModelData
from QDataPipeline import QDataPipeline
from tqdm import tqdm

from QModel import QModelPredict
from QMultiModel import QPredictor
from QImageClusterer import QClusterer

TEST_BATCH_SIZE = 0.99
VALIDATION_DATASETS_PATH = "content/test_data/test"
S_PREDICTOR = QModelPredict(
    "QModel/SavedModels/QModel_1.json",
    "QModel/SavedModels/QModel_2.json",
    "QModel/SavedModels/QModel_3.json",
    "QModel/SavedModels/QModel_4.json",
    "QModel/SavedModels/QModel_5.json",
    "QModel/SavedModels/QModel_6.json",
)
M_PREDICTOR_0 = QPredictor("QModel/SavedModels/QMultiType_0.json")
M_PREDICTOR_1 = QPredictor("QModel/SavedModels/QMultiType_1.json")
M_PREDICTOR_2 = QPredictor("QModel/SavedModels/QMultiType_2.json")
qcr = QClusterer(model_path="QModel/SavedModels/cluster.joblib")


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
    md_predictor = ModelData()
    md_result = md_predictor.IdentifyPoints(data_path=filename)
    if isinstance(md_result, int):
        md_result = [1, 1, 1, 1, 1, 1]
    predictions = []
    for item in md_result:
        if isinstance(item, list):
            predictions.append(item[0][0])
        else:
            predictions.append(item)
    return list(zip(predictions, act_poi))


def test_mm_on_file(filename, act_poi):
    label = qcr.predict_label(filename)
    if label == 0:
        predictions, candidates = M_PREDICTOR_0.predict(
            filename, type=label, act=act_poi
        )
    elif label == 1:
        predictions, candidates = M_PREDICTOR_1.predict(
            filename, type=label, act=act_poi
        )
    elif label == 2:
        predictions, candidates = M_PREDICTOR_2.predict(
            filename, type=label, act=act_poi
        )
    else:
        raise ValueError(f"Invalid predicted label was: {label}")
    good = []
    bad = []
    initial = 0.01
    post = 0.05
    for i, (x, y) in enumerate(zip(predictions, act_poi)):
        if i < 3:
            if abs(x - y) >= initial * y:
                bad.append((filename, act_poi, predictions, label))
                break
        else:
            if abs(x - y) >= post * y:
                bad.append((filename, act_poi, predictions, label))
                break
    good.append((filename, act_poi, predictions))
    return list(zip(predictions, act_poi)), good, bad


def test_qmp_on_file(filename, act_poi):
    qdp = QDataPipeline(filename)
    time_delta = qdp.find_time_delta()
    # if time_delta == -1:
    qdp.preprocess(poi_filepath=None)

    predictions = S_PREDICTOR.predict(filename)
    return list(zip(predictions, act_poi))
    # else:
    #     print("[INFO] QM Skipping due to time delta")


def compute_deltas(results):
    deltas = []
    if results is not None:
        for prediction, actual in results:
            deltas.append((actual - prediction))
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


def metrics_view(
    model_1, model_2, model_3, test_name, model_1_name, model_2_name, model_3_name, note
):
    points = np.arange(1, 7)

    # Width of each bar
    bar_width = 0.30

    # Set the positions of the bars
    r1 = points - bar_width
    r2 = points
    r3 = points + bar_width
    plt.figure(figsize=(12, 7))
    bars1 = plt.bar(
        r1,
        model_1,
        width=bar_width,
        label=f"{model_1_name} {test_name}",
        color="blue",
        edgecolor="black",
    )
    bars2 = plt.bar(
        r2,
        model_2,
        width=bar_width,
        label=f"{model_2_name} {test_name}",
        color="red",
        edgecolor="black",
    )
    bars3 = plt.bar(
        r3,
        model_3,
        width=bar_width,
        label=f"{model_3_name} {test_name}",
        color="yellow",
        edgecolor="black",
    )

    plt.title(
        f"Comparison of {test_name} Scores for {model_1_name} and {model_2_name}")
    plt.xlabel("POI #")
    plt.ylabel(f"{test_name} Score")
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

    for bar in bars3:
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


def median_absolute_error(predictions, actual):
    # Calculate the absolute errors
    absolute_errors = [abs(p - a) for p, a in zip(predictions, actual)]

    # Return the median of these absolute errors
    return np.median(absolute_errors)


def mean_absolute_percentage_error_without_outliers(predictions, actual):
    percentage_errors = [
        abs((a - p) / a) for p, a in zip(predictions, actual) if a != 0
    ]
    errors_array = np.array(percentage_errors)
    Q1 = np.percentile(errors_array, 25)
    Q3 = np.percentile(errors_array, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_errors = errors_array[
        (errors_array >= lower_bound) & (errors_array <= upper_bound)
    ]
    return np.mean(filtered_errors) * 100 if len(filtered_errors) > 0 else float("nan")


def mean_absolute_error_without_outliers(predictions, actual):
    absolute_errors = [abs(p - a) for p, a in zip(predictions, actual)]
    errors_array = np.array(absolute_errors)

    Q1 = np.percentile(errors_array, 25)
    Q3 = np.percentile(errors_array, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_errors = errors_array[
        (errors_array >= lower_bound) & (errors_array <= upper_bound)
    ]
    return np.mean(filtered_errors) if len(filtered_errors) > 0 else float("nan")


def poi_k_metrics(predictions, actual, k, verbose):
    mae = mean_absolute_error(predictions, actual)
    mse = mean_squared_error(predictions, actual)
    rmse = root_mean_squared_error(predictions, actual)
    r2 = r_squared(predictions, actual)
    mape = mean_absolute_percentage_error_without_outliers(predictions, actual)
    med_ape = mean_absolute_error_without_outliers(predictions, actual)
    if verbose:
        print(f"\n<<POI {k} METRICS>>")
        print("MAE:", mae)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R2:", r2)
        print("MAPE:", mape)
        print("MAE No Outliers:", med_ape)
    return mae, mse, rmse, r2, mape, med_ape


def run():
    VERBOSE = False
    qmp_deltas, md_deltas, mm_deltas = [], [], []
    qmp_list, md_list, mm_list = [], [], []
    content = load_test_dataset(VALIDATION_DATASETS_PATH, TEST_BATCH_SIZE)
    good_list = []
    bad_list = []
    for filename in tqdm(content, desc="<<Running Tests>>"):
        if (
            filename.endswith(".csv")
            and not filename.endswith("_poi.csv")
            and not filename.endswith("_lower.csv")
        ):
            test_file = filename
            poi_file = filename.replace(".csv", "_poi.csv")
            if os.path.exists(poi_file):
                act_poi = pd.read_csv(poi_file, header=None).values
                act_poi = [int(x[0]) for x in act_poi]

                mm_results, good, bad = test_mm_on_file(test_file, act_poi)
                good_list.append(good)
                bad_list.append(bad)
                qmp_results = test_qmp_on_file(test_file, act_poi)
                md_results = test_md_on_file(test_file, act_poi)

                mm_list.append(mm_results)
                qmp_list.append(qmp_results)
                md_list.append(md_results)

                mm_deltas.append(compute_deltas(mm_results))
                qmp_deltas.append(compute_deltas(qmp_results))
                md_deltas.append(compute_deltas(md_results))
    # for bad in bad_list:
    #     print(bad)
    #     if len(bad) > 0:
    #         file = bad[0][0]
    #         act_poi = bad[0][1]
    #         predictions = bad[0][2]
    #         label = bad[0][3]
    #         plt.figure()
    #         df = pd.read_csv(file)
    #         plt.plot(df["Dissipation"].values, label="Dissipation")
    #         for i, poi in enumerate(act_poi):
    #             plt.axvline(x=poi, linestyle="--", color="black", label=f"Actual {i}")
    #         for i, poi in enumerate(predictions):
    #             plt.axvline(x=poi, color="red", label=f"Predicted {i}")
    #         plt.legend()
    #         plt.title(f"Type {label} run")
    #         plt.show()
    over_prediction = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    under_prediction = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    exact = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for delta in mm_deltas:
        for i, d in enumerate(delta):
            if d > 0:
                over_prediction[i + 1] = over_prediction[i + 1] + 1
            if d < 0:
                under_prediction[i + 1] = under_prediction[i + 1] + 1
            if d == 0:
                exact[i + 1] = exact[i + 1] + 1
    print(
        f"[INFO] Prediction Quality:\n\t- Over={over_prediction}\n\t- Under={under_prediction}\n\t- Exact={exact}"
    )
    mm_ppr = extract_results(mm_list)
    qmp_ppr = extract_results(qmp_list)
    md_ppr = extract_results(md_list)
    mm_mae, mm_mse, mm_rmse, mm_r2, mm_mape, mm_med_ape = [], [], [], [], [], []
    qmp_mae, qmp_mse, qmp_rmse, qmp_r2, qmp_mape, qmp_med_ape = [], [], [], [], [], []
    md_mae, md_mse, md_rmse, md_r2, md_mape, md_med_ape = [], [], [], [], [], []
    ############################################################
    # MM
    ############################################################
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        mm_ppr[0]["predicted"], mm_ppr[0]["actual"], 1, VERBOSE
    )
    mm_mae.append(mae)
    mm_mse.append(mse)
    mm_rmse.append(rmse)
    mm_r2.append(r2)
    mm_mape.append(mape)
    mm_med_ape.append(med_ape)
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        mm_ppr[1]["predicted"], mm_ppr[1]["actual"], 2, VERBOSE
    )
    mm_mae.append(mae)
    mm_mse.append(mse)
    mm_rmse.append(rmse)
    mm_r2.append(r2)
    mm_mape.append(mape)
    mm_med_ape.append(med_ape)
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        mm_ppr[2]["predicted"], mm_ppr[2]["actual"], 3, VERBOSE
    )
    mm_mae.append(mae)
    mm_mse.append(mse)
    mm_rmse.append(rmse)
    mm_r2.append(r2)
    mm_mape.append(mape)
    mm_med_ape.append(med_ape)
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        mm_ppr[3]["predicted"], mm_ppr[3]["actual"], 4, VERBOSE
    )
    mm_mae.append(mae)
    mm_mse.append(mse)
    mm_rmse.append(rmse)
    mm_r2.append(r2)
    mm_mape.append(mape)
    mm_med_ape.append(med_ape)
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        mm_ppr[4]["predicted"], mm_ppr[4]["actual"], 5, VERBOSE
    )
    mm_mae.append(mae)
    mm_mse.append(mse)
    mm_rmse.append(rmse)
    mm_r2.append(r2)
    mm_mape.append(mape)
    mm_med_ape.append(med_ape)
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        mm_ppr[5]["predicted"], mm_ppr[5]["actual"], 6, VERBOSE
    )
    mm_mae.append(mae)
    mm_mse.append(mse)
    mm_rmse.append(rmse)
    mm_r2.append(r2)
    mm_mape.append(mape)
    mm_med_ape.append(med_ape)
    ############################################################
    # MD
    ############################################################
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        qmp_ppr[0]["predicted"], qmp_ppr[0]["actual"], 1, VERBOSE
    )
    qmp_mae.append(mae)
    qmp_mse.append(mse)
    qmp_rmse.append(rmse)
    qmp_r2.append(r2)
    qmp_mape.append(mape)
    qmp_med_ape.append(med_ape)
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        qmp_ppr[1]["predicted"], qmp_ppr[1]["actual"], 2, VERBOSE
    )
    qmp_mae.append(mae)
    qmp_mse.append(mse)
    qmp_rmse.append(rmse)
    qmp_r2.append(r2)
    qmp_mape.append(mape)
    qmp_med_ape.append(med_ape)

    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        qmp_ppr[2]["predicted"], qmp_ppr[2]["actual"], 3, VERBOSE
    )
    qmp_mae.append(mae)
    qmp_mse.append(mse)
    qmp_rmse.append(rmse)
    qmp_r2.append(r2)
    qmp_mape.append(mape)
    qmp_med_ape.append(med_ape)
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        qmp_ppr[3]["predicted"], qmp_ppr[3]["actual"], 4, VERBOSE
    )
    qmp_mae.append(mae)
    qmp_mse.append(mse)
    qmp_rmse.append(rmse)
    qmp_r2.append(r2)
    qmp_mape.append(mape)
    qmp_med_ape.append(med_ape)

    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        qmp_ppr[4]["predicted"], qmp_ppr[4]["actual"], 5, VERBOSE
    )
    qmp_mae.append(mae)
    qmp_mse.append(mse)
    qmp_rmse.append(rmse)
    qmp_r2.append(r2)
    qmp_mape.append(mape)
    qmp_med_ape.append(med_ape)
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        qmp_ppr[5]["predicted"], qmp_ppr[5]["actual"], 6, VERBOSE
    )
    qmp_mae.append(mae)
    qmp_mse.append(mse)
    qmp_rmse.append(rmse)
    qmp_r2.append(r2)
    qmp_mape.append(mape)
    qmp_med_ape.append(med_ape)

    ############################################################
    # MD
    ############################################################
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        md_ppr[0]["predicted"], md_ppr[0]["actual"], 1, VERBOSE
    )
    md_mae.append(mae)
    md_mse.append(mse)
    md_rmse.append(rmse)
    md_r2.append(r2)
    md_mape.append(mape)
    md_med_ape.append(med_ape)

    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        md_ppr[1]["predicted"], md_ppr[1]["actual"], 2, VERBOSE
    )
    md_mae.append(mae)
    md_mse.append(mse)
    md_rmse.append(rmse)
    md_r2.append(r2)
    md_mape.append(mape)
    md_med_ape.append(med_ape)

    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        md_ppr[2]["predicted"], md_ppr[2]["actual"], 3, VERBOSE
    )
    md_mae.append(mae)
    md_mse.append(mse)
    md_rmse.append(rmse)
    md_r2.append(r2)
    md_mape.append(mape)
    md_med_ape.append(med_ape)

    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        md_ppr[2]["predicted"], md_ppr[3]["actual"], 4, VERBOSE
    )
    md_mae.append(mae)
    md_mse.append(mse)
    md_rmse.append(rmse)
    md_r2.append(r2)
    md_mape.append(mape)
    md_med_ape.append(med_ape)
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        md_ppr[4]["predicted"], md_ppr[4]["actual"], 5, VERBOSE
    )
    md_mae.append(mae)
    md_mse.append(mse)
    md_rmse.append(rmse)
    md_r2.append(r2)
    md_mape.append(mape)
    md_med_ape.append(med_ape)
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        md_ppr[5]["predicted"], md_ppr[5]["actual"], 6, VERBOSE
    )
    md_mae.append(mae)
    md_mse.append(mse)
    md_rmse.append(rmse)
    md_r2.append(r2)
    md_mape.append(mape)
    md_med_ape.append(med_ape)

    metrics_view(
        mm_mae,
        qmp_mae,
        md_mae,
        "MAE",
        "QMultiModel",
        "QSingleModel",
        "ModelData",
        "Average magnitude of errors between\npredicted/actual ignoring direction.\n(Lower is better)",
    )
    metrics_view(
        mm_mse,
        qmp_mse,
        md_mse,
        "MSE",
        "QMultiModel",
        "QSingleModel",
        "ModelData",
        "Average of squared differences between predicted/actual\nvalues emphasizes larger errors.\n(Lower is better)",
    )
    metrics_view(
        mm_rmse,
        qmp_rmse,
        md_rmse,
        "RMSE",
        "QMultiModel",
        "QSingleModel",
        "ModelData",
        "Sqrt of MSE.\nUnits are the same as target variable.\n(Lower is better)",
    )
    metrics_view(
        mm_r2,
        qmp_r2,
        md_r2,
        "R^2",
        "QMultiModel",
        "QSingleModel",
        "ModelData",
        "How well the model 'fits' the data.\n(Bigger is better)",
    )
    metrics_view(
        mm_mape,
        qmp_mape,
        md_mape,
        "MAPE",
        "QMultiModel",
        "QSingleModel",
        "ModelData",
        "Average magnitude of errors as a percentage\nof the actual values.\n(Lower is better)",
    )

    metrics_view(
        mm_med_ape,
        qmp_med_ape,
        md_med_ape,
        "Median Absolute Error",
        "QMultiModel",
        "QSingleModel",
        "ModelData",
        "MAE with no Outliers",
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
