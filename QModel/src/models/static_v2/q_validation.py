import os
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from ModelData import ModelData
from q_data_pipeline import QDataPipeline
from tqdm import tqdm

from collections import defaultdict
from q_model_predictor import QModelPredictor
from q_model import QModelPredict
from q_multi_model import QPredictor
from q_image_clusterer import QClusterer

TEST_BATCH_SIZE = 0.50
VALIDATION_DATASETS_PATH = "content/static/test"
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
QMODEL_V2_PREDICTOR = QModelPredictor(booster_path=os.path.join(
    "QModel", "SavedModels", "qmodel_v2", "qmodel_v2.json"), scaler_path=os.path.join(
        "QModel", "SavedModels", "qmodel_v2", "qmodel_scaler.pkl"))
qcr = QClusterer(model_path="QModel/SavedModels/cluster.joblib")


def load_test_dataset(path, test_size):
    content = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # if "10CP".lower() in file.lower():
            #     content.append(os.path.join(root, file))
            content.append(os.path.join(root, file))

    num_files_to_select = int(len(content) * test_size)

    if num_files_to_select == 0 and len(content) > 0:
        num_files_to_select = 1

    if num_files_to_select > len(content):
        return content

    return random.sample(content, num_files_to_select)


def test_md_on_file(filename, act_poi):
    qdp = QDataPipeline(data_filepath=filename)
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
    try:
        if label == 0:
            candidates = M_PREDICTOR_0.predict(
                filename, run_type=label, act=act_poi)
        elif label == 1:
            try:
                candidates = M_PREDICTOR_1.predict(
                    filename, run_type=label, act=act_poi)
            except:
                candidates = M_PREDICTOR_0.predict(
                    filename, run_type=label, act=act_poi)
        elif label == 2:
            try:
                candidates = M_PREDICTOR_2.predict(
                    filename, run_type=label, act=act_poi)
            except:
                candidates = M_PREDICTOR_0.predict(
                    filename, run_type=label, act=act_poi)
        else:
            raise ValueError(f"Invalid predicted label was: {label}")
    except:
        candidates = [[1], [2], [3], [4], [5], [6]]
    good = []
    bad = []
    initial = 0.01
    post = 0.05
    pois = []
    for c in candidates:
        pois.append(c[0][0])

    for i, (x, y) in enumerate(zip(pois, act_poi)):
        if i < 3:
            if abs(x - y) >= initial * y:
                bad.append((filename, act_poi, pois, label))
                break
        else:
            if abs(x - y) >= post * y:
                bad.append((filename, act_poi, pois, label))
                break
    good.append((filename, act_poi, pois))
    return list(zip(pois, act_poi)), good, bad


def test_qmodel_v2(filename, act_poi):
    candidates = QMODEL_V2_PREDICTOR.predict(filename)
    poi_1 = candidates.get("POI1").get("indices")[0]
    poi_2 = candidates.get("POI2").get("indices")[0]
    poi_3 = candidates.get("POI3").get("indices")[0]
    poi_4 = candidates.get("POI4").get("indices")[0]
    poi_5 = candidates.get("POI5").get("indices")[0]
    poi_6 = candidates.get("POI6").get("indices")[0]
    pois = [poi_1, poi_2, poi_3, poi_4, poi_5, poi_6]
    return list(zip(pois, act_poi))


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
    model_1, model_2, model_3, model_4, test_name,
    model_1_name, model_2_name, model_3_name, model_4_name, note
):
    points = np.arange(1, 7)

    # Width of each bar
    bar_width = 0.30

    # Evenly distribute 4 bars centered on each point.
    r1 = points - 1.5 * bar_width
    r2 = points - 0.5 * bar_width
    r3 = points + 0.5 * bar_width
    r4 = points + 1.5 * bar_width

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
    bars4 = plt.bar(
        r4,
        model_4,
        width=bar_width,
        label=f"{model_4_name} {test_name}",
        color="green",  # Changed to a distinct color for model_4
        edgecolor="black",
    )

    plt.title(
        f"Comparison of {test_name} Scores for {model_1_name} and {model_2_name}"
    )
    plt.xlabel("POI #")
    plt.ylabel(f"{test_name} Score")
    plt.xticks(points)  # Set x-ticks to be the point indices
    plt.legend()
    plt.grid(axis="y")

    # Add text labels above each bar.
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
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
    qmp_deltas, md_deltas, mm_deltas, qmodel_v2_deltas = [], [], [], []
    qmp_list, md_list, mm_list, qmodel_v2_list = [], [], [], []
    content = load_test_dataset(VALIDATION_DATASETS_PATH, TEST_BATCH_SIZE)
    good_list = []
    bad_list = []
    longest_run = (-1, "")
    partial_fills = []
    long_runs = []
    out_of_order = 0
    count = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
    }
    ratios = {3: [], 4: [], 5: []}

    def compute_relative_time_ratios(r_time_poi):
        T = r_time_poi[1] - r_time_poi[0]
        relative_times = np.array([r_time_poi[3] - r_time_poi[0],
                                   r_time_poi[4] - r_time_poi[0],
                                   r_time_poi[5] - r_time_poi[0]])
        scaled_ratios = relative_times / T
        return T, scaled_ratios

    for filename in tqdm(content, desc="<<Running Tests>>"):
        if (
            filename.endswith(".csv")
            and not filename.endswith("_poi.csv")
            and not filename.endswith("_lower.csv")
        ):
            test_file = filename
            poi_file = filename.replace(".csv", "_poi.csv")

            if os.path.exists(poi_file):
                test_df = pd.read_csv(test_file)
                max_index = test_df.index.max()
                act_poi = pd.read_csv(poi_file, header=None).values
                act_poi = [int(x[0]) for x in act_poi]

                if max(act_poi) >= max_index - 1:
                    partial_fills.append(test_file)
                    continue

                r_time = test_df["Relative_time"].values
                r_time_poi = r_time[act_poi]

                # Ensure there are at least 6 points
                if len(r_time_poi) >= 6:
                    T, scaled_ratios = compute_relative_time_ratios(r_time_poi)

                    # Store computed ratios
                    for i, index in enumerate([3, 4, 5]):
                        ratios[index].append(scaled_ratios[i])

                test_qdp = QDataPipeline(data_filepath=test_file)
                if (
                    test_qdp.find_time_delta() > 0
                    and max(test_df["Relative_time"]) < 1800
                ):
                    long_runs.append(test_file)
                    if max(test_df["Relative_time"]) > longest_run[0]:
                        longest_run = (
                            max(test_df["Relative_time"]), test_file)
                        continue

                mm_results, good, bad = test_mm_on_file(test_file, act_poi)
                good_list.append(good)
                bad_list.append(bad)
                qmp_results = test_qmp_on_file(test_file, act_poi)
                md_results = test_md_on_file(test_file, act_poi)
                qmodel_v2_results = test_qmodel_v2(test_file, act_poi)
                pois = []
                for res in mm_results:
                    pois.append(res)
                if pois != sorted(pois):
                    out_of_order += 1

                    for i in range(1, len(pois)):
                        if pois[i] < pois[i - 1]:
                            count[i] += 1
                mm_list.append(mm_results)
                qmp_list.append(qmp_results)
                md_list.append(md_results)
                qmodel_v2_list.append(qmodel_v2_results)
                qmp_deltas.append(compute_deltas(qmp_results))
                md_deltas.append(compute_deltas(md_results))
                qmodel_v2_deltas.append(compute_deltas(qmodel_v2_results))

    # Compute statistics
    stats = {}
    for i in [3, 4, 5]:
        if ratios[i]:
            ratios_array = np.array(ratios[i])
            avg_ratio = np.mean(ratios_array)
            min_ratio = np.min(ratios_array)
            max_ratio = np.max(ratios_array)
            q1 = np.percentile(ratios_array, 25)
            q3 = np.percentile(ratios_array, 75)
            iqr = q3 - q1
            stats[i] = {
                "avg": avg_ratio,
                "min": min_ratio,
                "max": max_ratio,
                "iqr": iqr
            }
        else:
            stats[i] = {"avg": None, "min": None, "max": None, "iqr": None}
    # Print final results
    print("\nFinal Ratio Statistics Across Directory:")
    for i in [3, 4, 5]:
        print(
            f"r_time_poi[{i}] - Avg: {stats[i]['avg']:.4f}, Min: {stats[i]['min']:.4f}, Max: {stats[i]['max']:.4f}, IQR: {stats[i]['iqr']:.4f}")
    input()
    print(f"--- EXPLORATORY RESULTS ---")
    print(f"Long Runs: {len(long_runs)}")
    for f in long_runs:
        print(f"\t-{f}")
    input()
    print(f"Longest run:  {longest_run}")
    print(f"Partial Fills: {len(partial_fills)}")
    for f in partial_fills:
        print(f"\t-{f}")
    input()
    intersection = [value for value in long_runs if value in partial_fills]

    print(f"Cross Reference: {len(intersection)}")
    for f in intersection:
        print(f"\t-{f}")
    input()
    print(f"Out-of-Order Precdictions={out_of_order}")
    print(f"OOO Prediction count={count}")
    input()
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
    qmodel_v2_ppr = extract_results(qmodel_v2_list)
    mm_mae, mm_mse, mm_rmse, mm_r2, mm_mape, mm_med_ape = [], [], [], [], [], []
    qmp_mae, qmp_mse, qmp_rmse, qmp_r2, qmp_mape, qmp_med_ape = [], [], [], [], [], []
    md_mae, md_mse, md_rmse, md_r2, md_mape, md_med_ape = [], [], [], [], [], []
    v2_mae, v2_mse, v2_rmse, v2_r2, v2_mape, v2_med_ape = [], [], [], [], [], []
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
    #######################
    # QMODEL V2
    #######################
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        qmodel_v2_ppr[0]["predicted"], qmodel_v2_ppr[0]["actual"], 1, VERBOSE
    )
    v2_mae.append(mae)
    v2_mse.append(mse)
    v2_rmse.append(rmse)
    v2_r2.append(r2)
    v2_mape.append(mape)
    v2_med_ape.append(med_ape)

    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        qmodel_v2_ppr[1]["predicted"], qmodel_v2_ppr[1]["actual"], 2, VERBOSE
    )
    v2_mae.append(mae)
    v2_mse.append(mse)
    v2_rmse.append(rmse)
    v2_r2.append(r2)
    v2_mape.append(mape)
    v2_med_ape.append(med_ape)

    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        qmodel_v2_ppr[2]["predicted"], qmodel_v2_ppr[2]["actual"], 3, VERBOSE
    )
    v2_mae.append(mae)
    v2_mse.append(mse)
    v2_rmse.append(rmse)
    v2_r2.append(r2)
    v2_mape.append(mape)
    v2_med_ape.append(med_ape)

    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        qmodel_v2_ppr[2]["predicted"], qmodel_v2_ppr[3]["actual"], 4, VERBOSE
    )
    v2_mae.append(mae)
    v2_mse.append(mse)
    v2_rmse.append(rmse)
    v2_r2.append(r2)
    v2_mape.append(mape)
    v2_med_ape.append(med_ape)
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        qmodel_v2_ppr[4]["predicted"], qmodel_v2_ppr[4]["actual"], 5, VERBOSE
    )
    v2_mae.append(mae)
    v2_mse.append(mse)
    v2_rmse.append(rmse)
    v2_r2.append(r2)
    v2_mape.append(mape)
    v2_med_ape.append(med_ape)
    mae, mse, rmse, r2, mape, med_ape = poi_k_metrics(
        qmodel_v2_ppr[5]["predicted"], qmodel_v2_ppr[5]["actual"], 6, VERBOSE
    )
    v2_mae.append(mae)
    v2_mse.append(mse)
    v2_rmse.append(rmse)
    v2_r2.append(r2)
    v2_mape.append(mape)
    v2_med_ape.append(med_ape)

    metrics_view(
        mm_mae,
        qmp_mae,
        md_mae,
        v2_mae,
        "MAE",
        "QMultiModel",
        "QSingleModel",
        "ModelData",
        "QModel_V2",
        "Average magnitude of errors between\npredicted/actual ignoring direction.\n(Lower is better)",
    )
    metrics_view(
        mm_mse,
        qmp_mse,
        md_mse,
        v2_mse,
        "MSE",
        "QMultiModel",
        "QSingleModel",
        "ModelData",
        "QModel_V2",
        "Average of squared differences between predicted/actual\nvalues emphasizes larger errors.\n(Lower is better)",
    )
    metrics_view(
        mm_rmse,
        qmp_rmse,
        md_rmse,
        v2_rmse,
        "RMSE",
        "QMultiModel",
        "QSingleModel",
        "ModelData",
        "QModel_V2",
        "Sqrt of MSE.\nUnits are the same as target variable.\n(Lower is better)",
    )
    metrics_view(
        mm_r2,
        qmp_r2,
        md_r2,
        v2_r2,
        "R^2",
        "QMultiModel",
        "QSingleModel",
        "ModelData",
        "QModel_V2",
        "How well the model 'fits' the data.\n(Bigger is better)",
    )
    metrics_view(
        mm_mape,
        qmp_mape,
        md_mape,
        v2_mape,
        "MAPE",
        "QMultiModel",
        "QSingleModel",
        "ModelData",
        "QModel_V2",
        "Average magnitude of errors as a percentage\nof the actual values.\n(Lower is better)",
    )

    metrics_view(
        mm_med_ape,
        qmp_med_ape,
        md_med_ape,
        v2_med_ape,
        "Median Absolute Error",
        "QMultiModel",
        "QSingleModel",
        "ModelData",
        "QModel_V2",
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

    # delta_distribution_view(mm_deltas, "QMultiModel")


if __name__ == "__main__":
    run()
