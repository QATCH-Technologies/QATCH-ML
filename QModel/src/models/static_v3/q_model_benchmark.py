from functools import singledispatch
import random
import logging
import pandas as pd
import numpy as np
import os
import shutil
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import math
from typing import List
from tqdm import tqdm
from ModelData import ModelData
from q_model_data_processor import QDataProcessor
from q_model_predictor import QModelPredictor
from q_image_clusterer import QClusterer
from q_multi_model import QPredictor
from q_single_model import QModelPredict
import time
import re
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
ERROR_CASE = {
    "POI1": {"indices": [-1], "confidences": [-1]},
    "POI2": {"indices": [-1], "confidences": [-1]},
    "POI3": {"indices": [-1], "confidences": [-1]},
    "POI4": {"indices": [-1], "confidences": [-1]},
    "POI5": {"indices": [-1], "confidences": [-1]},
    "POI6": {"indices": [-1], "confidences": [-1]},
}
QMODEL_V2_BOOSTER_PATH = os.path.join(
    "QModel", "SavedModels", "qmodel_v3", "qmodel_v3_booster.json")
QMODEL_V2_SCALER_PATH = os.path.join(
    "QModel", "SavedModels", "qmodel_v3", "qmodel_v3_scaler.pkl")
Q_IMAGE_CLUSTER_PATH = os.path.join("QModel", "SavedModels", "cluster.joblib")
Q_MULTI_MODEL_0_PATH = os.path.join(
    "QModel", "SavedModels", "QMultiType_0.json")
Q_MULTI_MODEL_1_PATH = os.path.join(
    "QModel", "SavedModels", "QMultiType_1.json")
Q_MULTI_MODEL_2_PATH = os.path.join(
    "QModel", "SavedModels", "QMultiType_2.json")

Q_SINGLE_PREDICTOR_1 = os.path.join("QModel", "SavedModels", "QModel_1.json")
Q_SINGLE_PREDICTOR_2 = os.path.join("QModel", "SavedModels", "QModel_2.json")
Q_SINGLE_PREDICTOR_3 = os.path.join("QModel", "SavedModels", "QModel_3.json")
Q_SINGLE_PREDICTOR_4 = os.path.join("QModel", "SavedModels", "QModel_4.json")
Q_SINGLE_PREDICTOR_5 = os.path.join("QModel", "SavedModels", "QModel_5.json")
Q_SINGLE_PREDICTOR_6 = os.path.join("QModel", "SavedModels", "QModel_6.json")


@singledispatch
def predict_dispatch(predictor, file_buffer, actual_poi_indices):
    raise NotImplementedError(
        f"Prediction not implemented for type {type(predictor)}"
    )


@predict_dispatch.register
def _(predictor: 'ModelData', file_buffer, actual_poi_indices):
    if isinstance(file_buffer, str):
        model_data_predictions = predictor.IdentifyPoints(file_buffer)
    else:
        file_buffer = predictor._reset_file_buffer(file_buffer)
        header = next(file_buffer)
        if isinstance(header, bytes):
            header = header.decode()
        csv_cols = (2, 4, 6, 7) if "Ambient" in header else (2, 3, 5, 6)
        file_data = np.loadtxt(file_buffer.readlines(),
                               delimiter=",", usecols=csv_cols)
        relative_time = file_data[:, 0]
        resonance_frequency = file_data[:, 2]
        data = file_data[:, 3]
        model_data_predictions = predictor.IdentifyPoints(
            data_path="Benchmark",
            times=relative_time,
            freq=resonance_frequency,
            diss=data
        )
    model_data_points = []
    if isinstance(model_data_predictions, list):
        for pt in model_data_predictions:
            if isinstance(pt, int):
                model_data_points.append(pt)
            elif isinstance(pt, list) and pt:
                model_data_points.append(max(pt, key=lambda x: x[1])[0])
    return model_data_points


@predict_dispatch.register
def _(predictor: 'QModelPredictor', file_buffer, actual_poi_indices):
    qmodel_v2_predictions = predictor.predict(file_buffer=file_buffer)
    qmodel_v2_points = []

    for poi in ["POI1", "POI2", "POI3", "POI4", "POI5", "POI6"]:
        if qmodel_v2_predictions is None:
            return ERROR_CASE
        qmodel_v2_points.append(qmodel_v2_predictions.get(
            poi, {}).get('indices', [None])[0])
    return qmodel_v2_points


@predict_dispatch.register
def _(predictor: 'QClusterer', file_buffer, actual_poi_indices):
    label = predictor.predict_label(file_buffer=file_buffer)
    predictor_0 = QPredictor(Q_MULTI_MODEL_0_PATH)
    predictor_1 = QPredictor(Q_MULTI_MODEL_1_PATH)
    predictor_2 = QPredictor(Q_MULTI_MODEL_2_PATH)
    try:
        if label == 0:
            candidates = predictor_0.predict(
                file_buffer=file_buffer, run_type=label)
        elif label == 1:
            candidates = predictor_1.predict(
                file_buffer=file_buffer, run_type=label)
        elif label == 2:
            candidates = predictor_2.predict(
                file_buffer=file_buffer, run_type=label)
        q_predictor_points = []
        for c in candidates:
            q_predictor_points.append(c[0][0])
    except Exception as e:
        logging.error(f"QMultiModel prediction failed with error {e}.")
        q_predictor_points = [-1, -1, -1, -1, -1, -1]

    return q_predictor_points


@predict_dispatch.register
def _(predictor: 'QModelPredict', file_buffer, actual_poi_indices):
    q_single_model_points = predictor.predict(file_buffer)
    return q_single_model_points


def compute_metrics(actual: List[int], predicted: List[int]) -> dict:
    if len(actual) != len(predicted):
        raise ValueError(
            "The number of predicted indices does not match the number of actual indices.")

    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    differences = np.abs(actual_np - predicted_np)
    mae = np.mean(differences)
    mse = np.mean((actual_np - predicted_np) ** 2)
    rmse = math.sqrt(mse)
    mean_error = np.mean(actual_np - predicted_np)
    medae = np.median(differences)
    ss_tot = np.sum((actual_np - np.mean(actual_np)) ** 2)
    ss_res = np.sum((actual_np - predicted_np) ** 2)
    r_squared = 1 - ss_res/ss_tot if ss_tot != 0 else None

    std_error = np.std(actual_np - predicted_np)

    try:
        mape = np.mean(differences / np.abs(actual_np)) * 100
    except ZeroDivisionError:
        mape = None

    return {
        "differences": differences.tolist(),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mean_error": mean_error,
        "median_absolute_error": medae,
        "r_squared": r_squared,
        "std_error": std_error,
        "mape": mape
    }


class Benchmarker:
    BAD_PLOTS_DIR = "bad_plots"

    def __init__(self, predictors: List, bad_cases_dir: str = "bad_case_plots"):
        self._predictors = predictors
        self.bad_cases_dir = bad_cases_dir
        # aggregate storage for final metrics
        self.results = {
            type(model).__name__: {"actual": [],
                                   "predicted": [], "runtimes": []}
            for model in predictors
        }
        # will hold dicts: data_file, actual, predicted, bad_flags, tolerance
        self.bad_cases: List[dict] = []

    def run(self, test_directory: str, test_size: int, plotting: bool = True):
        # 1) Recreate bad‐cases directory
        if os.path.exists(self.bad_cases_dir):
            shutil.rmtree(self.bad_cases_dir)
        os.makedirs(self.bad_cases_dir, exist_ok=True)
        self.bad_cases.clear()

        # 2) Load test set
        test_content = QDataProcessor.load_content(
            data_dir=test_directory, num_datasets=test_size
        )
        random.shuffle(test_content)

        for data_file, poi_file in tqdm(test_content, desc="<Benchmarking>"):
            # ground‐truth POIs
            poi_indices_df = pd.read_csv(poi_file, header=None)
            actual_indices = poi_indices_df.values.flatten().tolist()

            df = pd.read_csv(data_file)
            num_samples = len(df)
            times = df["Relative_time"].values
            max_time = times.max()

            tol_low = max_time / 1000
            tol_high = max_time / 100
            tol_list = [tol_low]*3 + [tol_high]*3

            for model in self._predictors:
                model_name = type(model).__name__
                start = time.perf_counter()
                predicted = predict_dispatch(
                    model, file_buffer=data_file, actual_poi_indices=poi_indices_df.values
                )
                elapsed = time.perf_counter() - start

                self.results[model_name]["actual"].append(actual_indices)
                self.results[model_name]["predicted"].append(predicted)
                self.results[model_name]["runtimes"].append(elapsed)
                if model_name == "QModelPredictor":
                    try:
                        diffs_time = [
                            abs(times[int(a)] - times[int(p)])
                            for a, p in zip(actual_indices, predicted)
                        ]
                        bad_flags = [
                            (diff > tol_list[i]) if i != 2 else False
                            for i, diff in enumerate(diffs_time)
                        ]
                        if any(bad_flags):
                            self.bad_cases.append({
                                "data_file":   data_file,
                                "actual":      actual_indices,
                                "predicted":   predicted,
                                "bad_flags":   bad_flags,
                                "tolerances":  tol_list
                            })
                    except Exception as e:
                        # if anything goes wrong (e.g. bad int conversion, index‐out‐of‐range, etc.), just skip it
                        # you could also log it if you want:
                        logging.warning(
                            f"Skipping diff calc for {data_file}: {e}")
                        pass

        # 3) Compute / plot metrics
        aggregated_metrics = self._aggregate_metrics()
        if plotting:
            self.plot_benchmark_metrics(aggregated_metrics)
            self._plot_bad_cases()
        return aggregated_metrics

    def plot_benchmark_metrics(self, aggregated_metrics: dict):
        model_name_mapping = {
            "ModelData": "ModelData",
            "QModelPredictor": "QModel V3",
            "QClusterer": "QModel V2",
            "QModelPredict": "QModel V1"
        }

        metric_keys = [
            "mae", "mse", "rmse", "mean_error",
            "median_absolute_error", "r_squared", "std_error"
        ]
        metric_descriptions = {
            "mae": "Mean Absolute Error: The average absolute difference between predictions and actual values.",
            "mse": "Mean Squared Error: The average squared difference between predictions and actual values.",
            "rmse": "Root Mean Squared Error: The square root of the MSE, measuring error magnitude.",
            "mean_error": "Mean Error: The average error, indicating bias in predictions.",
            "median_absolute_error": "Median Absolute Error: The median of the absolute differences, less sensitive to outliers.",
            "r_squared": "R^2: Proportion of variance in the data explained by the model.",
            "std_error": "Standard Error: The standard deviation of the prediction errors."
        }
        runtime_description = "Average Runtime: The mean execution time per model (in seconds)."
        all_metrics = metric_keys + ["avg_time"]
        model_names = list(aggregated_metrics.keys())
        poi_names = list(aggregated_metrics[model_names[0]].keys())

        preferred_style = 'seaborn-whitegrid'
        if preferred_style in plt.style.available:
            plt.style.use(preferred_style)
        else:
            plt.style.use('ggplot')

        for metric in metric_keys:
            fig, ax = plt.subplots(figsize=(12, 8))
            pois_to_plot = [poi for poi in poi_names if "POI" in poi]
            x = np.arange(len(pois_to_plot))
            bar_width = 0.8 / len(model_names)

            for idx, model in enumerate(model_names):
                display_model = model_name_mapping.get(model, model)
                metric_values = [aggregated_metrics[model]
                                 [poi][metric] for poi in pois_to_plot]
                bar_positions = x + idx * bar_width
                bars = ax.bar(
                    bar_positions,
                    metric_values,
                    bar_width,
                    label=display_model,
                    edgecolor='black'
                )
                for bar in bars:
                    height = bar.get_height()
                    offset = max(metric_values) * \
                        0.01 if max(metric_values) > 0 else 0.05
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + offset,
                        f'{height:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=10
                    )
            # Adjust x-axis ticks.
            ax.set_xticks(x + bar_width * (len(model_names) - 1) / 2)
            ax.set_xticklabels(pois_to_plot, fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{metric.upper()} by POI for Each Model',
                         fontsize=14, fontweight='bold')
            ax.legend(title="Models", fontsize=10, title_fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)

            description = metric_descriptions.get(metric, "")
            ax.text(0.98, 0.98, description,
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.tight_layout()
            plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        model_runtimes = [aggregated_metrics[model]["avg_time"]
                          for model in model_names]
        display_model_names = [model_name_mapping.get(
            model, model) for model in model_names]
        bars = ax.bar(display_model_names, model_runtimes, edgecolor='black')
        ax.set_ylabel("Average Runtime (s)", fontsize=12)
        ax.set_title("Average Runtime for Each Model",
                     fontsize=14, fontweight='bold')

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height * 1.01,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        ax.text(0.98, 0.98, runtime_description,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.tight_layout()
        plt.show()

    def _aggregate_metrics(self):
        aggregated = {}
        for model_name, data in self.results.items():
            aggregated[model_name] = {}
            preds = data.get("predicted", [])
            if not preds:
                # no predictions → skip POI metrics
                continue

            for idx in range(6):
                # build the “actual” list exactly as before
                actual_i = [a[idx] for a in data["actual"]]
                key = f"POI{idx+1}"

                # robust per-element extraction
                predicted_i = []
                for p in preds:
                    if isinstance(p, dict):
                        # dict-style: must have key "POI1", …, "POI6"
                        predicted_i.append(p[key])
                    else:
                        # sequence-style: list, tuple or ndarray
                        predicted_i.append(p[idx])

                aggregated[model_name][key] = compute_metrics(
                    actual_i, predicted_i)

            # finally average runtime
            aggregated[model_name]["avg_time"] = float(
                np.mean(data["runtimes"]))

        return aggregated

    def _plot_bad_cases(self):
        """Plot each bad run and save into POI-specific subfolders under bad_cases_dir."""
        # 1) wipe & recreate base bad_cases_dir
        if os.path.exists(self.bad_cases_dir):
            shutil.rmtree(self.bad_cases_dir)
        os.makedirs(self.bad_cases_dir, exist_ok=True)

        for case in self.bad_cases:
            # --- data & plot setup ---
            df = pd.read_csv(case["data_file"])
            times = df["Relative_time"].values
            diss_raw = df['Dissipation'].values
            # diff_raw = df['Difference'].values
            rf_raw = df['Resonance_Frequency'].values

            # Min–max normalize each series
            def minmax(arr):
                return (arr - arr.min()) / (arr.max() - arr.min())

            diss = minmax(diss_raw)
            # diff = minmax(diff_raw)
            rf = minmax(rf_raw)
            tol_list = case["tolerances"]
            bad_flags = case["bad_flags"]
            actual_idxs = case["actual"]
            pred_idxs = case["predicted"]
            base = os.path.splitext(os.path.basename(case["data_file"]))[0]

            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            ax.plot(times, diss, label="Dissipation",
                    linewidth=1.5, color='red')
            # ax.plot(times, diff, label='Difference',
            #         linewidth=1.5, color='blue')
            ax.plot(times, rf,   label='Resonance_Frequency',
                    linewidth=1.5, color='green')
            y_max = ax.get_ylim()[1]

            # --- annotate each failing POI ---
            for i, is_bad in enumerate(bad_flags):
                if not is_bad:
                    continue

                actual_idx = actual_idxs[i]
                pred_idx = pred_idxs[i]
                tol_secs = tol_list[i]
                act_t = times[actual_idx]
                pred_t = times[pred_idx]
                act_d = diss[actual_idx]
                time_off = abs(pred_t - act_t)

                # shade tolerance window (seconds)
                t_min = max(act_t - tol_secs, times[0])
                t_max = min(act_t + tol_secs, times[-1])
                ax.axvspan(t_min, t_max, color='red', alpha=0.2)

                # actual POI marker
                ax.scatter(act_t, act_d,
                           marker='o', s=100,
                           edgecolors='black', facecolors='none',
                           label='_nolegend_')
                # predicted POI line
                ax.axvline(pred_t,
                           color='red',
                           linestyle='--',
                           linewidth=2,
                           label='_nolegend_')

                # annotations
                ax.text(act_t, y_max * 0.95,
                        f"tol={tol_secs:.1f}s",
                        fontsize=10, fontweight='bold',
                        color="grey",
                        ha='center', va='top')
                ax.text(pred_t, y_max * 0.90,
                        f"off={time_off:.2f}s",
                        fontsize=10, fontweight='bold',
                        color="grey",
                        ha='center', va='top')
                ax.text(act_t, act_d,
                        f"A{i+1}",
                        fontsize=10, fontweight='bold',
                        color="black",
                        ha='right', va='bottom')
                ax.text(pred_t, y_max * 0.85,
                        f"P{i+1}",
                        fontsize=10, fontweight='bold',
                        color="grey",
                        ha='right', va='bottom')

            # --- styling & legend ---
            ax.set_title(f"Bad run: {base}", fontsize=16, fontweight='bold')
            ax.set_xlabel("Relative_time (s)", fontsize=14)
            ax.set_ylabel("Dissipation", fontsize=14)
            ax.grid(which='both', linestyle='--', alpha=0.6)

            region_patch = Line2D([0], [0], color='red', alpha=0.2, lw=10)
            actual_marker = Line2D([0], [0],
                                   marker='o', color='black',
                                   markerfacecolor='none', markersize=10, lw=0)
            pred_line = Line2D([0], [0],
                               linestyle='--', color='red', lw=2)
            ax.legend([actual_marker, pred_line, region_patch],
                      ["Actual POI", "Predicted POI", "Tolerance region"],
                      fontsize=12, loc="lower left")

            plt.tight_layout()

            # 2) save into each POI# subdir
            for i, is_bad in enumerate(bad_flags):
                if not is_bad:
                    continue
                poi_dir = os.path.join(self.bad_cases_dir, f"POI{i+1}")
                os.makedirs(poi_dir, exist_ok=True)
                fname = f"{base}_POI{i+1}_bad.png"
                fig.savefig(os.path.join(poi_dir, fname), bbox_inches="tight")

            plt.close(fig)


if __name__ == "__main__":
    test_directory = os.path.join("content", "dropbox_dump")
    predictors = [
        QModelPredictor(booster_path=QMODEL_V2_BOOSTER_PATH,
                        scaler_path=QMODEL_V2_SCALER_PATH),
        QClusterer(model_path="QModel/SavedModels/cluster.joblib"),
        QModelPredict(predictor_path_1=Q_SINGLE_PREDICTOR_1, predictor_path_2=Q_SINGLE_PREDICTOR_2, predictor_path_3=Q_SINGLE_PREDICTOR_3,
                      predictor_path_4=Q_SINGLE_PREDICTOR_4, predictor_path_5=Q_SINGLE_PREDICTOR_5, predictor_path_6=Q_SINGLE_PREDICTOR_6)

    ]
    benchmarker = Benchmarker(predictors=predictors)
    aggregated_metrics = benchmarker.run(
        test_directory=test_directory, test_size=300)
