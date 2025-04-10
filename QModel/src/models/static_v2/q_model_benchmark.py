from functools import singledispatch
import random
import logging
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from typing import List
from tqdm import tqdm
from ModelData import ModelData
from q_model_data_processor import QDataProcessor
from q_model_predictor import QModelPredictor
import time
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

QMODEL_V2_BOOSTER_PATH = os.path.join(
    "QModel", "SavedModels", "qmodel_v2", "qmodel_v2.json")
QMODEL_V2_SCALER_PATH = os.path.join(
    "QModel", "SavedModels", "qmodel_v2", "qmodel_scaler.pkl")


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
                # Taking the index corresponding to the maximum value in the sublist
                model_data_points.append(max(pt, key=lambda x: x[1])[0])
    return model_data_points


@predict_dispatch.register
def _(predictor: 'QModelPredictor', file_buffer, actual_poi_indices):
    qmodel_v2_predictions = predictor.predict(file_buffer=file_buffer)
    qmodel_v2_points = []
    for poi in ["POI1", "POI2", "POI3", "POI4", "POI5", "POI6"]:
        qmodel_v2_points.append(qmodel_v2_predictions.get(
            poi, {}).get('indices', [None])[0])
    return qmodel_v2_points


def compute_metrics(actual: List[int], predicted: List[int]) -> dict:
    if len(actual) != len(predicted):
        raise ValueError(
            "The number of predicted indices does not match the number of actual indices.")

    # Convert to NumPy arrays for efficiency
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)

    differences = np.abs(actual_np - predicted_np)
    mae = np.mean(differences)
    mse = np.mean((actual_np - predicted_np) ** 2)
    rmse = math.sqrt(mse)

    # Mean error (bias)
    mean_error = np.mean(actual_np - predicted_np)

    # Median Absolute Error
    medae = np.median(differences)

    # R-squared Score
    ss_tot = np.sum((actual_np - np.mean(actual_np)) ** 2)
    ss_res = np.sum((actual_np - predicted_np) ** 2)
    r_squared = 1 - ss_res/ss_tot if ss_tot != 0 else None

    # Standard Deviation of the Differences
    std_error = np.std(actual_np - predicted_np)

    # Optionally, Mean Absolute Percentage Error (if no actual value is zero)
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
    def __init__(self, predictors: List):
        self._predictors = predictors
        # Now each model’s result dictionary includes a "runtimes" list.
        self.results = {
            type(model).__name__: {"actual": [],
                                   "predicted": [], "runtimes": []}
            for model in predictors
        }

    def run(self, test_directory: str, test_size: int, plotting: bool = True):
        # Load and shuffle test content
        test_content = QDataProcessor.load_content(
            data_dir=test_directory, num_datasets=test_size)
        random.shuffle(test_content)

        for data_file, poi_file in tqdm(test_content, desc="<Benchmarking>"):
            # Read the actual POI indices (each file should have 6 values)
            poi_indices_df = pd.read_csv(poi_file, header=None)
            actual_indices = poi_indices_df.values.flatten().tolist()

            for model in self._predictors:
                model_name = type(model).__name__

                # Start timing the prediction call.
                start_time = time.perf_counter()
                predictions = predict_dispatch(
                    model,
                    file_buffer=data_file,
                    actual_poi_indices=poi_indices_df.values
                )
                # Stop timing and calculate elapsed time.
                elapsed_time = time.perf_counter() - start_time

                # Record predictions and actual values.
                self.results[model_name]["predicted"].append(predictions)
                self.results[model_name]["actual"].append(actual_indices)
                # Record the runtime.
                self.results[model_name]["runtimes"].append(elapsed_time)

        aggregated_metrics = {}
        for model_name, data in self.results.items():
            aggregated_metrics[model_name] = {}
            for poi_index in range(6):
                actual_i = [actual[poi_index] for actual in data["actual"]]
                predicted_i = [pred[poi_index] for pred in data["predicted"]]
                try:
                    metrics = compute_metrics(
                        actual=actual_i, predicted=predicted_i)
                    aggregated_metrics[model_name][f"POI{poi_index+1}"] = metrics
                    logging.info(
                        f"Aggregated metrics for {model_name} - POI {poi_index+1}: {metrics}"
                    )
                except ValueError as ve:
                    logging.error(
                        f"Error computing aggregated metrics for {model_name} - POI {poi_index+1}: {ve}"
                    )
            # Calculate the average runtime (in seconds) for the model.
            avg_runtime = np.mean(data["runtimes"])
            aggregated_metrics[model_name]["avg_time"] = avg_runtime
            logging.info(f"Average runtime for {model_name}: {avg_runtime}")

        if plotting:
            self.plot_benchmark_metrics(aggregated_metrics)
        return aggregated_metrics

    def plot_benchmark_metrics(self, aggregated_metrics: dict):
        metric_keys = ["mae", "mse", "rmse", "mean_error",
                       "median_absolute_error", "r_squared", "std_error"]
        # Add runtime to the list of metrics if desired
        all_metrics = metric_keys + ["avg_time"]

        model_names = list(aggregated_metrics.keys())
        poi_names = list(aggregated_metrics[model_names[0]].keys())
        plt.style.use('ggplot')

        # Plot metrics that are computed per POI.
        for metric in metric_keys:
            fig, ax = plt.subplots(figsize=(12, 8))
            # exclude avg_time (if it shares same structure)
            x = np.arange(len(poi_names) - 1)
            bar_width = 0.8 / len(model_names)

            for idx, model in enumerate(model_names):
                metric_values = [aggregated_metrics[model]
                                 [poi][metric] for poi in poi_names if "POI" in poi]
                bar_positions = x + idx * bar_width
                bars = ax.bar(
                    bar_positions,
                    metric_values,
                    bar_width,
                    label=model,
                    edgecolor='black'
                )
                offset = max(metric_values) * \
                    0.01 if max(metric_values) > 0 else 0.05
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + offset,
                        f'{height:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=10
                    )
            ax.set_xticks(x + bar_width * (len(model_names) - 1) / 2)
            ax.set_xticklabels(
                [poi for poi in poi_names if "POI" in poi], fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{metric.upper()} by POI for each Model',
                         fontsize=14, fontweight='bold')
            ax.legend(title="Models", fontsize=10, title_fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

        # Plot average runtime (global, not per-POI)
        fig, ax = plt.subplots(figsize=(8, 6))
        model_runtimes = [aggregated_metrics[model]["avg_time"]
                          for model in model_names]
        bars = ax.bar(model_names, model_runtimes, edgecolor='black')
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
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test_directory = os.path.join("content", "static", "test")
    predictors = [
        ModelData(),
        QModelPredictor(booster_path=QMODEL_V2_BOOSTER_PATH,
                        scaler_path=QMODEL_V2_SCALER_PATH),
    ]
    benchmarker = Benchmarker(predictors=predictors)
    aggregated_metrics = benchmarker.run(
        test_directory=test_directory, test_size=np.inf)
