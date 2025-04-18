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
from q_image_clusterer import QClusterer
from q_multi_model import QPredictor
from q_single_model import QModelPredict
import time
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
    def __init__(self, predictors: List):
        self._predictors = predictors
        self.results = {
            type(model).__name__: {"actual": [],
                                   "predicted": [], "runtimes": []}
            for model in predictors
        }

    def run(self, test_directory: str, test_size: int, plotting: bool = True):
        test_content = QDataProcessor.load_content(
            data_dir=test_directory, num_datasets=test_size)
        random.shuffle(test_content)

        for data_file, poi_file in tqdm(test_content, desc="<Benchmarking>"):
            poi_indices_df = pd.read_csv(poi_file, header=None)
            actual_indices = poi_indices_df.values.flatten().tolist()

            for model in self._predictors:
                model_name = type(model).__name__
                start_time = time.perf_counter()
                predictions = predict_dispatch(
                    model,
                    file_buffer=data_file,
                    actual_poi_indices=poi_indices_df.values
                )
                elapsed_time = time.perf_counter() - start_time

                self.results[model_name]["predicted"].append(predictions)
                self.results[model_name]["actual"].append(actual_indices)
                self.results[model_name]["runtimes"].append(elapsed_time)

        aggregated_metrics = {}
        for model_name, data in self.results.items():
            aggregated_metrics[model_name] = {}
            for poi_index in range(6):

                try:
                    actual_i = [actual[poi_index] for actual in data["actual"]]
                    predicted_i = [pred[poi_index]
                                   for pred in data["predicted"]]
                    metrics = compute_metrics(
                        actual=actual_i, predicted=predicted_i)
                    aggregated_metrics[model_name][f"POI{poi_index+1}"] = metrics
                    logging.info(
                        f"Aggregated metrics for {model_name} - POI {poi_index+1}: {metrics}"
                    )
                except Exception as ve:
                    logging.error(
                        f"Error computing aggregated metrics for {model_name} - POI {poi_index+1}: {ve}"
                    )
                    pred = data["predicted"]
                    logging.error(f"-- List was {pred}")
            try:
                avg_runtime = np.mean(data["runtimes "])
                aggregated_metrics[model_name]["avg_time"] = avg_runtime
                logging.info(
                    f"Average runtime for {model_name}: {avg_runtime}")
            except:
                aggregated_metrics[model_name]["avg_time"] = 0

        if plotting:
            self.plot_benchmark_metrics(aggregated_metrics)
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


if __name__ == "__main__":
    test_directory = os.path.join("content", "static", "test")
    predictors = [
        QModelPredictor(booster_path=QMODEL_V2_BOOSTER_PATH,
                        scaler_path=QMODEL_V2_SCALER_PATH),
        QClusterer(model_path="QModel/SavedModels/cluster.joblib"),
        QModelPredict(predictor_path_1=Q_SINGLE_PREDICTOR_1, predictor_path_2=Q_SINGLE_PREDICTOR_2, predictor_path_3=Q_SINGLE_PREDICTOR_3,
                      predictor_path_4=Q_SINGLE_PREDICTOR_4, predictor_path_5=Q_SINGLE_PREDICTOR_5, predictor_path_6=Q_SINGLE_PREDICTOR_6)

    ]
    benchmarker = Benchmarker(predictors=predictors)
    aggregated_metrics = benchmarker.run(
        test_directory=test_directory, test_size=100)
