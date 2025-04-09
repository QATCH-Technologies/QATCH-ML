import io
import xgboost as xgb
import logging
from sklearn.pipeline import Pipeline
import pickle
import os
import pandas as pd
from q_model_data_processor import QDataProcessor
from ModelData import ModelData
import numpy as np
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

""" The percentage of the run data to ignore from the head of a difference curve. """
HEAD_TRIM_PERCENTAGE = 0.05
""" The percentage of the run data to ignore from the tail of a difference curve. """
TAIL_TRIM_PERCENTAGE = 0.5


class QModelPredictor:
    def __init__(self, booster_path: str, scaler_path: str) -> None:
        if booster_path is None or booster_path == "" or not os.path.exists(booster_path):
            logging.error(
                f'Booster path `{booster_path}` is empty string or does not exist.')
        if scaler_path is None or scaler_path == "" or not os.path.exists(scaler_path):
            logging.error(
                f'Scaler path `{scaler_path}` is empty string or does not exist.')

        self._booster: xgb.Booster = xgb.Booster()
        self._scaler: Pipeline = None
        try:
            self._booster.load_model(fname=booster_path)
            logging.info(f'Booster loaded from path `{booster_path}`.')
        except Exception as e:
            logging.error(
                f'Error loading booster from path `{booster_path}` with exception: `{e}`')
        try:
            self._scaler: Pipeline = self._load_scaler(scaler_path=scaler_path)
            logging.info(f'Scaler loaded from path `{scaler_path}`.')
        except Exception as e:
            logging.error(
                f'Error loading model from path `{scaler_path}` with exception: `{e}`')

    def _load_scaler(self, scaler_path: str) -> Pipeline:
        scaler = None
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        if scaler is None:
            raise IOError("Scaler could not be loaded from specified path.")
        return scaler

    def _filter_labels(self, predicted_labels: np.ndarray, multiplier: float = 1.5):
        filtered_labels = predicted_labels.copy()
        for cls in range(1, 7):
            class_indices = np.where(predicted_labels == cls)[0]
            if len(class_indices) == 0:
                continue
            centroid = np.median(class_indices)
            distances = np.abs(class_indices - centroid)
            mad = np.median(distances)
            threshold = multiplier * (mad if mad > 0 else 1)
            invalid_indices = class_indices[distances > threshold]
            filtered_labels[invalid_indices] = 0
        return filtered_labels

    def _get_model_data_predictions(self, file_buffer: str):
        model = ModelData()
        if isinstance(file_buffer, str):
            model_data_predictions = model.IdentifyPoints(file_buffer)
        else:
            file_buffer = self._reset_file_buffer(file_buffer)
            header = next(file_buffer)
            if isinstance(header, bytes):
                header = header.decode()
            csv_cols = (2, 4, 6, 7) if "Ambient" in header else (2, 3, 5, 6)
            file_data = np.loadtxt(
                file_buffer.readlines(), delimiter=",", usecols=csv_cols)
            relative_time = file_data[:, 0]
            resonance_frequency = file_data[:, 2]
            data = file_data[:, 3]
            model_data_predictions = model.IdentifyPoints(
                data_path="QModel Passthrough",
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

    def _reset_file_buffer(self, file_buffer: str):
        if isinstance(file_buffer, str):
            return file_buffer
        if hasattr(file_buffer, "seekable") and file_buffer.seekable():
            file_buffer.seek(0)
            return file_buffer
        else:
            raise Exception(
                "Cannot `seek` stream prior to passing to processing.")

    def _validate_file_buffer(self, file_buffer: str):
        if not isinstance(file_buffer, str) or file_buffer.strip() == "":
            raise ValueError(
                "File buffer must be a non-empty string containing CSV data.")

        try:
            df = pd.read_csv(file_buffer)
        except pd.errors.EmptyDataError:
            raise ValueError("The provided data file is empty.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing data file: `{str(e)}`")
        except Exception as e:
            raise ValueError(
                f"An unexpected error occurred while reading the data file: `{str(e)}`")

        if df.empty:
            raise ValueError("The data file does not contain any data.")

        required_columns = {"Dissipation",
                            "Resonance_Frequency", "Relative_time"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(
                f"Data file missing required columns: `{', '.join(missing)}`.")

        return df

    def _extract_predictions(self, predicted_probablities: np.ndarray, model_data_labels: np.ndarray):
        poi_results = {}
        predicted_labels = np.argmax(predicted_probablities, axis=1)
        for poi in range(1, 7):
            key = f"POI{poi}"
            indices = np.where(predicted_labels == poi)[0]

            if indices.size == 0:

                poi_results[key] = {"indices": [
                    model_data_labels[poi - 1]], "confidences": [0]}
            else:
                confidences = predicted_probablities[indices, poi]
                sort_order = np.argsort(-confidences)
                sorted_indices = indices[sort_order].tolist()
                sorted_confidences = confidences[sort_order].tolist()

                poi_results[key] = {"indices": sorted_indices,
                                    "confidences": sorted_confidences}

        return poi_results

    def predict(self, file_buffer: str, forecast_start: int = -1, forecast_end: int = -1, actual_poi_indices: np.ndarray = None):
        try:
            df = self._validate_file_buffer(file_buffer=file_buffer)
        except Exception as e:
            logging.error(
                f"File buffer `{file_buffer}` could not be validated because of error: `{e}`.")
            return
        model_data_labels = self._get_model_data_predictions(
            file_buffer=file_buffer)
        file_buffer = self._reset_file_buffer(file_buffer=file_buffer)
        feature_vector = QDataProcessor.process_data(
            file_buffer=file_buffer, live=True)
        transformed_feature_vector = self._scaler.transform(
            feature_vector.values)
        ddata = xgb.DMatrix(transformed_feature_vector)
        predicted_probabilites = self._booster.predict(ddata)
        extracted_predictions = self._extract_predictions(
            predicted_probabilites, model_data_labels)

        # Retrieve the initial fill region, i.e. left and right boundaries
        (lb_time, lb_index), (rb_time, rb_index) = QDataProcessor.find_initial_fill_region(
            feature_vector['Difference'], df['Relative_time'])
        # --- Plotting Section (Optional) ---
        if actual_poi_indices is not None:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 6))
            dissipation = feature_vector['Dissipation'].values
            poi_colors = {1: 'pink', 2: 'blue', 3: 'green',
                          4: 'orange', 5: 'purple', 6: 'cyan'}

            # --- Plot 1: Model Data Labels ---
            plt.subplot(1, 2, 1)
            plt.plot(dissipation, color='grey')
            plt.scatter(model_data_labels,
                        dissipation[model_data_labels])
            plt.scatter(actual_poi_indices,
                        dissipation[actual_poi_indices], marker='x', color='red')
            plt.axvline(lb_index, color='black', linestyle='dotted')
            plt.axvline(rb_index, color='black', linestyle='dotted')
            plt.xlabel("Sample Index")
            plt.ylabel("Dissipation")
            plt.title("Dissipation vs Index (Model Data Labels)")
            plt.clim(-0.5, 5.5)

            # --- Plot 2: Filtered Labels ---
            plt.subplot(1, 2, 2)
            plt.plot(dissipation, color='grey')
            for poi in range(1, 7):
                key = f"POI{poi}"
                poi_data = extracted_predictions.get(key, {})
                indices = poi_data.get("indices", [])
                if indices:
                    indices_arr = np.array(indices)
                    y_vals = dissipation[indices_arr]
                    # Plot all predictions with a lower alpha
                    plt.scatter(indices_arr, y_vals, color=poi_colors[poi],
                                marker='o', alpha=0.5, label=f'POI {poi}')
                    # Highlight the highest confidence guess (first in the sorted list)
                    highest_index = indices_arr[0]
                    plt.scatter(highest_index, dissipation[highest_index],
                                color=poi_colors[poi], marker='o', s=100,
                                alpha=1.0, edgecolor='k', linewidth=1.5)
            plt.axvline(lb_index, color='black', linestyle='dotted')
            plt.axvline(rb_index, color='black', linestyle='dotted')
            plt.scatter(actual_poi_indices, dissipation[actual_poi_indices],
                        color='red', marker='x')
            plt.xlabel("Sample Index")
            plt.ylabel("Dissipation")
            plt.title("Dissipation vs Index (QModel Labels)")
            plt.clim(-0.5, 5.5)

            plt.tight_layout()
            plt.show()
        return extracted_predictions


if __name__ == "__main__":
    booster_path = os.path.join(
        "QModel", "SavedModels", "qmodel_v2", "qmodel_v2.json")
    scaler_path = os.path.join(
        "QModel", "SavedModels", "qmodel_v2", "qmodel_scaler.pkl")
    test_dir = os.path.join('content', 'static', 'test')
    test_content = QDataProcessor.load_content(data_dir=test_dir)
    qmp = QModelPredictor(booster_path=booster_path, scaler_path=scaler_path)
    for i, (data_file, poi_file) in enumerate(test_content):
        poi_indices = pd.read_csv(poi_file, header=None)
        qmp.predict(file_buffer=data_file, actual_poi_indices=poi_indices)
