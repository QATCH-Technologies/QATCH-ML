import xgboost as xgb
import logging
from sklearn.pipeline import Pipeline
import pickle
import os
import pandas as pd
from ModelData import ModelData
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from q_model_data_processor import QDataProcessor
from discriminator_predictor import DiscriminatorPredictor

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('numba.core').setLevel(logging.WARNING)


class PostProcesses:
    @staticmethod
    def _normalize(data: np.ndarray):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    @njit
    def ordering_numba(
        poi1_indices, poi3_indices, poi4_indices, poi5_indices, poi6_indices,
        relative_times
    ):
        valid1 = np.zeros(poi1_indices.shape[0], dtype=np.bool_)
        valid3 = np.zeros(poi3_indices.shape[0], dtype=np.bool_)
        valid4 = np.zeros(poi4_indices.shape[0], dtype=np.bool_)
        valid5 = np.zeros(poi5_indices.shape[0], dtype=np.bool_)
        valid6 = np.zeros(poi6_indices.shape[0], dtype=np.bool_)

        for i in range(poi1_indices.shape[0]):
            t1 = relative_times[poi1_indices[i]]
            for j in range(poi3_indices.shape[0]):
                t3 = relative_times[poi3_indices[j]]
                if t3 <= t1:
                    continue
                delta1 = t3 - t1
                for k in range(poi4_indices.shape[0]):
                    t4 = relative_times[poi4_indices[k]]
                    if t4 <= t3:
                        continue
                    delta2 = t4 - t3
                    if delta2 < delta1:
                        continue
                    for l in range(poi5_indices.shape[0]):
                        t5 = relative_times[poi5_indices[l]]
                        if t5 <= t4:
                            continue
                        delta3 = t5 - t4
                        if delta3 < delta1 or delta3 < delta2:
                            continue
                        for m in range(poi6_indices.shape[0]):
                            t6 = relative_times[poi6_indices[m]]
                            if t6 <= t5:
                                continue
                            delta4 = t6 - t5
                            if delta4 < delta1 or delta4 < delta2 or delta4 < delta3:
                                continue
                            valid1[i] = True
                            valid3[j] = True
                            valid4[k] = True
                            valid5[l] = True
                            valid6[m] = True
        return valid1, valid3, valid4, valid5, valid6

    @staticmethod
    def ordering(predictions: dict, relative_times: np.ndarray):
        poi1_indices = np.array(predictions["POI1"]["indices"])
        poi3_indices = np.array(predictions["POI3"]["indices"])
        poi4_indices = np.array(predictions["POI4"]["indices"])
        poi5_indices = np.array(predictions["POI5"]["indices"])
        poi6_indices = np.array(predictions["POI6"]["indices"])

        valid1, valid3, valid4, valid5, valid6 = PostProcesses.ordering_numba(
            poi1_indices, poi3_indices, poi4_indices, poi5_indices, poi6_indices,
            relative_times
        )

        # Rebuild the predictions dictionary only with valid candidates:
        filtered_predictions = {}
        for poi in ["POI1", "POI3", "POI4", "POI5", "POI6"]:
            data = predictions[poi]
            # Create boolean mask for each POI.
            if poi == "POI1":
                mask = valid1
            elif poi == "POI3":
                mask = valid3
            elif poi == "POI4":
                mask = valid4
            elif poi == "POI5":
                mask = valid5
            elif poi == "POI6":
                mask = valid6
            filtered_indices = [idx for idx, valid in zip(
                data["indices"], mask) if valid]
            filtered_confidences = [conf for conf, valid in zip(
                data["confidences"], mask) if valid]
            filtered_predictions[poi] = {
                "indices": filtered_indices,
                "confidences": filtered_confidences
            }
        # POI2 is not part of the combination check.
        filtered_predictions["POI2"] = predictions["POI2"]
        return filtered_predictions

    @staticmethod
    def final_check(filtered_predictions: dict, extracted_predictions: dict, model_data_labels: list) -> dict:
        poi_keys = [f"POI{i}" for i in range(1, 7)]

        for key in poi_keys:
            fp = filtered_predictions.get(key, {})
            fp_indices = fp.get("indices", [])

            if not fp_indices:
                ep = extracted_predictions.get(key, {})
                ep_confidences = ep.get("confidences", [])
                filtered_predictions.setdefault(
                    key, {})['indices'] = [model_data_labels[i]]
                filtered_predictions[key]['confidences'] = ep_confidences[0]

        return filtered_predictions

    def post_poi_2(candidates: dict, feature_vector: pd.DataFrame, actual: np.ndarray = None):
        poi_2_candidates = candidates.get("POI2").get("indices")
        if actual is not None and abs(candidates.get("POI5").get("indices")[0] - actual[1]) > 20:
            plt.figure()
            plt.plot(PostProcesses._normalize(
                feature_vector["Dissipation"].values), label="Dissipation", color='grey')
            plt.axvline(actual[1], linestyle='dashed',
                        color='red', label='Actual')
            plt.axvline(candidates.get("POI2").get("indices")
                        [0], color='pink', label='Predicted')
            plt.legend()
            delta = abs(candidates.get("POI2").get("indices")[0] - actual[1])
            plt.title(f"POI2 Error of {delta}")
            plt.show()
        return candidates

    def post_poi_4(candidates: dict, feature_vector: pd.DataFrame, actual: np.ndarray = None):
        return candidates

    def post_poi_5(candidates: dict, feature_vector: pd.DataFrame, actual: np.ndarray = None):
        if actual is not None and abs(candidates.get("POI5").get("indices")[0] - actual[4]) > 20:
            plt.figure()
            plt.plot(PostProcesses._normalize(
                feature_vector["Dissipation"].values), label="Dissipation", color='grey')
            plt.scatter()
            plt.axvline(actual[4], linestyle='dashed',
                        color='red', label='Actual')
            plt.axvline(candidates.get("POI4").get("indices")
                        [0], color='pink', label='Predicted')
            plt.legend()
            delta = abs(candidates.get("POI5").get("indices")[0] - actual[4])
            plt.title(f"POI5 Error of {delta}")
            plt.show()
        return candidates


class QModelPredictor:

    def __init__(self, booster_path: str, scaler_path: str, discriminator_path: str) -> None:
        if booster_path is None or booster_path == "" or not os.path.exists(booster_path):
            logging.error(
                f'Booster path `{booster_path}` is empty string or does not exist.')
        if scaler_path is None or scaler_path == "" or not os.path.exists(scaler_path):
            logging.error(
                f'Scaler path `{scaler_path}` is empty string or does not exist.')
        if discriminator_path is None or discriminator_path == "" or not os.path.exists(discriminator_path):
            logging.error(
                f'Discriminator path `{discriminator_path}` is empty string or does not exist.')

        self._booster: xgb.Booster = xgb.Booster()
        self._scaler: Pipeline = None
        self._discriminator = DiscriminatorPredictor(discriminator_path)

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

        self._feature_vector = None
        self._model_data_labels = None

    def get_feature_vector(self) -> pd.DataFrame:
        return self._feature_vector

    def get_model_data_labels(self) -> list:
        return self._model_data_labels

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

    def _get_discriminator_features(self, extracted_predictions: np.ndarray, model_data_labels: list, feature_vector: pd.DataFrame):

        poi_features = {f"POI{i}": pd.DataFrame() for i in range(1, 7)}

        # Define neighbor mapping to include candidate indices from neighboring POIs.
        neighbors_map = {
            "POI4": ["POI5"],  # For POI4, include POI5's candidates.
            "POI5": ["POI6"],  # For POI5, include POI6's candidates.
            "POI6": ["POI5"]   # For POI6, include POI5's candidates.
        }

        # Process each POI.
        for i, poi in enumerate(poi_features.keys()):
            diff_values = feature_vector["Difference"].values
            diss_values = feature_vector["Dissipation"].values
            rf_values = feature_vector["Resonance_Frequency"].values
            n_features = len(diff_values)
            x_full = np.arange(n_features)
            # Clean any infinite or NaN values.
            for i, poi in enumerate(poi_features.keys()):
                # --- Build the candidate index list with source labeling ---
                # Get candidate indices from the POI and its neighbors.
                self_candidates = extracted_predictions.get(
                    poi, {}).get("indices", []).copy()

                # Create dictionary mapping candidate index -> source label.
                candidate_sources = {}
                for cand in self_candidates:
                    candidate_sources[cand] = "self"

                # Include the reference candidate from model_data_labels (assumed "self").
                ref = model_data_labels[i]
                if ref not in candidate_sources:
                    candidate_sources[ref] = "self"

                # Create sorted candidate indices.
                combined_indices = sorted(candidate_sources.keys())
                # Convert to a NumPy array and ensure all indices fall within the feature vector range.
                x_candidates = np.array(combined_indices)
                x_candidates = np.clip(x_candidates, 0, n_features - 1)

                # --- Interpolate feature values on the full x-axis ---
                # np.interp automatically extends values outside xp by using the first/last fp values.
                diff_interp = np.interp(
                    x_full, x_candidates, diff_values[x_candidates])
                diss_interp = np.interp(
                    x_full, x_candidates, diss_values[x_candidates])
                rf_interp = np.interp(x_full, x_candidates,
                                      rf_values[x_candidates])

                # Build the discriminator DataFrame with length equal to the full feature vector.
                discrim_vector = pd.DataFrame({
                    "Difference": diff_interp,
                    "Dissipation": diss_interp,
                    "Resonance_Frequency": rf_interp,
                })

                discrim_vector.replace([np.inf, -np.inf], np.nan, inplace=True)
                discrim_vector.fillna(0, inplace=True)

                poi_features[poi] = pd.concat(
                    [poi_features[poi], discrim_vector], ignore_index=True)
        return poi_features

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
        self._model_data_labels = model_data_labels
        file_buffer = self._reset_file_buffer(file_buffer=file_buffer)
        feature_vector = QDataProcessor.process_data(
            file_buffer=file_buffer, live=True)
        self._feature_vector = feature_vector
        transformed_feature_vector = self._scaler.transform(
            feature_vector.values)
        ddata = xgb.DMatrix(transformed_feature_vector)
        predicted_probabilites = self._booster.predict(ddata)
        extracted_predictions = self._extract_predictions(
            predicted_probabilites, model_data_labels)
        # discrim_features = self._get_discriminator_features(
        #     extracted_predictions=extracted_predictions, model_data_labels=model_data_labels, feature_vector=feature_vector)
        # poi_1_discrim = self._discriminator.predict(
        #     discrim_features, poi="POI1")
        # poi_2_discrim = self._discriminator.predict(
        #     discrim_features, poi="POI2")
        # poi_3_discrim = self._discriminator.predict(
        #     discrim_features, poi="POI3")
        # poi_4_discrim = self._discriminator.predict(
        #     discrim_features, poi="POI4")
        # poi_5_discrim = self._discriminator.predict(
        #     discrim_features, poi="POI5")
        # poi_6_discrim = self._discriminator.predict(
        #     discrim_features, poi="POI6")
        # logging.info(f'POI1: {np.argmax(poi_1_discrim)}')
        # discrim_pred_1 = np.argmax(poi_1_discrim)

        # logging.info(f'POI2: {np.argmax(poi_2_discrim)}')
        # discrim_pred_2 = np.argmax(poi_2_discrim)

        # logging.info(f'POI3: {np.argmax(poi_3_discrim)}')
        # discrim_pred_3 = np.argmax(poi_3_discrim)

        # logging.info(f'POI4: {np.argmax(poi_4_discrim)}')
        # discrim_pred_4 = np.argmax(poi_4_discrim)

        # logging.info(f'POI5: {np.argmax(poi_5_discrim)}')
        # discrim_pred_5 = np.argmax(poi_5_discrim)

        # logging.info(f'POI6: {np.argmax(poi_6_discrim)}')
        # discrim_pred_6 = np.argmax(poi_6_discrim)

        # --- Plotting Section (Optional) ---
        if actual_poi_indices is not None:
            plt.figure(figsize=(14, 6))
            dissipation = feature_vector['Dissipation_smooth'].values
            poi_colors = {1: 'pink', 2: 'blue', 3: 'green',
                          4: 'orange', 5: 'purple', 6: 'cyan'}

            # --- Plot 1: Model Data Labels ---
            plt.subplot(1, 2, 1)
            plt.plot(dissipation, color='grey')
            plt.scatter(model_data_labels,
                        dissipation[model_data_labels])
            plt.scatter(actual_poi_indices,
                        dissipation[actual_poi_indices], marker='*', color='red')
            plt.xlabel("Sample Index")
            plt.ylabel("Dissipation")
            plt.title("Dissipation vs Index (Model Data Labels)")
            plt.clim(-0.5, 5.5)

            # --- Plot 2: Filtered Labels ---
            plt.subplot(1, 2, 2)
            plt.plot(dissipation, color='grey')
            for poi in range(1, 7):
                key = f"POI{poi}"
                unfiltered_poi_data = extracted_predictions.get(key, {})
                unfiltered_indices = unfiltered_poi_data.get("indices", [])
                if unfiltered_indices:
                    indices_arr = np.array(unfiltered_indices)
                    y_vals = dissipation[indices_arr]
                    # Plot all predictions with a lower alpha
                    plt.scatter(indices_arr, y_vals, color=poi_colors[poi],
                                marker='x', alpha=0.5, label=f'POI {poi}')
                    # Highlight the highest confidence guess (first in the sorted list)
                    highest_index = indices_arr[0]
                    plt.scatter(highest_index, dissipation[highest_index],
                                color=poi_colors[poi], marker='x', s=100,
                                alpha=1.0, edgecolor='black', linewidth=1.5)
                filtered_poi_data = extracted_predictions.get(key, {})
                filtered_indices = filtered_poi_data.get("indices", [])
                if filtered_indices:
                    indices_arr = np.array(filtered_indices)
                    y_vals = dissipation[indices_arr]
                    # Plot all predictions with a lower alpha
                    plt.scatter(indices_arr, y_vals, color=poi_colors[poi],
                                marker='o', alpha=0.5, label=f'POI {poi}')
                    # Highlight the highest confidence guess (first in the sorted list)
                    highest_index = indices_arr[0]
                    plt.scatter(highest_index, dissipation[highest_index],
                                color=poi_colors[poi], marker='o', s=100,
                                alpha=1.0, edgecolor='black', linewidth=1.5)
            # plt.axvline(discrim_pred_1, color='pink')
            # plt.axvline(discrim_pred_2, color='blue')
            # plt.axvline(discrim_pred_3, color='green')
            # plt.axvline(discrim_pred_4, color='yellow')
            # plt.axvline(discrim_pred_5, color='purple')
            # plt.axvline(discrim_pred_6, color='cyan')

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
    discriminator_path = os.path.join(
        "QModel", "SavedModels", "qmodel_v2", "point_descriminator.pkl")
    test_dir = os.path.join('content', 'static', 'valid')
    test_content = QDataProcessor.load_content(data_dir=test_dir)
    qmp = QModelPredictor(booster_path=booster_path,
                          scaler_path=scaler_path, discriminator_path=discriminator_path)
    import random
    random.shuffle(test_content)
    for i, (data_file, poi_file) in enumerate(test_content):
        logging.info(f"Predicting on data file `{data_file}`.")
        poi_indices = pd.read_csv(poi_file, header=None)
        logging.info(poi_indices)
        qmp.predict(file_buffer=data_file,
                    actual_poi_indices=poi_indices.values)
