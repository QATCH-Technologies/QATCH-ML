from typing import Union, IO, List, Dict
import xgboost as xgb
import numpy as np
import pickle
import logging
from q_model_data_processor import QDataProcessor
import pandas as pd
from collections import OrderedDict

from ModelData import ModelData
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class QModelPredictor:
    """
    Standalone predictor for sequential POI models.

    Loads all XGBoost boosters and scalers from a pickle file and
    generates sequential predictions on a given data buffer or file path.

    Returns a dictionary with keys 'POI1' to 'POI6', each mapping to
    an ordered dict of {index: confidence} sorted descending by confidence.
    """

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._boosters: Dict[int, xgb.Booster] = {}
        self._scalers: Dict[int, any] = {}
        self.load_models(model_path)

    def load_models(self, path: str) -> None:
        """
        Load boosters and scalers from the given pickle file.
        """
        logging.info(f"Loading models and scalers from {path}")
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._boosters = data.get('boosters', {})
            self._scalers = data.get('scalers', {})
            logging.info("Models and scalers loaded successfully.")
        except FileNotFoundError:
            logging.error(f"Model file not found at {path}")
            raise
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise

    def predict(self, data_source: Union[str, IO]) -> Dict[str, Dict[int, float]]:
        """
        Generates sequential POI predictions on the provided data.

        Args:
            data_source: File path or file-like buffer for the raw data CSV.

        Returns:
            Dict with keys 'POI1'..'POI6'. Each value is an ordered dict
            mapping possible indices to their confidence score (float),
            sorted in descending order of confidence.
        """
        # Load raw feature DataFrame
        df = QDataProcessor.process_data(data_source, live=True)
        n = len(df)
        results: Dict[str, Dict[int, float]] = {}

        for poi_idx in range(1, 7):
            # Build Prev_POI labels from previous predictions
            prev = np.zeros(n, dtype=int)
            if poi_idx > 1 and results:
                # assign blocks based on last predicted index
                preds_prev = list(results[f'POI{poi_idx-1}'].keys())[:1]
                last_pred = preds_prev[0] if preds_prev else 0
                prev[:last_pred] = poi_idx - 1
                # ensure no labels exceed allowed
                prev[prev > (poi_idx - 1)] = 0
            df['Prev_POI'] = prev

            # Prepare feature matrix
            X = df.values
            scaler = self._scalers.get(poi_idx)
            if scaler is None:
                raise ValueError(f"Scaler for POI{poi_idx} not loaded.")
            X_scaled = scaler.transform(X)

            # Predict probabilities
            booster = self._boosters.get(poi_idx)
            if booster is None:
                raise ValueError(f"Booster for POI{poi_idx} not loaded.")
            dmat = xgb.DMatrix(X_scaled)
            confidences = booster.predict(dmat)

            # Map each index to its confidence
            idxs = np.arange(n)
            sorted_idx = idxs[np.argsort(confidences)[::-1]]
            sorted_conf = confidences[sorted_idx]
            # Build ordered dict of index: confidence
            poi_key = f'POI{poi_idx}'
            ordered = OrderedDict()
            for idx, conf in zip(sorted_idx, sorted_conf):
                ordered[int(idx)] = float(conf)

            results[poi_key] = ordered
            logging.debug(
                f"POI{poi_idx} confidences computed ({len(ordered)} entries)")

        return results

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
