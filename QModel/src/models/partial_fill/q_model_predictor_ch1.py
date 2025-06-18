import os
import pickle
from typing import Union
import pandas as pd
import numpy as np
import xgboost as xgb
from q_model_data_processor import QDataProcessor


class QModelPredictorCh1:
    """
    Standalone predictor for QModel, channel 1.
    Loads an XGBoost model and a preprocessing scaler pipeline from a directory,
    and provides a predict() method for new data.
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.scaler_path = os.path.join(
            self.model_dir, "pf_qmodel_scaler_ch1.pkl")
        self.model_path = os.path.join(self.model_dir, "pf_model_ch1.json")

        self._load_scaler()
        self._load_model()

    def _load_scaler(self):
        """Load the preprocessing scaler pipeline from disk."""
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def _load_model(self):
        """Load the XGBoost Booster model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.booster = xgb.Booster()
        self.booster.load_model(self.model_path)

    def predict(self, data: Union[pd.DataFrame, str, np.ndarray]) -> np.ndarray:
        """
        Predict POI labels for the given data.

        :param data: One of:
            - pandas.DataFrame: feature DataFrame (will drop 'POI' column if present)
            - str: path to a CSV file to load into DataFrame
            - numpy.ndarray: 2D array of features
        :return: numpy array of predicted integer labels
        """
        # Load DataFrame if path provided
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise TypeError(
                "data must be a DataFrame, CSV path, or numpy array")
        X = QDataProcessor.process_data(df)
        # Scale features
        X_scaled = self.scaler.transform(X)

        # Create DMatrix and predict
        dmat = xgb.DMatrix(X_scaled)
        preds = self.booster.predict(dmat)

        # Handle multiclass vs binary outputs
        if preds.ndim == 2:
            return np.argmax(preds, axis=1)
        else:
            return (preds > 0.5).astype(int)


if __name__ == "__main__":
    predictor = QModelPredictorCh1(
        r"C:\Users\QATCH\dev\QATCH-ML\QModel\SavedModels\pf")
    results = predictor.predict(
        "content/static/test/00005/M240625W10B_%30ETH_J4_3rd.csv")
    print(results)
