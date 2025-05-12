import os
import pickle
from typing import Union, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pf_data_processor_v2 import PFDataProcessor


class PFPredictor:
    """
    Standalone predictor for PF runs. Handles loading model assets, preprocessing,
    and outputting POI count predictions for individual runs.
    """

    def __init__(self, model_dir: str) -> None:
        """
        Initialize PFPredictor by loading the scaler and booster from storage.

        Args:
            model_dir: Path to directory containing 'pf_scaler.pkl' and 'pf_booster.json'.
        """
        self.model_dir = model_dir
        self.scaler = self._load_scaler()
        self.booster = self._load_booster()

    def _load_scaler(self) -> any:
        """
        Load the scaler object from disk.

        Returns:
            Fitted scaler pipeline.

        Raises:
            FileNotFoundError: If scaler file is missing.
        """
        scaler_path = os.path.join(self.model_dir, 'pf_scaler.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)

    def _load_booster(self) -> xgb.Booster:
        """
        Load the XGBoost booster from disk.

        Returns:
            Loaded xgboost.Booster instance.

        Raises:
            FileNotFoundError: If booster file is missing.
        """
        booster_path = os.path.join(self.model_dir, 'pf_booster.json')
        if not os.path.exists(booster_path):
            raise FileNotFoundError(f"Booster file not found: {booster_path}")
        booster = xgb.Booster()
        booster.load_model(booster_path)
        return booster

    def predict(
        self,
        data: Union[str, pd.DataFrame],
        detected_poi1: Optional[int] = None
    ) -> int:
        """
        Predict the number of POIs for a single run.

        Args:
            data: Path to CSV file or a DataFrame containing the run data.
            detected_poi1: Optional simulated or known POI1 index for feature generation.

        Returns:
            Predicted number of POIs as an integer.
        """
        # Load DataFrame if path provided
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        plt.figure()
        plt.plot(df['Dissipation'])
        plt.show()
        # Generate features using the same processor used during training
        features = PFDataProcessor.generate_features(
            dataframe=df,
            sampling_rate=1.0,
            detected_poi1=detected_poi1
        )

        # Scale features
        X_scaled = self.scaler.transform(features)

        # Create DMatrix and predict
        ddata = xgb.DMatrix(X_scaled)
        probs = self.booster.predict(ddata)
        pred = np.argmax(probs, axis=1)[0]
        return pred


if __name__ == '__main__':
    # Example usage
    MODEL_DIR = os.path.join('QModel', 'SavedModels', 'pf')
    predictor = PFPredictor(model_dir=MODEL_DIR)

    # Path to a new run CSV
    run_csv = r'C:\Users\paulm\dev\QATCH-ML\content\static\test\02398\M240913W11_10CP_I9_3rd.csv'
    poi_csv = r'C:\Users\paulm\dev\QATCH-ML\content\static\test\02398\M240913W11_10CP_I9_3rd_poi.csv'
    poi_df = pd.read_csv(poi_csv, header=None).values
    predicted_pois = predictor.predict(run_csv, detected_poi1=poi_df[0])
    print(f"Predicted number of POIs: {predicted_pois}")
