import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


class DiscriminatorPredictor:
    """
    Loads saved models and scaler to generate predictions for new data.
    Uses the native XGBoost API for prediction.
    """

    def __init__(self, model_path="models_scaler.pkl"):
        self.scaler = None
        self._models = {}
        self.load(model_path)

    def load(self, model_path="models_scaler.pkl"):
        """
        Loads the scaler and trained models from disk.
        """
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.scaler = data.get("scaler")
                self._models = data.get("models", {})
            print(f"Models and scaler loaded successfully from {model_path}.")
        except Exception as e:
            print(f"Error loading models and scaler: {e}")

    def predict(self, new_data: dict, poi: str = "POI1"):
        """
        Scales new input data and uses the specified POI model to generate predictions.
        Converts the input data into a DMatrix for the native XGBoost prediction.
        """
        if self.scaler is None:
            print("Scaler has not been loaded or fitted.")
            return None
        data = new_data.get(poi)
        scaled_data = self.scaler.transform(data)
        booster = self._models.get(poi)
        if booster is None:
            print(f"No model available for {poi}.")
            return None
        ddata = xgb.DMatrix(scaled_data)
        y_prob = booster.predict(ddata)
        predictions = (y_prob > 0.5).astype(int)
        return y_prob
