import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter
from imblearn.over_sampling import SMOTE
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline


class PointDescriminatorTrainer:
    def __init__(self):
        self._poi_features = {"POI1": None, "POI2": None,
                              "POI3": None, "POI4": None, "POI5": None, "POI6": None}
        # Define a scaler that combines StandardScaler and MinMaxScaler
        self.scaler = Pipeline([
            ('standard', StandardScaler()),
            ('minmax', MinMaxScaler(feature_range=(0, 1)))
        ])
        self._models = {}
        self._data_splits = {}
        self._load_data(os.path.join("cache", "q_model"))

    def _load_data(self, cache_path: str):
        for poi in self._poi_features.keys():
            file_path = os.path.join(cache_path, f"{poi}_dataset.pkl")
            try:
                with open(file_path, "rb") as f:
                    data_loaded = pickle.load(f)
                    self._poi_features[poi] = data_loaded
                print(f"Loaded {poi} successfully.")
            except Exception as e:
                print(f"Error loading {poi}: {e}")

    def _balance_data(self, X: np.ndarray, y: np.ndarray):
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res

    def train(self):
        for poi, data in self._poi_features.items():
            if data is None:
                print(f"No data available for {poi}, skipping training.")
                continue
            if 'Target' not in data.columns:
                print(
                    f"Data for {poi} does not have a 'Target' column, skipping training.")
                continue

            # Separate features and target
            X = data.drop(columns=['Target'])
            y = data['Target']

            # Fit the scaler on the training data
            X_scaled = self.scaler.fit_transform(X)

            # Split the scaled data (80% train, 20% test)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            self._data_splits[poi] = {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }

            # Balance the training data using SMOTE
            X_train_bal, y_train_bal = self._balance_data(X_train, y_train)
            print(f"After balancing, class distribution for {poi}:", Counter(
                y_train_bal))

            # Train a simple model
            model = xgb.XGBClassifier(
                use_label_encoder=False, eval_metric='logloss', random_state=42)
            model.fit(X_train_bal, y_train_bal)
            self._models[poi] = model
            print(f"Trained initial model for {poi}.")

    def tune(self):
        param_grid = {
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        }
        for poi, splits in self._data_splits.items():
            print(f"Tuning hyperparameters for {poi}...")
            X_train = splits['X_train']
            y_train = splits['y_train']

            pipeline = ImbPipeline([
                ('scaler', self.scaler),  # Incorporate the scaler
                ('smote', SMOTE(random_state=42)),
                ('classifier', xgb.XGBClassifier(
                    use_label_encoder=False, eval_metric='logloss', random_state=42))
            ])

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring='accuracy',
                cv=3,
                verbose=1,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            self._models[poi] = best_pipeline
            print(f"Best parameters for {poi}: {grid_search.best_params_}")

    def test(self):
        for poi, splits in self._data_splits.items():
            X_test = splits['X_test']
            y_test = splits['y_test']
            model = self._models.get(poi)
            if model is None:
                print(f"No model available for {poi}, skipping testing.")
                continue
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            print(f"Test Accuracy for {poi}: {acc:.4f}")

    def save(self, model_path="models_scaler.pkl"):
        """
        Saves both the scaler and models to disk.
        """
        try:
            save_dict = {
                "scaler": self.scaler,
                "models": self._models
            }
            with open(model_path, "wb") as f:
                pickle.dump(save_dict, f)
            print(f"Models and scaler saved successfully to {model_path}.")
        except Exception as e:
            print(f"Error saving models and scaler: {e}")

    def load(self, model_path="models_scaler.pkl"):
        """
        Loads both the scaler and models from disk.
        """
        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.scaler = data.get("scaler")
                self._models = data.get("models", {})
            print(f"Models and scaler loaded successfully from {model_path}.")
        except Exception as e:
            print(f"Error loading models and scaler: {e}")

    def predict_new(self, new_data: pd.DataFrame, poi: str = "POI1"):
        """
        Scales new input data and uses the specified POI model for prediction.
        """
        if self.scaler is None:
            print("Scaler has not been loaded or fitted.")
            return None
        scaled_data = self.scaler.transform(new_data)
        model = self._models.get(poi)
        if model is None:
            print(f"No model available for {poi}.")
            return None
        predictions = model.predict(scaled_data)
        return predictions


if __name__ == "__main__":
    point_descrim = PointDescriminatorTrainer()
    point_descrim.train()
    point_descrim.test()
    point_descrim.save(os.path.join("QModel", "SavedModels",
                       "qmodel_v2", "point_descriminator.pkl"))
