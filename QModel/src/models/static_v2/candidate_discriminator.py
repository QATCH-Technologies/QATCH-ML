import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from q_model_data_processor import QDataProcessor
from q_model_predictor import QModelPredictor
import logging
from sklearn.preprocessing import LabelEncoder


class DescriminatorTrainer:
    """
    Handles loading data, training models, testing models, tuning hyperparameters with Optuna,
    and saving the scaler and trained models. This version uses the native XGBoost API.
    """

    def __init__(self, cache_path=os.path.join("cache", "q_model")):
        self._poi_features = {"POI1": None, "POI2": None,
                              "POI3": None, "POI4": None, "POI5": None, "POI6": None}
        # Define a scaler that combines StandardScaler and MinMaxScaler.
        self.scaler = Pipeline([
            ('standard', StandardScaler()),
            ('minmax', MinMaxScaler(feature_range=(0, 1)))
        ])
        self._models = {}
        self._data_splits = {}
        self.cache_path = cache_path
        self._load_data()

    def _load_data(self):
        """Loads cached data for each Point of Interest (POI) from disk."""
        for poi in self._poi_features.keys():
            file_path = os.path.join(self.cache_path, f"{poi}_dataset.pkl")
            try:
                with open(file_path, "rb") as f:
                    data_loaded = pickle.load(f)
                    self._poi_features[poi] = data_loaded
                print(f"Loaded {poi} successfully.")
            except Exception as e:
                print(f"Error loading {poi}: {e}")

    def _balance_data(self, X: np.ndarray, y: np.ndarray):
        """
        Balances data using SMOTE.
        """
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res

    def train(self):
        """
        For each POI, scales the data, splits it into training and testing sets,
        balances the training data using SMOTE, and trains an XGBoost model
        using the native API.
        """
        # Default parameters for initial training.
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cuda",
            "seed": 42
        }
        num_boost_round = 100

        for poi, data in self._poi_features.items():
            if data is None:
                print(f"No data available for {poi}, skipping training.")
                continue
            if 'Target' not in data.columns:
                print(
                    f"Data for {poi} does not have a 'Target' column, skipping training.")
                continue

            # Separate features and target.
            X = data.drop(columns=['Target'])
            y = data['Target']
            # le = LabelEncoder()

            # Convert the "Candidate_Source" column from strings to numeric labels
            # data['Candidate_Source'] = le.fit_transform(
            #     data['Candidate_Source'])

            # Now separate features and target
            X = data.drop(columns=['Target'])
            y = data['Target']
            # Scale the features.
            X_scaled = self.scaler.fit_transform(X)

            # Split data into training and testing sets.
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            self._data_splits[poi] = {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }
            # Balance the training data using SMOTE.
            X_train_bal, y_train_bal = self._balance_data(X, y)
            print(
                f"After balancing, class distribution for {poi}: {Counter(y_train_bal)}")

            # Create a DMatrix for training.
            dtrain = xgb.DMatrix(X_train_bal, label=y_train_bal)

            # Train using the native API.
            booster = xgb.train(
                params, dtrain, num_boost_round=num_boost_round)
            self._models[poi] = booster
            print(f"Trained initial model for {poi}.")

    def tune(self, n_trials=10):
        """
        Uses Optuna to perform hyperparameter tuning for each POI's model.
        The tuning process uses xgb.cv for 3-fold cross-validation to evaluate each
        candidate parameter configuration.
        """
        for poi, splits in self._data_splits.items():
            print(f"Tuning hyperparameters for {poi} with Optuna...")
            X_train = splits['X_train']
            y_train = splits['y_train']

            # Balance the training data with SMOTE.
            X_train_bal, y_train_bal = self._balance_data(X_train, y_train)
            dtrain = xgb.DMatrix(X_train_bal, label=y_train_bal)

            def objective(trial):
                # Suggest hyperparameters.
                param = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "seed": 42,
                    "tree_method": "hist",
                    "device": "cuda",
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
                }
                # Suggest number of boosting rounds.
                n_estimators = trial.suggest_int("n_estimators", 50, 200)

                # Perform 3-fold cross-validation.
                cv_results = xgb.cv(
                    param,
                    dtrain,
                    num_boost_round=n_estimators,
                    nfold=3,
                    metrics='error',
                    seed=42,
                    verbose_eval=False
                )
                # Retrieve the last boosting round's test error.
                mean_error = cv_results['test-error-mean'].iloc[-1]
                mean_accuracy = 1 - mean_error  # maximize accuracy.
                return mean_accuracy

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_trial = study.best_trial
            print(
                f"Best trial for {poi}: Value: {best_trial.value}, Params: {best_trial.params}")

            # Refit model with best parameters.
            best_params = best_trial.params
            final_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "seed": 42,
                "max_depth": best_params["max_depth"],
                "learning_rate": best_params["learning_rate"]
            }
            booster = xgb.train(final_params, dtrain,
                                num_boost_round=best_params["n_estimators"])
            self._models[poi] = booster

    def test(self):
        """
        Tests each trained model on its corresponding test set and prints the accuracy.
        Uses the native API for prediction by converting the test data into a DMatrix.
        """
        for poi, splits in self._data_splits.items():
            X_test = splits['X_test']
            y_test = splits['y_test']
            booster = self._models.get(poi)
            if booster is None:
                print(f"No model available for {poi}, skipping testing.")
                continue
            # Create a DMatrix for testing.
            dtest = xgb.DMatrix(X_test)
            # Get prediction probabilities.
            y_prob = booster.predict(dtest)
            # Threshold at 0.5 to obtain binary class labels.
            predictions = (y_prob > 0.5).astype(int)
            acc = accuracy_score(y_test, predictions)
            print(f"Test Accuracy for {poi}: {acc:.4f}")

    def save(self, model_path="models_scaler.pkl"):
        """
        Saves the scaler and the trained models to disk.
        Note: XGBoost boosters can be saved using pickle, or via booster.save_model.
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
        scaled_data = self.scaler.transform(new_data)
        booster = self._models.get(poi)
        if booster is None:
            print(f"No model available for {poi}.")
            return None
        ddata = xgb.DMatrix(scaled_data)
        y_prob = booster.predict(ddata)
        predictions = (y_prob > 0.5).astype(int)
        return y_prob


def discriminator_dataset_gen(booster_path: str, scaler_path: str, training_directory: str, output_directory: str):
    training_content = QDataProcessor.load_content(data_dir=training_directory)
    qmp = QModelPredictor(
        booster_path,
        scaler_path,
        os.path.join("QModel", "SavedModels")
    )

    # Create POI features for POI1 to POI6 (POI4, POI5, and POI6 are later processed,
    # but here we initialize all, adjust if needed)
    poi_features = {f"POI{i}": pd.DataFrame() for i in range(1, 7)}

    from tqdm import tqdm
    for data_file, poi_file in tqdm(training_content, desc="< Processing Training Data >"):
        poi_indices = pd.read_csv(poi_file, header=None).values.flatten()
        predictions = qmp.predict(file_buffer=data_file)
        model_data_labels = qmp.get_model_data_labels()
        feature_vector = qmp.get_feature_vector()

        # Extract the full feature arrays.
        diff_values = feature_vector["Difference_smooth"].values
        diss_values = feature_vector["Dissipation_smooth"].values
        rf_values = feature_vector["Resonance_Frequency_smooth"].values

        # Create an x-axis that spans the entire length of the feature vector.
        n_features = len(diff_values)
        x_full = np.arange(n_features)

        for i, poi in enumerate(poi_features.keys()):
            # --- Build the candidate index list with source labeling ---
            # Get candidate indices from the POI and its neighbors.
            self_candidates = predictions.get(
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

            # --- Determine the target candidate ---
            # Initialize the Target column with 0.
            discrim_vector["Target"] = 0
            actual = poi_indices[i]

            discrim_vector.loc[actual, "Target"] = 1

            # --- Plot for visual verification ---
            # import matplotlib.pyplot as plt
            # x_axis = np.arange(len(discrim_vector))
            # plt.figure(figsize=(10, 6))
            # plt.plot(
            #     x_axis, discrim_vector["Dissipation"], label="Dissipation")
            # plt.axvline(x=actual, color='grey',
            #             linestyle='--', label="Target")
            # plt.xlabel("Uniform Candidate Index")
            # plt.ylabel("Feature Value")
            # plt.title("Discriminator Candidate Features with Source Labels")
            # plt.legend()
            # plt.show()
            discrim_vector.replace([np.inf, -np.inf], np.nan, inplace=True)
            discrim_vector.fillna(0, inplace=True)

            poi_features[poi] = pd.concat(
                [poi_features[poi], discrim_vector], ignore_index=True)

    # Save the datasets.
    for poi, df in poi_features.items():
        if not df.empty:
            output_file = os.path.join(output_directory, f"{poi}_dataset.pkl")
            df.to_pickle(output_file)
            logging.info(f"Dataset for {poi} saved to {output_file}")
        else:
            logging.warning(f"Dataset for {poi} is empty and was not saved.")


if __name__ == "__main__":
    booster_path = os.path.join(
        "QModel", "SavedModels", "qmodel_v2", "qmodel_v2.json")
    scaler_path = os.path.join(
        "QModel", "SavedModels", "qmodel_v2", "qmodel_scaler.pkl")
    test_dir = os.path.join('content', 'static', 'valid')
    discriminator_path = os.path.join(
        "QModel", "SavedModels", "qmodel_v2", "point_descriminator.pkl")
    discriminator_dataset_gen(booster_path, scaler_path, test_dir, os.path.join(
        "cache", "q_model"))
    # ----------- Training Phase -----------
    trainer = DescriminatorTrainer()
    trainer.train()
    trainer.tune(n_trials=50)
    # trainer.test()
    # Optionally, run hyperparameter tuning with Optuna.
    # For example, use 50 trials:

    trainer.save(discriminator_path)

    # ----------- Prediction Phase -----------
    # When you need to make new predictions, use the predictor class.
    # Example usage:
    # new_data = pd.read_csv("path_to_new_data.csv")  # Load or create new data as a DataFrame.
    # Wrap the new data in a dict with keys corresponding to POIs. For example:
    # new_data_dict = {"POI1": new_data}
    # predictor = DescriminatorPredictor(save_file)
    # predictions = predictor.predict(new_data_dict, poi="POI1")
    # print("Predictions:", predictions)
