from typing import List, Tuple, Optional
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import logging
from collections import Counter
from imblearn.over_sampling import SMOTE
import optuna
import pickle
from q_model_data_processor import QDataProcessor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
RANDOM_STATE = 42
MULTICLASS_OBJECTIVES = ['multi:softmax', 'multi:softprob']
MULTICLASS_EVAL_METRICS = ['merror', 'mlogloss', 'auc', 'aucpr']
TRAIN_PATH = os.path.join("content", "static", 'train')
TEST_PATH = os.path.join("content", "static", "test")
VALID_PATH = os.path.join("content", "static", "valid")
TRAIN_NUM = np.inf
TEST_VALID_NUM = np.inf


class QModelTrainer:
    def __init__(self,
                 classes: List[str],
                 regen_data: bool = False,
                 training_directory: str = TRAIN_PATH,
                 validation_directory: str = VALID_PATH,
                 test_directory: str = TEST_PATH,
                 cache_directory: str = os.path.join("cache", "q_model")):
        # 1) remember full list, then pick only up through POI4 if we're regenerating
        self.original_classes = classes
        self.regen_data = regen_data
        if self.regen_data:
            # drop POI5 and POI6
            self.classes = classes[:6]
        else:
            self.classes = classes

        # 2) build params AFTER classes is set
        self.params = self.build_params()
        self.booster: xgb.Booster = None

        # 3) label mapping uses reduced class list
        self.label_to_index = {
            str(label): idx for idx, label in enumerate(self.classes)}

        self.train_content = None
        self.test_content = None
        self.validation_content = None
        self.train_datasets = None
        self.test_datasets = None
        self.validation_datasets = None

        # Cache directory for persisting DMatrices, scaler and test slices
        self.cache_dir = cache_directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # File path for persisting the scaler
        self.scaler_file = os.path.join(
            "QModel", "SavedModels", "pf", "pf_qmodel_scaler_ch2.pkl")
        self.scaler: Optional[Pipeline] = None

        # In-memory caches
        self.cached_train_dmatrix: Optional[xgb.DMatrix] = None
        self.cached_val_dmatrix: Optional[xgb.DMatrix] = None
        self.cached_test_datasets: Optional[List] = None

        # Internal flags so that regeneration happens only once
        self._data_regenerated = False
        self._test_data_regenerated = False

        self.live_plot_enabled = False
        self.regen_data = regen_data
        if self.regen_data:
            self.load_datasets(training_directory=training_directory,
                               test_directory=test_directory,
                               validation_directory=validation_directory)
        else:
            # Attempt to load an existing scaler if available.
            if os.path.exists(self.scaler_file):
                self._load_scaler()

    def toggle_live_plot(self, enabled: bool = True):
        """Toggle live plotting on or off."""
        self.live_plot_enabled = enabled

    def _trim_dataset_at_random(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cut off `df` at a random index between the last POI=4 and the first POI=5,
        then drop any remaining POI>=5 so only [0–4] remain.
        """
        poi = df["POI"]
        idx4 = df.index[poi == 5].tolist()
        idx5 = df.index[poi == 6].tolist()

        if idx4 and idx5 and min(idx5) > max(idx4):
            # pick a random cut between end of POI4 and start of POI5
            cut = random.randint(max(idx4) + 1, min(idx5))
            df = df.iloc[:cut]

        # now drop any POI5 or POI6
        df = df[df["POI"] < 6].reset_index(drop=True)
        return df

    def build_params(self) -> dict:
        """XGBoost params must reflect new num_class."""
        return {
            "objective": "binary:logistic",
            "learning_rate": 0.1,
            "eval_metric": MULTICLASS_EVAL_METRICS[2],
            "tree_method": "hist",
            "device": "cuda",
            "num_class": len(self.classes),
        }
        return params

    def load_datasets(self, training_directory: str, test_directory: str, validation_directory: str):
        """Load training and test/validation datasets."""
        self.train_content, _ = QDataProcessor.load_balanced_content(
            training_directory, num_datasets=TRAIN_NUM, opt=True)
        self.test_content, _ = QDataProcessor.load_balanced_content(
            test_directory, num_datasets=TEST_VALID_NUM, opt=True)
        self.validation_content, _ = QDataProcessor.load_balanced_content(
            validation_directory, num_datasets=TEST_VALID_NUM, opt=True)
        logging.info("Datasets loaded.")

    def _save_scaler(self):
        """Persist the fitted scaler pipeline to disk."""
        with open(self.scaler_file, "wb") as f:
            pickle.dump(self.scaler, f)
        logging.info("Scaler saved to disk.")

    def _load_scaler(self):
        """Load the scaler pipeline from disk."""
        with open(self.scaler_file, "rb") as f:
            self.scaler = pickle.load(f)
        logging.info("Scaler loaded from disk.")

    def get_data(self, data_file: str, poi_file: str, live: bool = False):
        try:
            data_df = QDataProcessor.process_data(data_file, live=True)
            poi_df = QDataProcessor.process_poi(
                poi_file, length_df=len(data_df))
            data_df['POI'] = poi_df
            return data_df
        except FileNotFoundError as e:
            logging.error("POI file not found.")
            return None

    def build_dataset(self, content, live=False):
        all_datasets = []
        for i, (data_file, poi_file) in enumerate(content):
            logging.info(f"[{i+1}] Processing dataset {data_file}")
            dataset = self.get_data(
                data_file=data_file, poi_file=poi_file, live=live)
            if dataset is not None:
                # only trim if regen_data=True
                if self.regen_data:
                    dataset = self._trim_dataset_at_random(dataset)
                all_datasets.append(dataset)

        random.shuffle(all_datasets)
        return all_datasets

    def _extract_features(self, datasets: list) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for df in datasets:
            y_list.append(df['POI'].values)
            X_list.append(df.drop(columns=['POI']).values)
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        return X, y

    def _balance_data(self, X: np.ndarray, y: np.ndarray, dataset_name: str = "dataset") -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to balance the dataset and log class distributions."""
        initial_distribution = Counter(y)
        logging.info(
            f"Class distribution before SMOTE ({dataset_name}): {initial_distribution}")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X, y)
        balanced_distribution = Counter(y_res)
        logging.info(
            f"Class distribution after SMOTE ({dataset_name}): {balanced_distribution}")
        return X_res, y_res

    def _create_dmatrix(self, X: np.ndarray, y: np.ndarray) -> xgb.DMatrix:
        """Create an XGBoost DMatrix from features and labels."""
        return xgb.DMatrix(X, label=y)

    def _prepare_dmatrices(self) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        """Prepare and cache DMatrices for training and validation, using the regeneration flag only once."""
        train_cache_path = os.path.join(self.cache_dir, "train_dmatrix.cache")
        val_cache_path = os.path.join(self.cache_dir, "val_dmatrix.cache")
        # Use regeneration only once if regen_data is True and not already regenerated.
        regenerate = self.regen_data and not self._data_regenerated

        # Training DMatrix
        if not regenerate and os.path.exists(train_cache_path):
            try:
                self.cached_train_dmatrix = xgb.DMatrix(train_cache_path)
                logging.info("Loaded cached training DMatrix from disk.")
            except Exception as e:
                logging.error(f"Error loading cached training DMatrix: {e}")
                self.cached_train_dmatrix = None

        if self.cached_train_dmatrix is None or regenerate:
            if self.train_datasets is None or regenerate:
                self.train_datasets = self.build_dataset(self.train_content)
            X_train, y_train = self._extract_features(self.train_datasets)
            X_train_res, y_train_res = self._balance_data(
                X_train, y_train, dataset_name="train")

            # Fit or load the scaler as needed.
            if regenerate or not os.path.exists(self.scaler_file):
                self.scaler = Pipeline([
                    ('standard', StandardScaler()),
                    ('minmax', MinMaxScaler(feature_range=(0, 1)))
                ])
                X_train_scaled = self.scaler.fit_transform(X_train_res)
                self._save_scaler()
            else:
                self._load_scaler()
                X_train_scaled = self.scaler.transform(X_train_res)

            dtrain = self._create_dmatrix(X_train_scaled, y_train_res)
            dtrain.save_binary(train_cache_path)
            self.cached_train_dmatrix = xgb.DMatrix(train_cache_path)
            logging.info("Created and cached training DMatrix to disk.")

        # Validation DMatrix
        if not regenerate and os.path.exists(val_cache_path):
            try:
                self.cached_val_dmatrix = xgb.DMatrix(val_cache_path)
                logging.info("Loaded cached validation DMatrix from disk.")
            except Exception as e:
                logging.error(f"Error loading cached validation DMatrix: {e}")
                self.cached_val_dmatrix = None

        if self.cached_val_dmatrix is None or regenerate:
            if self.validation_datasets is None or regenerate:
                self.validation_datasets = self.build_dataset(
                    self.validation_content)
            X_val, y_val = self._extract_features(self.validation_datasets)
            X_val_res, y_val_res = self._balance_data(
                X_val, y_val, dataset_name="validation")

            # Use the same scaler as in training.
            if self.scaler is None:
                self._load_scaler()
            X_val_scaled = self.scaler.transform(X_val_res)

            dval = self._create_dmatrix(X_val_scaled, y_val_res)
            dval.save_binary(val_cache_path)
            self.cached_val_dmatrix = xgb.DMatrix(val_cache_path)
            logging.info("Created and cached validation DMatrix to disk.")

        # Mark that data regeneration has been performed once.
        self._data_regenerated = True
        return self.cached_train_dmatrix, self.cached_val_dmatrix

    def train(self):
        """Train the XGBoost model using cached training/validation DMatrices with early stopping."""
        logging.info("Starting training...")
        dtrain, dval = self._prepare_dmatrices()

        # Dynamically set the evaluation metric
        # If self.problem_type is defined, you could branch based on that
        if hasattr(self, "problem_type") and self.problem_type == "multiclass":
            eval_metric = self.params.get("eval_metric", "mlogloss")
        else:
            eval_metric = self.params.get("eval_metric", "rmse")

        total_rounds = 50
        early_stopping_rounds = 5
        best_val_loss = float("inf")
        no_improvement = 0
        best_round = 0
        best_booster = None
        losses = []

        # Prepare live plotting if enabled
        if self.live_plot_enabled:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))

        booster = self.booster
        for round in range(total_rounds):
            evals_result = {}
            booster = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params.get('n_boost_round', 10),
                evals=[(dtrain, "train"), (dval, "val")],
                evals_result=evals_result,
                xgb_model=booster
            )

            # Use the dynamic evaluation metric to get losses
            train_loss = evals_result["train"][eval_metric][0]
            val_loss = evals_result["val"][eval_metric][0]
            losses.append((train_loss, val_loss))
            logging.info(
                f"Round {round+1}: train {eval_metric}={train_loss}, val {eval_metric}={val_loss}")

            # Early stopping logic based on validation loss improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_round = round
                best_booster = booster
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= early_stopping_rounds:
                logging.info(
                    f"Early stopping triggered after {early_stopping_rounds} rounds without improvement.")
                break

            # Update plot if live plotting is enabled
            if self.live_plot_enabled:
                ax.clear()
                rounds_range = list(range(1, round + 2))
                train_losses, val_losses = zip(*losses)
                ax.plot(rounds_range, train_losses, label="Train Loss")
                ax.plot(rounds_range, val_losses, label="Val Loss")
                ax.legend()
                ax.set_xlabel("Boosting Round")
                ax.set_ylabel(eval_metric)
                ax.set_title("Training vs. Validation Loss")
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)

        if best_booster is not None:
            self.booster = best_booster
            logging.info(
                f"Training completed. Best round: {best_round+1} with val {eval_metric}={best_val_loss}")
        else:
            self.booster = booster
            logging.info("Training completed without improvement tracking.")

        if self.live_plot_enabled:
            plt.ioff()
            plt.show()

    def test(self, booster_path: str):
        # Load booster if it hasn't been loaded yet.
        if self.booster is None:
            self.booster = xgb.Booster()
            self.booster.load_model(booster_path)

        # Ensure the scaler is loaded for test transformations.
        if self.scaler is None:
            if os.path.exists(self.scaler_file):
                self._load_scaler()
            else:
                logging.error(
                    "Scaler not available. Please train the model first.")
                return

        test_cache_path = os.path.join(self.cache_dir, "test_dmatrix.pkl")
        regenerate = self.regen_data and not self._test_data_regenerated

        # Attempt to load cached test datasets if available and regeneration is not required.
        if not regenerate and os.path.exists(test_cache_path):
            try:
                with open(test_cache_path, "rb") as f:
                    self.cached_test_datasets = pickle.load(f)
                logging.info("Loaded cached test datasets from disk.")
            except Exception as e:
                logging.error(f"Error loading cached test datasets: {e}")
                self.cached_test_datasets = None

        # If no cached data or regeneration is needed, build and cache the test dataset.
        if self.cached_test_datasets is None or regenerate:
            test_datasets = self.build_dataset(self.test_content, live=False)
            if not test_datasets:
                logging.warning("No test datasets found.")
                return
            self.cached_test_datasets = test_datasets
            with open(test_cache_path, "wb") as f:
                pickle.dump(test_datasets, f)
            logging.info("Cached test datasets to disk.")

        # Mark that test slice regeneration has been completed.
        self._test_data_regenerated = True
        test_slices = self.cached_test_datasets

        # Set up matplotlib live plotting: two vertically arranged subplots.
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Define specific colors for POI predictions 1-6.
        poi_colors = {1: 'red', 2: 'blue', 3: 'green',
                      4: 'orange', 5: 'purple', 6: 'cyan'}

        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        dataset_accuracies = []

        # Loop through each test slice and update plots live.
        for slice_index, df in enumerate(test_slices):
            # Separate out the features and labels.
            print(df)
            y_test = df['POI']
            X_test = df.drop(columns=['POI']).values

            # Transform test data using the persisted scaler.
            X_test_transformed = self.scaler.transform(X_test)
            dtest = xgb.DMatrix(X_test_transformed)
            pred_probs = self.booster.predict(dtest)

            # Adapt prediction handling for multiclass.
            if pred_probs.ndim == 2:
                predictions = np.argmax(pred_probs, axis=1)
            else:
                predictions = pred_probs.astype(int)

            # Compute accuracy for this slice.
            correct = np.sum(predictions == y_test)
            total_correct += correct
            total_samples += len(y_test)
            slice_acc = correct / len(y_test)
            dataset_accuracies.append(slice_acc)

            # Store predictions and true labels.
            all_predictions.extend(predictions.tolist() if hasattr(
                predictions, 'tolist') else predictions)
            all_labels.extend(y_test.tolist() if hasattr(
                y_test, 'tolist') else y_test)

            # ----------------- Update the Upper Plot -----------------
            # Clear the current axis.
            ax1.clear()

            # Plot the Dissipation curve (assumes your df has a 'Dissipation' column).
            x_axis = np.arange(len(df))
            ax1.plot(x_axis, df['Dissipation'],
                     color='black', label='Dissipation')

            # Overlay predicted POI lines (only for POI values 1-6).
            for poi in range(1, 7):
                indices = np.where(predictions == poi)[0]
                if len(indices) > 0:
                    # Get corresponding Dissipation values for these indices.
                    y_vals = df['Dissipation'].values[indices]
                    # Plot a line with markers.
                    ax1.scatter(indices, y_vals, color=poi_colors[poi],
                                marker='o', linestyle='-',
                                label=f'POI {poi}')
            ax1.set_title("Dissipation Curve with Predicted POI (1-6)")
            ax1.set_xlabel("Sample Index")
            ax1.set_ylabel("Dissipation")
            ax1.legend()

            # ----------------- Update the Lower Plot -----------------
            ax2.clear()
            ax2.plot(range(len(dataset_accuracies)), dataset_accuracies,
                     marker='o', linestyle='-')
            ax2.set_ylim(0, 1)
            ax2.set_title("Prediction Accuracy per Test Slice")
            ax2.set_xlabel("Test Slice")
            ax2.set_ylabel("Accuracy")
            ax2.grid(True)

            # Pause briefly to update the live plot.
            plt.pause(1)

        overall_accuracy = total_correct / total_samples
        logging.info(f"Test accuracy: {overall_accuracy * 100:.2f}%")
        from sklearn.metrics import classification_report, confusion_matrix
        logging.info(
            f"Classification Report:\n{classification_report(all_labels, all_predictions)}")
        logging.info(
            f"Confusion Matrix:\n{confusion_matrix(all_labels, all_predictions)}")

        # Finalize the plots.
        plt.ioff()
        plt.show()

        return overall_accuracy

    def tune(self, n_trials: int = 50, num_boost_round: int = 50, early_stopping_rounds: int = 10):
        """Tune model hyperparameters using Optuna with cached DMatrices."""
        dtrain, dval = self._prepare_dmatrices()
        logging.info("Using cached DMatrices for tuning.")

        def objective(trial: optuna.Trial):
            trial_params = self.params.copy()
            trial_params["learning_rate"] = trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True)
            trial_params["gamma"] = trial.suggest_float("gamma", 0, 10)
            trial_params["eta"] = trial.suggest_float("eta", 0.01, 0.3)
            trial_params["reg_alpha"] = trial.suggest_float("reg_alpha", 0, 10)
            trial_params["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree", 0.5, 1.0)
            trial_params["colsample_bynode"] = trial.suggest_float(
                "colsample_bynode", 0.5, 1.0)
            trial_params["colsample_bylevel"] = trial.suggest_float(
                "colsample_bylevel", 0.5, 1.0)
            trial_params["min_child_weight"] = trial.suggest_float(
                "min_child_weight", 1, 10)
            trial_params["max_delta_step"] = trial.suggest_float(
                "max_delta_step", 0, 10)
            trial_params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            trial_params["subsample"] = trial.suggest_float(
                "subsample", 0.5, 1.0)
            trial_n_boost_round = trial.suggest_int(
                "n_boost_round", 10, num_boost_round)

            booster = xgb.train(
                trial_params,
                dtrain,
                num_boost_round=trial_n_boost_round,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            optimal_rounds = booster.best_iteration
            trial.set_user_attr("optimal_boost_rounds", optimal_rounds)
            return booster.best_score
        if self.params.get("eval_metric") == "auc" or self.params.get("eval_metric") == "aucpr":
            direction = "maximize"
        else:
            direction = "minimize"
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)

        logging.info(f"Best trial: {study.best_trial.number}")
        logging.info(f"Best accuracy: {study.best_trial.value:.4f}")
        logging.info(f"Best parameters: {study.best_trial.params}")

        optimal_boost_rounds = study.best_trial.user_attrs.get(
            "optimal_boost_rounds")
        logging.info(
            f"Optimal boosting rounds determined: {optimal_boost_rounds}")

    def save_model(self, save_path: str):
        """Save the trained booster model."""
        if self.booster is None:
            logging.error("No booster model available to save.")
            return
        self.booster.save_model(save_path)
        logging.info(f"Model saved to {save_path}")

    def search(self, num_boost_round: int = 10):
        """Search for the best combination of objective and evaluation metric using cached DMatrices."""
        dtrain, dval = self._prepare_dmatrices()
        logging.info("Using cached DMatrices for objective/metric search.")

        best_accuracy = -1.0
        best_objective = None
        best_eval_metric = None
        results = []

        for obj in MULTICLASS_OBJECTIVES:
            for eval_metric in MULTICLASS_EVAL_METRICS:
                params_copy = self.params.copy()
                params_copy["objective"] = obj
                params_copy["eval_metric"] = eval_metric

                try:
                    booster = xgb.train(params_copy, dtrain,
                                        num_boost_round=num_boost_round)
                except Exception as e:
                    logging.error(
                        f"Error training with objective {obj} and eval_metric {eval_metric}: {e}")
                    continue

                preds = booster.predict(dval)
                if preds.ndim == 1:
                    pred_labels = (preds > 0.5).astype(int)
                else:
                    pred_labels = np.argmax(preds, axis=1)

                y_val = dval.get_label()
                accuracy = np.mean(pred_labels == y_val)
                results.append({
                    "objective": obj,
                    "eval_metric": eval_metric,
                    "accuracy": accuracy
                })
                logging.info(
                    f"Tested objective: {obj}, eval_metric: {eval_metric}, accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_objective = obj
                    best_eval_metric = eval_metric

        logging.info(
            f"Best combination: objective: {best_objective}, eval_metric: {best_eval_metric}, accuracy: {best_accuracy:.4f}")
        self.params["objective"] = best_objective
        self.params["eval_metric"] = best_eval_metric
        self.objective = best_objective
        self.eval_metric = best_eval_metric
        return best_objective, best_eval_metric, best_accuracy, results


if __name__ == "__main__":
    TRAINING = True
    booster_path = os.path.join(
        "QModel", "SavedModels", "pf", "pf_model_v2_ch2.json")
    if TRAINING:
        ft = QModelTrainer(classes=[
            "NO_POI", "POI1", "POI2", "POI3", "POI4", "POI5"], regen_data=True)
        ft.toggle_live_plot(True)  # Enable live plotting
        ft.search(num_boost_round=10)
        ft.tune()
        ft.train()
        ft.save_model(save_path=booster_path)
        ft.test(booster_path=booster_path)
