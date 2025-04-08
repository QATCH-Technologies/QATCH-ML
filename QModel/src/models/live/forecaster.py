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
from forecaster_data_processor import DataProcessor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

TEST_SIZE = 0.2
RANDOM_STATE = 42
DOWNSAMPLE_FACTOR = 5
SLICE_SIZE = 100

EVAL_METRIC = "mae"
OBJECTIVE = "binary:logistic"
DIRECTION = "minimize"
BINARY_OBJECTIVES = ['reg:squarederror', 'reg:logistic', 'reg:pseudohubererror', 'reg:absoluteerror',
                     'binary:logistic', 'binary:logitraw', 'binary:hinge',
                     'count:poisson', 'survival:cox', 'rank:pairwise', 'reg:tweedie']
BINARY_EVAL_METRICS = ['mae']
MULTICLASS_OBJECTIVES = ['multi:softmax', 'multi:softprob']
MULTICLASS_EVAL_METRICS = ['merror', 'mlogloss', 'auc', 'aucpr']
TRAIN_PATH = os.path.join("content", "live", 'train')
TEST_PATH = os.path.join("content", "live", "test")
VALID_PATH = os.path.join("content", "live", "valid")
TRAIN_NUM = np.inf
TEST_VALID_NUM = np.inf


class ForcasterTrainer:
    def __init__(self,
                 classes: list,
                 num_features: int,
                 regen_data: bool = False,
                 training_directory: str = TRAIN_PATH,
                 validation_directory: str = VALID_PATH,
                 test_directory: str = TEST_PATH,
                 cache_directory: str = os.path.join("cache", "forecaster")) -> None:
        if not isinstance(classes, list):
            raise TypeError("classes must be a list")
        if not isinstance(num_features, int):
            raise TypeError("num_features must be an integer")

        self.classes = classes
        self.num_features = num_features
        self.params = self.build_params()
        self.booster: xgb.Booster = None
        self.label_to_index = {
            str(label): idx for idx, label in enumerate(classes)}

        self.train_content = None
        self.test_content = None
        self.validation_content = None
        self.training_slices = None
        self.validation_slices = None
        self.test_slices = None  # temporary in-memory storage for test slices

        # Cache directory for persisting DMatrices, scaler and test slices
        self.cache_dir = cache_directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # File path for persisting the scaler
        self.scaler_file = os.path.join(self.cache_dir, "scaler.pkl")
        self.scaler: Optional[Pipeline] = None

        # In-memory caches
        self.cached_train_dmatrix: Optional[xgb.DMatrix] = None
        self.cached_val_dmatrix: Optional[xgb.DMatrix] = None
        self.cached_test_slices: Optional[List] = None

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

    def build_params(self) -> dict:
        params = {
            "objective": "binary:logistic",
            "learning_rate": 0.1,
            "eval_metric": EVAL_METRIC,
            "tree_method": "hist",
            "device": "cuda"
        }
        return params

    def load_datasets(self, training_directory: str, test_directory: str, validation_directory: str):
        """Load training and test/validation datasets."""
        self.train_content, _ = DataProcessor.load_balanced_content(
            training_directory, num_datasets=TRAIN_NUM, opt=True)
        self.test_content, _ = DataProcessor.load_balanced_content(
            test_directory, num_datasets=TEST_VALID_NUM, opt=True)
        self.validation_content, _ = DataProcessor.load_balanced_content(
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

    def get_slices(self, data_file: str, poi_file: str, slice_size: int = SLICE_SIZE):
        """Process a file and return data slices."""
        try:
            data_df = pd.read_csv(data_file)
            fill_df = DataProcessor.process_fill(
                poi_file, length_df=len(data_df))
            data_df['Fill'] = fill_df
            data_df = data_df[::DOWNSAMPLE_FACTOR]
            total_rows = len(data_df)
            slices = []
            for i in range(0, total_rows, slice_size):
                result = DataProcessor.preprocess_data(
                    data_df.iloc[0:i + slice_size])
                if result is not None:
                    slices.append(result)
            return slices
        except FileNotFoundError as e:
            logging.error("POI file not found.")
            return None

    def get_training_set(self):
        """Build training slices from training content."""
        all_slices = []
        for i, (data_file, poi_file) in enumerate(self.train_content):
            logging.info(f'[{i+1}] Processing training dataset {data_file}')
            slices = self.get_slices(
                data_file, poi_file, slice_size=SLICE_SIZE)
            if slices is not None:
                all_slices.extend(slices)
        random.shuffle(all_slices)
        return all_slices

    def get_validation_set(self):
        """Build validation slices from validation content."""
        all_slices = []
        for i, (data_file, poi_file) in enumerate(self.validation_content):
            logging.info(f'[{i+1}] Processing validation dataset {data_file}')
            slices = self.get_slices(
                data_file, poi_file, slice_size=SLICE_SIZE)
            if slices is not None:
                all_slices.extend(slices)
        random.shuffle(all_slices)
        return all_slices

    def _extract_features(self, slices: list) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from a list of slices."""
        X_list, y_list = [], []
        for df in slices:
            y_list.append(df['Fill'].values)
            X_list.append(df.drop(columns=['Fill']).values)
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
            if self.training_slices is None or regenerate:
                self.training_slices = self.get_training_set()
            X_train, y_train = self._extract_features(self.training_slices)
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
            if self.validation_slices is None or regenerate:
                self.validation_slices = self.get_validation_set()
            X_val, y_val = self._extract_features(self.validation_slices)
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
        dtrain, dval = self._prepare_dmatrices()
        logging.info('Using cached training and validation DMatrices.')
        total_rounds = 50
        early_stopping_rounds = 5  # Stop if no improvement in validation loss for 5 rounds
        best_val_loss = float("inf")
        no_improvement = 0
        best_round = 0
        best_booster = None
        losses = []

        if self.live_plot_enabled:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))

        booster = self.booster
        for round in range(total_rounds):
            evals_result = {}
            booster = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params['n_boost_round'],
                evals=[(dtrain, "train"), (dval, "val")],
                evals_result=evals_result,
                xgb_model=booster
            )
            train_loss = evals_result["train"][EVAL_METRIC][0]
            val_loss = evals_result["val"][EVAL_METRIC][0]
            losses.append((train_loss, val_loss))
            logging.info(
                f"Round {round+1}: train {EVAL_METRIC}={train_loss}, val {EVAL_METRIC}={val_loss}")

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

            if self.live_plot_enabled:
                ax.clear()
                rounds_range = list(range(1, round + 2))
                train_losses, val_losses = zip(*losses)
                ax.plot(rounds_range, train_losses, label="Train Loss")
                ax.plot(rounds_range, val_losses, label="Val Loss")
                ax.legend()
                ax.set_xlabel("Boosting Round")
                ax.set_ylabel(EVAL_METRIC)
                ax.set_title("Training vs. Validation Loss")
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)

        if best_booster is not None:
            self.booster = best_booster
            logging.info(
                f"Training completed. Best round: {best_round+1} with val {EVAL_METRIC}={best_val_loss}")
        else:
            self.booster = booster
            logging.info("Training completed without improvement tracking.")

        if self.live_plot_enabled:
            plt.ioff()
            plt.show()

    def test(self, booster_path: str):
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

        test_cache_path = os.path.join(self.cache_dir, "test_slices.pkl")
        # Use regeneration only once if needed.
        regenerate = self.regen_data and not self._test_data_regenerated

        if not regenerate and os.path.exists(test_cache_path):
            try:
                with open(test_cache_path, "rb") as f:
                    self.cached_test_slices = pickle.load(f)
                logging.info("Loaded cached test slices from disk.")
            except Exception as e:
                logging.error(f"Error loading cached test slices: {e}")
                self.cached_test_slices = None

        if self.cached_test_slices is None or regenerate:
            test_slices = []
            for i, (data_file, poi_file) in enumerate(self.test_content):
                logging.info(f'[{i+1}] Processing test dataset: {data_file}')
                slices = self.get_slices(
                    data_file, poi_file, slice_size=SLICE_SIZE)
                if slices:
                    test_slices.extend(slices)
            if not test_slices:
                logging.warning("No test slices found.")
                return
            self.cached_test_slices = test_slices
            with open(test_cache_path, "wb") as f:
                pickle.dump(test_slices, f)
            logging.info("Cached test slices to disk.")

        # Mark that test slice regeneration has been done.
        self._test_data_regenerated = True

        test_slices = self.cached_test_slices

        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        slice_accuracies = []

        if self.live_plot_enabled:
            plt.ion()
            # Create three subplots instead of two.
            fig, (ax1, ax2, ax3) = plt.subplots(
                nrows=3, ncols=1, figsize=(6, 12))
        for i, df in enumerate(test_slices):
            y_slice = df['Fill']
            X_slice = df.drop(columns=['Fill']).values

            # Transform test data using the persisted scaler.
            X_slice_transformed = self.scaler.transform(X_slice)

            dtest = xgb.DMatrix(X_slice_transformed)
            pred_probs = self.booster.predict(dtest)

            if pred_probs.ndim == 1:
                predictions = (pred_probs > 0.7).astype(int)
            else:
                predictions = np.argmax(pred_probs, axis=1)

            correct = np.sum(predictions == y_slice)
            total_correct += correct
            total_samples += len(y_slice)

            all_predictions.extend(predictions)
            all_labels.extend(y_slice)

            slice_acc = correct / len(y_slice)
            slice_accuracies.append(slice_acc)

            for i, df in enumerate(test_slices):
                y_slice = df['Fill']
                X_slice = df.drop(columns=['Fill']).values

                # Transform test data using the persisted scaler.
                X_slice_transformed = self.scaler.transform(X_slice)

                dtest = xgb.DMatrix(X_slice_transformed)
                pred_probs = self.booster.predict(dtest)

                if pred_probs.ndim == 1:
                    predictions = (pred_probs > 0.7).astype(int)
                else:
                    predictions = np.argmax(pred_probs, axis=1)

                correct = np.sum(predictions == y_slice)
                total_correct += correct
                total_samples += len(y_slice)

                all_predictions.extend(predictions)
                all_labels.extend(y_slice)

                slice_acc = correct / len(y_slice)
                slice_accuracies.append(slice_acc)

                if self.live_plot_enabled:
                    ax1.clear()
                    ax2.clear()
                    ax3.clear()  # Clear the transformed data plot before updating

                    x_vals = np.arange(len(df))

                    # Plot the original Dissipation data and predicted classes on ax1.
                    ax1.plot(x_vals, df['Dissipation'], label="Dissipation")
                    ax1.set_title(
                        "Dissipation with Predicted Classes & Confidence")
                    ax1.set_xlabel("Index")
                    ax1.set_ylabel("Dissipation")
                    unique_classes = np.unique(predictions)
                    cmap = plt.get_cmap('Set1', len(unique_classes))
                    for idx, cls in enumerate(unique_classes):
                        mask = (predictions == cls)
                        ax1.fill_between(x_vals, df['Dissipation'], alpha=0.3,
                                         where=mask,
                                         color=cmap(idx),
                                         label=f"Predicted Class {cls}")
                    if pred_probs.ndim == 1:
                        conf_1 = pred_probs
                    else:
                        conf_1 = pred_probs[:, 1]
                    ax1.plot(x_vals, conf_1 * max(df['Dissipation']), '--', color='black',
                             label='Confidence (Class 1)')
                    ax1.legend()

                    # Plot test slice accuracies on ax2.
                    ax2.plot(slice_accuracies, marker='o')
                    ax2.set_title("Test Slice Accuracy")
                    ax2.set_xlabel("Slice Number")
                    ax2.set_ylabel("Accuracy")

                    # --- New Plot for Transformed Data on ax3 ---
                    # Here we loop over each feature in the transformed data.
                    for col in range(3):
                        ax3.plot(
                            x_vals, X_slice_transformed[:, col], label=f"Feature {col}")
                    ax3.set_title("Transformed Data")
                    ax3.set_xlabel("Index")
                    ax3.set_ylabel("Scaled Value")
                    ax3.legend()

                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.1)

        overall_accuracy = total_correct / total_samples
        logging.info(f"Test accuracy: {overall_accuracy * 100:.2f}%")
        from sklearn.metrics import classification_report, confusion_matrix
        logging.info(
            f"Classification Report:\n{classification_report(all_labels, all_predictions)}")
        logging.info(
            f"Confusion Matrix:\n{confusion_matrix(all_labels, all_predictions)}")

        if self.live_plot_enabled:
            plt.ioff()
            plt.show()

        return overall_accuracy

    def save_model(self, save_path: str):
        """Save the trained booster model."""
        if self.booster is None:
            logging.error("No booster model available to save.")
            return
        self.booster.save_model(save_path)
        logging.info(f"Model saved to {save_path}")

    def tune(self, n_trials: int = 50, num_boost_round: int = 50, early_stopping_rounds: int = 10):
        """Tune model hyperparameters using Optuna with cached DMatrices."""
        dtrain, dval = self._prepare_dmatrices()
        logging.info("Using cached DMatrices for tuning.")

        def objective(trial):
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

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        logging.info(f"Best trial: {study.best_trial.number}")
        logging.info(f"Best accuracy: {study.best_trial.value:.4f}")
        logging.info(f"Best parameters: {study.best_trial.params}")

        optimal_boost_rounds = study.best_trial.user_attrs.get(
            "optimal_boost_rounds")
        logging.info(
            f"Optimal boosting rounds determined: {optimal_boost_rounds}")

        self.params.update(
            {k: v for k, v in study.best_trial.params.items() if k != "importance_ratio"})

    def search(self, num_boost_round: int = 10):
        """Search for the best combination of objective and evaluation metric using cached DMatrices."""
        dtrain, dval = self._prepare_dmatrices()
        logging.info("Using cached DMatrices for objective/metric search.")

        best_accuracy = -1.0
        best_objective = None
        best_eval_metric = None
        results = []

        for obj in BINARY_OBJECTIVES:
            for eval_metric in BINARY_EVAL_METRICS:
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


# Execution control
if __name__ == "__main__":
    TRAINING = True
    booster_path = os.path.join(
        "QModel", "SavedModels", "bff.json")
    if TRAINING:
        # Pass True to regen_data if you want to regenerate the datasets (and caches) only once.
        ft = ForcasterTrainer(num_features=16, classes=[
                              "no_fill", "full_fill"], regen_data=True)
        ft.toggle_live_plot(True)  # Enable live plotting
        ft.search(num_boost_round=10)
        ft.tune()
        ft.train()
        ft.save_model(save_path=booster_path)
        ft.test(booster_path=booster_path)
