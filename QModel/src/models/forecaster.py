from typing import List, Dict, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import logging
from collections import Counter
from imblearn.over_sampling import SMOTE
import optuna
from concurrent.futures import ThreadPoolExecutor, as_completed


from forecaster_data_processor import DataProcessor  # assuming this exists

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

TRAIN_PATH = os.path.join("content", "training_data", "full_fill")
TEST_PATH = os.path.join("content", "test_data")


# Make sure these globals/constants are defined somewhere in your code:
# EVAL_METRIC, SLICE_SIZE, TEST_SIZE, RANDOM_STATE, DOWNSAMPLE_FACTOR, etc.


class ForcasterTrainer:
    def __init__(self,
                 classes: list,
                 num_features: int,
                 training_directory: str = TRAIN_PATH,
                 validation_directory: str = TEST_PATH) -> None:
        if not isinstance(classes, list):
            raise TypeError("classes must be a list")
        if not isinstance(num_features, int):
            raise TypeError("num_features must be an integer")

        self.classes: list = classes
        self.num_features: int = num_features
        self.params: dict = self.build_params()  # set up booster parameters
        self.booster: xgb.Booster = None
        self.label_to_index: dict = {
            str(label): idx for idx, label in enumerate(classes)}
        self.content: list = None
        self.train_content: list = None
        self.test_content: list = None
        self.validation_content: list = None

        self.training_slices = None
        self.validation_slices = None
        self.test_slices = None

        # Toggle flag for live plotting (default is off)
        self.live_plot_enabled: bool = False
        self.load_datasets(training_directory=training_directory,
                           validation_directory=validation_directory)

    def toggle_live_plot(self, enabled: bool = True):
        """Toggle live plotting on or off."""
        self.live_plot_enabled = enabled

    def build_params(self) -> dict:
        params = {
            "objective": "binary:logistic",  # changed from multi:softprob to binary:logistic
            "learning_rate": 0.1,
            "eval_metric": EVAL_METRIC,
            "tree_method": "hist",
            "device": "cuda"
        }
        return params

    def load_datasets(self, training_directory: str, validation_directory: str):
        """
        Loads and splits the training data.
        """
        self.content = DataProcessor.load_content(
            training_directory, num_datasets=200)
        self.train_content, self.test_content = train_test_split(
            self.content, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
        self.validation_content = DataProcessor.load_content(
            validation_directory, num_datasets=100)
        logging.info("Datasets loaded.")

    def get_slices(self, data_file: str, poi_file: str, slice_size: int = SLICE_SIZE):
        try:
            data_df = pd.read_csv(data_file)
            fill_df = DataProcessor.process_fill(
                poi_file, length_df=len(data_df))
            data_df['Fill'] = fill_df
            data_df = data_df[::DOWNSAMPLE_FACTOR]
            total_rows = len(data_df)

            slices = []
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(DataProcessor.preprocess_data, data_df.iloc[0:i+slice_size]): i
                    for i in range(0, total_rows, slice_size)
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        slices.append(result)
            return slices
        except FileNotFoundError as e:
            logging.error('POI file not found.')

    def get_training_set(self):
        all_slices = []
        for i, (data_file, poi_file) in enumerate(self.train_content):
            logging.info(
                f'[{i + 1}] Processing training dataset `{data_file}`')
            slices = self.get_slices(
                data_file=data_file, poi_file=poi_file, slice_size=SLICE_SIZE)
            if slices is None:
                continue
            all_slices.extend(slices)
        random.shuffle(all_slices)
        return all_slices

    def get_validation_set(self):
        all_slices = []
        for i, (data_file, poi_file) in enumerate(self.validation_content):
            logging.info(
                f'[{i + 1}] Processing validation dataset: `{data_file}`')
            slices = self.get_slices(
                data_file=data_file, poi_file=poi_file, slice_size=SLICE_SIZE)
            if slices is None:
                continue
            all_slices.extend(slices)
        random.shuffle(all_slices)
        return all_slices

    def train(self):
        # Prepare training and validation slices
        # Prepare training and validation data (same as in train())
        if self.training_slices is None:
            self.training_slices = self.get_training_set()
        if self.validation_slices is None:
            self.validation_slices = self.get_validation_set()
        if not self.training_slices or not self.validation_slices:
            logging.error("Training or validation slices not found.")
            return
        logging.info(
            f'Number of training slices loaded: {len(self.training_slices)}')
        logging.info(
            f'Number of validation slices loaded: {len(self.validation_slices)}')

        # Build training arrays from slices
        X_train_list, y_train_list = [], []
        for df in self.training_slices:
            y_train_list.append(df['Fill'].values)
            df = df.drop(columns=['Fill'])
            X_train_list.append(df.values)
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)

        # Measure class distribution before applying SMOTE
        initial_distribution = Counter(y_train)
        logging.info(
            f"Class distribution before SMOTE: {initial_distribution}")

        # Balance the training set using SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        balanced_distribution = Counter(y_train_res)
        logging.info(
            f"Class distribution after SMOTE: {balanced_distribution}")

        # Build validation arrays from slices
        X_val_list, y_val_list = [], []
        for df in self.validation_slices:
            y_val_list.append(df['Fill'].values)
            df = df.drop(columns=['Fill'])
            X_val_list.append(df.values)
        X_val = np.vstack(X_val_list)
        y_val = np.hstack(y_val_list)

        # Create XGBoost DMatrix objects using the resampled training data
        dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
        dval = xgb.DMatrix(X_val, label=y_val)

        total_rounds = 50  # Total number of boosting rounds
        losses = []       # To store (train_loss, val_loss) for each round

        if self.live_plot_enabled:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))

        booster = self.booster  # Initially None
        for round in range(total_rounds):
            evals_result = {}
            booster = xgb.train(
                self.params,
                dtrain,
                num_boost_round=1,
                evals=[(dtrain, "train"), (dval, "val")],
                evals_result=evals_result,
                xgb_model=booster
            )
            train_loss = evals_result["train"][EVAL_METRIC][0]
            val_loss = evals_result["val"][EVAL_METRIC][0]
            losses.append((train_loss, val_loss))
            logging.info(
                f"Round {round+1}: train {EVAL_METRIC}={train_loss}, val {EVAL_METRIC}={val_loss}")

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

        self.booster = booster
        logging.info("Training completed.")
        if self.live_plot_enabled:
            plt.ioff()
            plt.show()

    def test(self):
        if self.test_content is None:
            logging.error(
                "Test content not loaded. Please run load_datasets first.")
            return

        test_slices = []
        for i, (data_file, poi_file) in enumerate(self.test_content):
            logging.info(f'[{i + 1}] Processing test dataset: {data_file}')
            slices = self.get_slices(
                data_file=data_file, poi_file=poi_file, slice_size=SLICE_SIZE)
            if slices is None:
                continue
            test_slices.extend(slices)

        if not test_slices:
            logging.warning("No test slices found.")
            return

        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        slice_accuracies = []

        if self.live_plot_enabled:
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))

        for i, df in enumerate(test_slices):
            y_slice = df['Fill']
            X_slice = df.drop(columns=['Fill']).values
            dtest = xgb.DMatrix(X_slice)
            pred_probs = self.booster.predict(dtest)

            if pred_probs.ndim == 1:
                predictions = (pred_probs > 0.5).astype(int)
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
                x_vals = np.arange(len(df))
                ax1.plot(x_vals, df['Dissipation'], label="Dissipation")
                ax1.set_title("Dissipation with Predicted Classes")
                ax1.set_xlabel("Index")
                ax1.set_ylabel("Dissipation")
                unique_classes = np.unique(predictions)
                cmap = plt.cm.get_cmap('Set1', len(unique_classes))
                for idx, cls in enumerate(unique_classes):
                    mask = (predictions == cls)
                    ax1.fill_between(x_vals, df['Dissipation'], alpha=0.3,
                                     where=mask,
                                     color=cmap(idx),
                                     label=f"Predicted Class {cls}")
                ax1.legend()
                ax2.plot(slice_accuracies)
                ax2.set_title("Test Slice Accuracy")
                ax2.set_xlabel("Slice Number")
                ax2.set_ylabel("Accuracy")
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
        """Saves the trained booster model to the specified file path."""
        if self.booster is None:
            logging.error("No booster model available to save.")
            return
        self.booster.save_model(save_path)
        logging.info(f"Model saved to {save_path}")

    def tune(self,
             n_trials: int = 50,
             num_boost_round: int = 50,
             early_stopping_rounds: int = 10):
        """
        Tunes hyperparameters using Optuna, including determining the optimal number of boosting rounds.

        Args:
            n_trials (int): Number of Optuna trials.
            num_boost_round (int): Maximum number of boosting rounds per trial.
            early_stopping_rounds (int): Early stopping rounds for each trial.
        """
        # Prepare training and validation data (same as in train())
        if not hasattr(self, "training_slices") or self.training_slices is None:
            self.training_slices = self.get_training_set()
        if not hasattr(self, "validation_slices") or self.validation_slices is None:
            self.validation_slices = self.get_validation_set()
        if not self.training_slices or not self.validation_slices:
            logging.error("Training or validation slices not found.")
            return

        # Build training arrays
        X_train_list, y_train_list = [], []
        for df in self.training_slices:
            y_train_list.append(df['Fill'].values)
            df = df.drop(columns=['Fill'])
            X_train_list.append(df.values)
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)

        # Balance the training set using SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Build validation arrays
        X_val_list, y_val_list = [], []
        for df in self.validation_slices:
            y_val_list.append(df['Fill'].values)
            df = df.drop(columns=['Fill'])
            X_val_list.append(df.values)
        X_val = np.vstack(X_val_list)
        y_val = np.hstack(y_val_list)

        # Create DMatrix objects for XGBoost
        dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, "train"), (dval, "val")]

        def objective(trial):
            # Suggest hyperparameters
            trial_params = self.params.copy()
            trial_params["learning_rate"] = trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True)
            trial_params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            trial_params["subsample"] = trial.suggest_float(
                "subsample", 0.5, 1.0)
            trial_params["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree", 0.5, 1.0)
            # Suggest the maximum number of boosting rounds for this trial (lower bound fixed to 10)
            trial_n_boost_round = trial.suggest_int(
                "n_boost_round", 10, num_boost_round)

            # Train with early stopping; early stopping will determine the optimal iteration
            booster = xgb.train(
                trial_params,
                dtrain,
                num_boost_round=trial_n_boost_round,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            # Record the actual optimal number of boosting rounds from early stopping
            optimal_rounds = booster.best_iteration
            trial.set_user_attr("optimal_boost_rounds", optimal_rounds)
            # Return the best validation score (assumes lower is better)
            return booster.best_score

        # Create and run the Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        logging.info(f"Best trial: {study.best_trial.number}")
        logging.info(f"Best score: {study.best_trial.value}")
        logging.info(f"Best parameters: {study.best_trial.params}")

        # Retrieve the optimal boosting rounds determined by early stopping from the best trial.
        optimal_boost_rounds = study.best_trial.user_attrs.get(
            "optimal_boost_rounds")
        logging.info(
            f"Optimal boosting rounds determined: {optimal_boost_rounds}")

        # Update the trainer with the best parameters found
        self.params.update(study.best_trial.params)


class Forecaster:
    def __init__(self):
        self.model: Optional[xgb.Booster] = None

    def load_model(self, model_path: str):
        """
        Loads a saved XGBoost model from the specified path.
        """
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        print(f"Model loaded from {model_path}")

    def predict_batch(self, batch_data: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the provided batch data.
        The batch_data is first processed into features using your DataProcessor.
        """
        if self.model is None:
            raise ValueError(
                "Model is not loaded. Please call load_model() first.")

        features = DataProcessor.generate_features(batch_data, live=True)
        if 'Relative_time' in features.columns:
            features.drop(columns=['Relative_time'], inplace=True)
        dmatrix = xgb.DMatrix(features.values)
        pred_probs = self.model.predict(dmatrix)
        predictions = np.argmax(pred_probs, axis=1)
        return predictions


# Execution control
TRAINING = True
TESTING = False

if __name__ == "__main__":
    if TRAINING:
        ft = ForcasterTrainer(num_features=16, classes=[
                              "no_fill", "filling", "full_fill"])
        ft.toggle_live_plot(True)  # Enable live plotting
        ft.tune()
        ft.train()
        ft.test()
        ft.save_model(save_path=os.path.join(
            "QModel", "SavedModels", "bff.json"))

    if TESTING:
        f = Forecaster()
        try:
            f.load_model(model_path=os.path.join(
                "QModel", "SavedModels", "bff.json"))
        except Exception as e:
            logging.error(f"Could not load model: {e}")
            exit(1)

        test_data_file = r'content/test_data/01037/W10_BUFFER_I10_3rd.csv'
        try:
            full_data = pd.read_csv(test_data_file)
        except Exception as e:
            logging.error(f"Error reading test data file: {e}")
            exit(1)

        full_data = full_data.iloc[::DOWNSAMPLE_FACTOR]

        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 6))

        for end_idx in range(100, len(full_data) + 1, 100):
            slice_df = full_data.iloc[:end_idx].reset_index(drop=True)
            try:
                predicted_classes = f.predict_batch(slice_df)
            except Exception as e:
                logging.error(
                    f"Error during prediction for slice ending at row {end_idx}: {e}")
                continue

            if "Dissipation" not in slice_df.columns:
                logging.warning(
                    f"Column 'Dissipation' not found for slice ending at row {end_idx}.")
                continue

            x_values = list(range(len(slice_df)))
            dissipation_values = slice_df["Dissipation"].to_numpy()
            min_val = dissipation_values.min()
            max_val = dissipation_values.max()
            if max_val - min_val == 0:
                norm_dissipation = np.zeros_like(dissipation_values)
            else:
                norm_dissipation = (dissipation_values -
                                    min_val) / (max_val - min_val)

            # Ensure predictions are 1D
            pred = predicted_classes.flatten()

            min_len = min(len(norm_dissipation), len(pred))
            x_values = x_values[:min_len]
            norm_dissipation = norm_dissipation[:min_len]
            pred = pred[:min_len]
            color_to_class_map = {0: "red", 1: "orange",
                                  2: "pink", 3: "green", 4: "blue"}
            ax.clear()
            ax.plot(x_values, norm_dissipation, label="Normalized Dissipation")
            for i, pred_val in enumerate(pred):
                color = color_to_class_map.get(pred_val, "gray")
                ax.axvspan(i, i + 1, facecolor=color, alpha=0.3)

            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=col, edgecolor='none', label=f'Class {cls}')
                               for cls, col in color_to_class_map.items()]
            line_handle, = ax.plot([], [], label="Normalized Dissipation")
            legend_elements.append(line_handle)

            ax.set_title(
                f"Normalized Dissipation and Predictions (Slice size: {end_idx})")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.legend(handles=legend_elements)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.draw()

        plt.ioff()
        plt.show()
