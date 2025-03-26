from IPython.display import clear_output
from typing import List, Dict, Tuple, Optional
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from collections import deque, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import logging
from tqdm import tqdm

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


class ForcasterTrainer:
    def __init__(self,
                 classes: List,
                 num_features: int,
                 training_directory: str = r'content/training_data/full_fill',
                 validation_directory: str = r'content/test_data') -> None:
        if not isinstance(classes, list):
            raise TypeError("classes must be a list")
        if not isinstance(num_features, int):
            raise TypeError("num_features must be an integer")

        self.classes: List = classes
        self.num_features: int = num_features
        self.params: Dict = self.build_params()  # set up booster parameters
        self.booster: Optional[xgb.Booster] = None
        self.label_to_index: Dict[str, int] = {
            str(label): idx for idx, label in enumerate(classes)}
        self.content: Optional[List[Tuple[str, str]]] = None
        self.train_content: Optional[List[Tuple[str, str]]] = None
        self.test_content: Optional[List[Tuple[str, str]]] = None
        self.validation_content: Optional[List[Tuple[str, str]]] = None

        # Toggle flag for live plotting (default is off)
        self.live_plot_enabled: bool = False
        self.load_datasets(training_directory=training_directory,
                           validation_directory=validation_directory)

    def toggle_live_plot(self, enabled: bool = True):
        """Toggle live plotting on or off."""
        self.live_plot_enabled = enabled

    def build_params(self) -> Dict:
        """
        Build parameters dictionary for xgb.train using GPU-accelerated methods.
        """
        params = {
            "objective": "multi:softprob",
            "num_class": len(self.classes),
            "learning_rate": 0.1,
            "eval_metric": "mlogloss",
            "tree_method": "hist",    # Use GPU-accelerated histogram algorithm
            # "predictor": "gpu_predictor"   # Use GPU for prediction
            "device": "cuda"
        }
        return params

    def load_datasets(self, training_directory: str, validation_directory: str):
        """
        Loads and splits the training data.
        """
        self.content = DataProcessor.load_content(
            training_directory, num_datasets=20)
        self.train_content, self.test_content = train_test_split(
            self.content, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
        self.validation_content = DataProcessor.load_content(
            validation_directory, num_datasets=5)
        logging.info("Datasets loaded.")

    def get_slices(self, data_file: str, poi_file: str, slice_size: int = SLICE_SIZE):
        data_df = pd.read_csv(data_file)
        start_idx = 0
        try:
            fill_df = DataProcessor.process_fill(
                poi_file, length_df=len(data_df))
            data_df['Fill'] = fill_df
            data_df = data_df[::DOWNSAMPLE_FACTOR]
            total_rows = len(data_df)
            slices = []
            while start_idx < total_rows:
                end_idx = min(start_idx + slice_size, total_rows)
                current_slice = data_df.iloc[:end_idx]
                processed_slice = DataProcessor.preprocess_data(current_slice)
                start_idx = end_idx
                if processed_slice is None:
                    continue
                slices.append(processed_slice)
            return slices
        except FileNotFoundError as e:
            logging.error(f'POI file not found.')

    def get_training_set(self):
        all_slices = []
        for i, (data_file, poi_file) in enumerate(self.train_content):
            logging.info(f'[{i + 1}] Processing training dataset {data_file}')
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
                f'[{i + 1}] Processing validation dataset: {data_file}')
            slices = self.get_slices(
                data_file=data_file, poi_file=poi_file, slice_size=SLICE_SIZE)
            if slices is None:
                continue
            all_slices.extend(slices)
        random.shuffle(all_slices)
        return all_slices

    def train(self):
        # Prepare training and validation slices
        train_slices = self.get_training_set()
        val_slices = self.get_validation_set()
        logging.info(f'Number of training slices loaded: {len(train_slices)}')
        logging.info(f'Number of validation slices loaded: {len(val_slices)}')

        # Build training arrays from slices
        X_train_list, y_train_list = [], []
        for df in train_slices:
            y_train_list.append(df['Fill'].values)
            df.drop(columns=['Fill'])
            X_train_list.append(df)
            # Converting labels to string keys to match self.label_to_index dictionary keys

        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)

        X_val_list, y_val_list = [], []
        for df in val_slices:
            y_val_list.append(df['Fill'].values)
            df.drop(columns=['Fill'])
            X_val_list.append(df)

        X_val = np.vstack(X_val_list)
        y_val = np.hstack(y_val_list)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        total_rounds = 50  # total number of boosting rounds
        losses = []  # To store (train_loss, val_loss) for each round

        if self.live_plot_enabled:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))

        booster = self.booster  # initially None
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
            # Extract losses for the current round
            train_loss = evals_result["train"]["mlogloss"][0]
            val_loss = evals_result["val"]["mlogloss"][0]
            losses.append((train_loss, val_loss))
            logging.info(
                f"Round {round+1}: train mlogloss={train_loss}, val mlogloss={val_loss}")

            if self.live_plot_enabled:
                ax.clear()
                rounds_range = list(range(1, round + 2))
                train_losses, val_losses = zip(*losses)
                ax.plot(rounds_range, train_losses, label="Train Loss")
                ax.plot(rounds_range, val_losses, label="Val Loss")
                ax.legend()
                ax.set_xlabel("Boosting Round")
                ax.set_ylabel("mlogloss")
                ax.set_title("Training vs. Validation Loss")
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)

        self.booster = booster  # Save the final booster
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
                data_file=data_file, poi_file=poi_file, slice_size=100)
            if slices is None:
                continue
            test_slices.extend(slices)

        if not test_slices:
            logging.warning("No test slices found.")
            return

        slice_accuracies = []
        if self.live_plot_enabled:
            plt.ion()
            fig, ax = plt.subplots(figsize=(6, 4))

        for i, df in enumerate(test_slices):
            X_slice = df.drop(columns=['Fill']).values
            y_slice = df['Fill'].astype(str).map(self.label_to_index).values
            dtest = xgb.DMatrix(X_slice)
            # Get predicted probabilities then choose the class with highest probability
            pred_probs = self.booster.predict(dtest)
            predictions = np.argmax(pred_probs, axis=1)
            acc = np.mean(predictions == y_slice)
            slice_accuracies.append(acc)
            if self.live_plot_enabled:
                ax.clear()
                ax.plot(slice_accuracies, marker='o')
                ax.set_title("Test Slice Accuracy")
                ax.set_xlabel("Slice Number")
                ax.set_ylabel("Accuracy")
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)

        overall_accuracy = np.mean(slice_accuracies)
        logging.info(f"Test accuracy: {overall_accuracy * 100:.2f}%")
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
TESTING = True

if __name__ == "__main__":
    if TRAINING:
        ft = ForcasterTrainer(num_features=16, classes=[
                              "no_fill", "init_fill", "ch1", "ch2", "full_fill"])
        ft.toggle_live_plot(True)  # Enable live plotting
        ft.train()
        ft.test()
        ft.save_model(save_path=r'QModel/SavedModels/bff')

    if TESTING:
        f = Forecaster()
        try:
            f.load_model(model_path=r'QModel/SavedModels/bff')
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
