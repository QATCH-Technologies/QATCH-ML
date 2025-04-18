from typing import List, Dict
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
from tqdm import tqdm
from q_model_data_processor import QDataProcessor
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

RANDOM_STATE = 42
TRAIN_PATH = os.path.join("content", "static", 'train')
TEST_PATH = os.path.join("content", "static", "test")
VALID_PATH = os.path.join("content", "static", "valid")
MODEL_PATH = os.path.join(
    "QModel", "SavedModels", "qmodel_v2a", "qmodel_boundary_aware.pkl")
TRAIN_NUM = np.inf
VALID_NUM = np.inf
TEST_NUM = np.inf


class QModelTrainerSequential:
    def __init__(self, training_directory: str = TRAIN_PATH,
                 validation_directory: str = VALID_PATH, train_mode: bool = False):
        self._boosters = self.make_boosters()
        self._scalers = self.make_scalers()
        self._params = self.make_params()
        if train_mode:
            self._training_splits, self._validation_splits = self.make_splits(training_directory,
                                                                              validation_directory)
            self._dtrain = self.make_dtrain()
            self._dvalid = self.make_dvalid()
        self._model_path = MODEL_PATH

    # ---------------------------------------------------- #
    # Data Processing
    # ---------------------------------------------------- #

    def make_splits(self, training_directory: str, validation_directory: str):
        training_content, _ = QDataProcessor.load_balanced_content(
            training_directory, num_datasets=TRAIN_NUM, opt=True)
        validation_content, _ = QDataProcessor.load_balanced_content(
            validation_directory, num_datasets=VALID_NUM, opt=True)
        training_splits = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame(
        ), 4: pd.DataFrame(), 5: pd.DataFrame(), 6: pd.DataFrame(), }
        validation_splits = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame(
        ), 4: pd.DataFrame(), 5: pd.DataFrame(), 6: pd.DataFrame(), }
        for split in training_splits.keys():
            training_splits[split] = self.load_training_data(
                training_content, split)
        for split in validation_splits.keys():
            validation_splits[split] = self.load_validation_data(
                validation_content, split)
        return training_splits, validation_splits

    def load_training_data(self, content: list, split_num: int):
        all_datasets = []
        for i, (data_file, poi_file) in tqdm(enumerate(content), desc=f"<<Loading Split {split_num} Training Data>>"):
            dataset = self._get_data(
                data_file=data_file, poi_file=poi_file, split_num=split_num)
            if dataset is not None:
                all_datasets.append(dataset)
        random.shuffle(all_datasets)
        X, y = self._extract_features(all_datasets)
        X, y = self._balance_data(X=X, y=y, dataset_name="Training Data")
        return X, y

    def load_validation_data(self, content: list, split_num: int):
        all_datasets = []
        for i, (data_file, poi_file) in tqdm(enumerate(content), desc=f"<<Loading Split {split_num} Validation Data>>"):
            dataset = self._get_data(
                data_file=data_file, poi_file=poi_file, split_num=split_num)
            if dataset is not None:
                all_datasets.append(dataset)
        random.shuffle(all_datasets)
        X, y = self._extract_features(all_datasets)
        X, y = self._balance_data(X=X, y=y, dataset_name="Validation Data")
        return X, y

    def _get_data(self, data_file: str, poi_file: str, split_num: int, live_poi_positions: list = None):
        try:
            data_df = QDataProcessor.process_data(data_file, live=True)
            poi_df, poi_positions = QDataProcessor.process_single_poi(
                poi_file, length_df=len(data_df), poi_num=split_num
            )
            data_df['POI'] = poi_df
            if live_poi_positions is not None:
                poi_positions = np.array(live_poi_positions)
            prev_labels = np.empty(len(data_df), dtype=int)
            prev_labels[:poi_positions[0]] = 0
            for i in range(5):
                start = poi_positions[i]
                end = poi_positions[i+1]
                prev_labels[start:end] = i + 1
            prev_labels[poi_positions[5]:] = 6
            prev_labels[prev_labels > (split_num - 1)] = 0
            data_df['Prev_POI'] = prev_labels
            return data_df
        except FileNotFoundError as e:
            logging.error("POI file not found.")
            return None

    def _extract_features(self, datasets: list):
        X_list, y_list = [], []
        for df in datasets:
            y_list.append(df['POI'].values)
            X_list.append(df.drop(columns=['POI']).values)
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        return X, y

    def _balance_data(self, X: np.ndarray, y: np.ndarray, dataset_name: str = "dataset"):
        initial_distribution = Counter(y)
        logging.info(
            f"Class distribution before SMOTE ({dataset_name}): {initial_distribution}")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X, y)
        balanced_distribution = Counter(y_res)
        logging.info(
            f"Class distribution after SMOTE ({dataset_name}): {balanced_distribution}")
        return X_res, y_res
    # ---------------------------------------------------- #
    # ML Components
    # ---------------------------------------------------- #

    def make_dtrain(self):
        dmatrices = {}
        for booster_index in self._boosters.keys():
            train_X, train_y = self._training_splits[booster_index]
            train_X_scaled = self._scalers[booster_index].fit_transform(
                train_X)
            dtrain = xgb.DMatrix(train_X_scaled, label=train_y)
            dmatrices[booster_index] = dtrain
        return dmatrices

    def make_dvalid(self):
        dmatrices = {}
        for booster_index in self._boosters.keys():
            valid_X, valid_y = self._validation_splits[booster_index]
            valid_X_scaled = self._scalers[booster_index].transform(valid_X)
            dvalid = xgb.DMatrix(valid_X_scaled, label=valid_y)
            dmatrices[booster_index] = dvalid
        return dmatrices

    def make_boosters(self):
        return {
            i: xgb.Booster()
            for i in range(1, 7)
        }

    def make_scalers(self):
        return {
            i: Pipeline([
                ('standard', StandardScaler()),
                ('minmax', MinMaxScaler(feature_range=(0, 1)))
            ])
            for i in range(1, 7)
        }

    def make_params(self):
        params = {
            "objective": "binary:logitraw",
            "learning_rate": 0.1,
            "eval_metric": "error",
            "tree_method": "hist",
            "device": "cuda",
        }
        return {
            i: params
            for i in range(1, 7)
        }

    def train(self):
        for booster_index in self._boosters.keys():
            self._train_booster(booster_index=booster_index,
                                num_boost_round=50, early_stopping=10, verbose=True)

    def tune(self):
        for booster_index in self._boosters.keys():
            self._tune_booster(booster_index=booster_index, n_trials=50)

    def test(self, test_path: str, model_path: str):
        if not os.path.exists(test_path):
            logging.error(
                f"Test directory cannot be found or does not exist {test_path}.")

        self.load(model_path)
        test_content = QDataProcessor.load_content(
            test_path, num_datasets=TEST_NUM)
        for run_idx, (data_file, poi_file) in enumerate(test_content, start=1):
            actual_df = pd.read_csv(poi_file, header=None)
            true_positions = actual_df[0].values
            all_predictions = []
            for poi_id in sorted(self._boosters.keys()):
                booster = self._boosters[poi_id]
                scaler = self._scalers[poi_id]

                dataset = self._get_data(
                    data_file=data_file,
                    poi_file=poi_file,
                    split_num=poi_id
                )
                if dataset is None:
                    continue

                X = dataset.drop(columns=["POI"])
                X_scaled = scaler.transform(X)
                dtest = xgb.DMatrix(X_scaled)
                probs = booster.predict(dtest)
                pred_idx = np.argmax(probs)
                all_predictions.append(pred_idx)

            errors = [abs(pred - true)
                      for pred, true in zip(all_predictions, true_positions)]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                           gridspec_kw={"height_ratios": [3, 1]})
            dissipation = dataset['Dissipation'].values
            ax1.plot(dissipation, label='Dissipation', color='grey')
            colors = {1: 'red', 2: 'orange', 3: 'yellow',
                      4: 'green', 5: 'blue', 6: 'purple'}
            for i, poi in enumerate(all_predictions, start=1):
                ax1.axvline(poi, color=colors[i],
                            linestyle='--', label=f'Pred POI{i} @ {poi}')
            for i, poi in enumerate(true_positions, start=1):
                ax1.axvline(poi, color=colors[i],
                            linestyle='-', label=f'True POI{i} @ {poi}')
            ax1.set_title(f'Run {run_idx}: Predicted vs True POIs')
            ax1.set_ylabel('Dissipation')
            ax1.legend(loc='upper right', fontsize='small')
            poi_labels = [f'POI{i}' for i in range(1, len(errors)+1)]
            bar_colors = [colors[i] for i in range(1, len(errors)+1)]
            ax2.bar(poi_labels, errors, color=bar_colors)
            ax2.set_ylabel('Absolute Error (indices)')
            ax2.set_xlabel('Point-of-Interest')
            ax2.set_title('Per-POI Error for This Run')
            for idx, err in enumerate(errors):
                ax2.text(idx, err + 0.1, str(err),
                         ha='center', va='bottom', fontsize='small')

            plt.tight_layout()
            plt.show()

    def save(self, path: str = None):
        file_path = path or self._model_path
        dir_path = os.path.dirname(file_path) or self._model_path
        os.makedirs(dir_path, exist_ok=True)
        to_dump = {
            "boosters": self._boosters,
            "scalers":  self._scalers
        }
        with open(file_path, "wb") as f:
            pickle.dump(to_dump, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Boosters saved to {path}.")

    def load(self, path: str = None):
        file_path = path or self._model_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No model file found at {file_path}")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self._boosters = data.get("boosters", {})
        self._scalers = data.get("scalers", {})
        logging.info(f"Boosters restored from {path}.")

    def _train_booster(self, booster_index: int, num_boost_round: int, early_stopping: int, verbose: bool):
        logging.info(f"Training booster for POI {booster_index}")
        dtrain = self._dtrain[booster_index]
        dvalid = self._dvalid[booster_index]
        evals = [(dtrain, 'train'), (dvalid, 'validation')]
        params = self._params[booster_index]
        trained_booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=verbose,
        )
        self._boosters[booster_index] = trained_booster

    def _tune_booster(self, booster_index: int, n_trials: int = 50, num_boost_round: int = 50, early_stopping_rounds: int = 10):
        dtrain = self._dtrain[booster_index]
        dvalid = self._dvalid[booster_index]

        def objective(trial: optuna.Trial):
            trial_params = self._params[booster_index].copy()
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
                evals=[(dtrain, "train"), (dvalid, "val")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
            optimal_rounds = booster.best_iteration
            trial.set_user_attr("optimal_boost_rounds", optimal_rounds)
            return booster.best_score

        direction = "minimize"
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        logging.info(f"TUNING RESULTS FOR BOOSTER {booster_index}")
        logging.info(f"Best trial: {study.best_trial.number}")
        logging.info(f"Best accuracy: {study.best_trial.value:.4f}")
        logging.info(f"Best parameters: {study.best_trial.params}")
        self._params[booster_index] = study.best_trial.params

        optimal_boost_rounds = study.best_trial.user_attrs.get(
            "optimal_boost_rounds")
        logging.info(
            f"Optimal boosting rounds determined: {optimal_boost_rounds}")


if __name__ == "__main__":
    qmts = QModelTrainerSequential(train_mode=True)
    qmts.tune()
    qmts.train()
    qmts.save()
    qmts.test(test_path=TEST_PATH, model_path=MODEL_PATH)
