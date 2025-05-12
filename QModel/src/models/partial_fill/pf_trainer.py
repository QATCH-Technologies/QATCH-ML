import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Optional
import random
from pf_data_processor_v2 import PFDataProcessor
from sklearn.feature_selection import mutual_info_classif

SEED = 42
TRAINING_DIRECTORY = os.path.join('content', 'static', 'train')
TEST_DIRECTORY = os.path.join('content', 'static', 'test')
VALIDATION_DIRECTORY = os.path.join('content', 'static', 'valid')
BASE_DIR = os.path.join("QModel", "SavedModels", "pf")


class PFTrainer:
    def __init__(
        self,
        training_directory: str,
        test_directory: str,
        validation_directory: str,
        classes: list
    ) -> None:
        self._classes = classes

        self._params = self._build_default_params()
        self._booster = xgb.Booster()
        self._scaler = None
        self._train_directory = training_directory
        self._test_directory = test_directory
        self._validation_directory = validation_directory
        self._test_content = None
        self._dtrain = None
        self._dvalid = None

    def train(self, eval_metric: str = 'mlogloss', plotting: bool = False) -> None:
        if self._dtrain is None or self._dvalid is None:
            dtrain, dval = self._load_ddata()
        else:
            dtrain = self._dtrain
            dval = self._dvalid
        total_rounds = 50
        early_stopping_rounds = 5
        best_val_loss = float("inf")
        no_improvement = 0
        best_round = 0
        best_booster = None
        losses = []
        if plotting:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))

        booster = self._booster
        for round in range(total_rounds):
            evals_result = {}
            booster = xgb.train(
                self._params,
                dtrain,
                num_boost_round=self._params.get('n_boost_round', 10),
                evals=[(dtrain, "train"), (dval, "val")],
                evals_result=evals_result
            )

            # Use the dynamic evaluation metric to get losses
            train_loss = evals_result["train"][eval_metric][0]
            val_loss = evals_result["val"][eval_metric][0]
            losses.append((train_loss, val_loss))
            print(
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
                print(
                    f"Early stopping triggered after {early_stopping_rounds} rounds without improvement.")
                break

            # Update plot if live plotting is enabled
            if plotting:
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
            self._booster = best_booster
            print(
                f"Training completed. Best round: {best_round+1} with val {eval_metric}={best_val_loss}")
        else:
            self.booster = booster
            print("Training completed without improvement tracking.")

        if plotting:
            plt.ioff()
            plt.show()

    def tune(self, n_trials: int = 50, timeout: Optional[int] = None) -> None:

        def objective(trial: optuna.Trial) -> float:
            trial_params = {
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'objective': self._params.get('objective', 'multi:softprob'),
                'eval_metric': self._params.get('eval_metric', 'mlogloss'),
                'n_jobs': self._params.get('n_jobs', -1),
                'num_class':        7,
            }
            self._params.update(trial_params)
            if self._dtrain is None or self._dvalid is None:
                dtrain, dval = self._load_ddata()
            else:
                dtrain = self._dtrain
                dval = self._dvalid
            evals_result = {}
            booster = xgb.train(
                trial_params,
                dtrain,
                num_boost_round=self._params.get('n_boost_round', 1000),
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=self._params.get(
                    'early_stopping_rounds', 10),
                evals_result=evals_result,
                verbose_eval=False,
            )
            val_history = evals_result['val'][trial_params['eval_metric']]
            best_val = val_history[-1]
            best_iter = len(val_history)
            trial.set_user_attr('best_iteration', best_iter)

            return best_val
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        self._params.update(study.best_params)
        best_it = study.best_trial.user_attrs.get('best_iteration')
        if best_it is not None:
            self._params['n_boost_round'] = best_it

        print(
            f"Optuna tuning done 🎉 best {self._params['eval_metric']}="
            f"{study.best_value:.5f} (trial #{study.best_trial.number+1}),"
            f" n_boost_round={self._params['n_boost_round']}"
        )

    def test(self, num_datasets: int, plotting: bool = False) -> None:
        self.load_model(BASE_DIR)
        self._test_content = PFDataProcessor.load_content(
            self._test_directory, num_datasets=num_datasets)

        y_true, y_pred = [], []

        # loop over each run and collect predictions
        for i, (data_file, poi_file) in enumerate(self._test_content):
            data_df = pd.read_csv(data_file)
            pois = pd.read_csv(poi_file, header=None).values
            slice_idx = random.randint(0, len(data_df))
            sliced_df = data_df.iloc[:slice_idx]
            pois = [poi for poi in pois if poi[0] < slice_idx]
            num_pois = len(pois)
            simulated_poi_1 = None
            if num_pois >= 1:
                simulated_poi_1 = random.randint(pois[0], pois[0] + 10)

            # generate + scale features
            features = PFDataProcessor.generate_features(
                sliced_df, detected_poi1=simulated_poi_1)
            scaled_features = self._scaler.transform(features)
            ddata = xgb.DMatrix(scaled_features)

            # predict
            probs = self._booster.predict(ddata)
            pred = np.argmax(probs, axis=1)[0]

            # record for overall stats
            y_true.append(num_pois)
            y_pred.append(pred)

            # per-run plotting (unchanged)
            if False:
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

                # dissipation curve
                ax1.plot(data_df['Relative_time'], data_df['Dissipation'],
                         label='Full Curve', linewidth=2)
                ax1.plot(data_df['Relative_time'][:slice_idx],
                         data_df['Dissipation'][:slice_idx],
                         label='Sliced Portion', linewidth=2, color='orange')
                ax1.set_xlabel('Relative Time')
                ax1.set_ylabel('Dissipation')
                ax1.set_title(f"Run {i+1}")
                ax1.grid(True, linestyle='--', linewidth=0.5)

                # actual vs. predicted POIs
                is_correct = (num_pois == pred)
                ax2.barh(['Actual'], [num_pois], label='Actual',
                         edgecolor='black', alpha=0.7)
                ax2.barh(['Predicted'], [pred], label='Predicted',
                         edgecolor='black', alpha=0.7)
                ax2.set_xlim(0, 6)
                ax2.set_xlabel('POI Count')
                ax2.set_title(
                    "Actual vs. Predicted POIs" + (" ✓" if is_correct else " ✗"))
                ax2.grid(True, axis='x', linestyle='--', linewidth=0.5)
                ax2.legend()

                plt.tight_layout()
                plt.show()

        # --- overall metrics & plots ---
        if plotting and y_true:
            # 1. Overall accuracy
            acc = accuracy_score(y_true, y_pred)
            print(f"Overall accuracy: {acc:.2%}")

            fig, ax = plt.subplots(figsize=(4, 5))
            ax.bar(['Accuracy'], [acc], edgecolor='black')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Accuracy')
            ax.set_title('Overall Model Accuracy')
            ax.text(0, acc + 0.02, f"{acc:.2%}",
                    ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            plt.show()

            # 2. Confusion matrix
            # assume classes are 0 through max observed
            classes = sorted(set(y_true) | set(y_pred))
            cm = confusion_matrix(y_true, y_pred, labels=classes)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)

            ax.set(
                xticks=np.arange(len(classes)),
                yticks=np.arange(len(classes)),
                xticklabels=classes, yticklabels=classes,
                xlabel='Predicted label',
                ylabel='True label',
                title='Confusion Matrix'
            )

            # annotate counts
            thresh = cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                ax.text(j, i, cm[i, j],
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

            plt.tight_layout()
            plt.show()

    def save_model(self, save_dir: str) -> None:
        try:
            self._save_scaler(save_dir)
        except Exception as e:
            raise IOError(f"Saving scaler failed with error: `{e}`")
        try:
            self._save_booster(save_dir)
        except Exception as e:
            raise IOError(f"Saving booster failed with error: `{e}`")

    def load_model(self, load_dir: str) -> None:
        try:
            self._load_scaler(load_dir)
        except Exception as e:
            raise IOError(f"Loading scaler failed with error: `{e}`")
        try:
            self._load_booster(load_dir)
        except Exception as e:
            raise IOError(f"Loading booster failed with error: `{e}`")

    def _build_default_params(self) -> dict:
        params = {
            "objective": "multi:softprob",
            "learning_rate": 0.1,
            "eval_metric": 'mlogloss',
            "tree_method": "hist",
            "device": "cuda",
            "num_class": len(self._classes) - 1,
        }
        return params

    def _generate_dataset(self, data_dir: str, num_datasets: int, plotting: bool = False):
        content = PFDataProcessor.load_content(
            data_dir=data_dir, num_datasets=num_datasets)
        dataset = pd.DataFrame()
        for data_file, poi_file in content:
            data_df = pd.read_csv(data_file)
            pois = pd.read_csv(poi_file, header=None).values
            for i in range(len(pois)):
                start = 0
                if i >= 1:
                    prev_poi = pois[i - 1]
                    detected_poi1 = pois[0]
                else:
                    prev_poi = start
                    detected_poi1 = None
                end = random.randint(prev_poi, pois[i])
                slice_df = data_df[start:end + 1]
                features = PFDataProcessor.generate_features(
                    dataframe=slice_df, sampling_rate=1.0, detected_poi1=detected_poi1)
                features['label'] = i
                dataset = pd.concat([dataset, features], axis=0)
        y = dataset['label']
        dataset.drop(columns=['label'], inplace=True)
        X = dataset

        def plot_feature_importances(X: pd.DataFrame,
                                     y: pd.Series,
                                     top_n: int = 5):
            """
            For each class in y:
            – Compute |Pearson r| against a one‐vs‐all indicator → “linear” importance
            – Compute mutual_info_classif          → “nonlinear” importance
            Then plot the top_n of each side by side.
            """
            classes = sorted(y.unique())

            for cls in classes:
                # one‐vs‐all indicator
                y_bin = (y == cls).astype(int)

                # linear importance: |corr|
                corrs = X.corrwith(y_bin).abs().sort_values(
                    ascending=False).head(top_n)

                # nonlinear importance: mutual info
                mi = mutual_info_classif(
                    X, y_bin, discrete_features=False, random_state=0)
                mi_ser = pd.Series(mi, index=X.columns).sort_values(
                    ascending=False).head(top_n)

                # new figure for this class
                fig, (ax_lin, ax_mi) = plt.subplots(
                    1, 2, figsize=(10, 4), sharey=False
                )

                corrs.plot.bar(ax=ax_lin)
                ax_lin.set_title(f"Class {cls}-Top {top_n} by |Pearson r|")
                ax_lin.set_ylabel("|Correlation|")
                ax_lin.tick_params(axis='x', rotation=90)

                mi_ser.plot.bar(ax=ax_mi)
                ax_mi.set_title(f"Class {cls}-Top {top_n} by Mutual Info")
                ax_mi.set_ylabel("Mutual Information")
                ax_mi.tick_params(axis='x', rotation=90)

                plt.tight_layout()
                plt.show()
        plot_feature_importances(X=X, y=y, top_n=20)
        return X, y

    def _load_ddata(self) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        # Generate datasets
        X_train, y_train = self._generate_dataset(
            self._train_directory, num_datasets=100)

        # Create and apply the scaler pipeline
        self._scaler = Pipeline([
            ('standard', StandardScaler()),
            ('minmax', MinMaxScaler(feature_range=(0, 1)))
        ])
        X_train_scaled = self._scaler.fit_transform(X_train)

        # Validate dataset
        X_valid, y_valid = self._generate_dataset(
            self._validation_directory, num_datasets=100)
        X_valid_scaled = self._scaler.transform(X_valid)

        # Remove columns with all zero values or constant columns
        # X_train_scaled = X_train_scaled[:, (X_train_scaled != 0).any(axis=0)]
        # X_valid_scaled = X_valid_scaled[:, (X_valid_scaled != 0).any(axis=0)]

        # Ensure data is in numeric NumPy array form
        X_train_scaled = np.array(X_train_scaled)
        X_valid_scaled = np.array(X_valid_scaled)

        # Ensure labels are numeric (in case they aren't)
        y_train = np.array(y_train, dtype=np.float32)
        y_valid = np.array(y_valid, dtype=np.float32)

        # Create XGBoost DMatrix objects
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dvalid = xgb.DMatrix(X_valid_scaled, label=y_valid)
        self._dtrain = dtrain
        self._dvalid = dvalid
        return dtrain, dvalid

    def _load_scaler(self, base_dir) -> None:
        load_path = os.path.join(base_dir, 'pf_scaler.pkl')
        if os.path.exists(load_path):
            with open(load_path, "rb") as f:
                self._scaler = pickle.load(f)
        else:
            raise IOError(f"Path to scaler does not exist `{load_path}`.")

    def _load_booster(self, base_dir: str) -> None:
        load_path = os.path.join(base_dir, 'pf_booster.json')
        if os.path.exists(load_path):
            self._booster.load_model(load_path)
        else:
            raise IOError(f"Path to booster does not exist `{load_path}`.")

    def _save_scaler(self, base_dir: str) -> None:
        save_path = os.path.join(base_dir, 'pf_scaler.pkl')
        with open(save_path, "wb") as f:
            pickle.dump(self._scaler, f)

    def _save_booster(self, base_dir: str) -> None:
        save_path = os.path.join(base_dir, 'pf_booster.json')
        self._booster.save_model(save_path)


if __name__ == "__main__":
    trainer = PFTrainer(training_directory=TRAINING_DIRECTORY,
                        test_directory=TEST_DIRECTORY,
                        validation_directory=VALIDATION_DIRECTORY,
                        classes=[0, 1, 2, 3, 4, 5, 6])
    # trainer.tune()
    # trainer.train(plotting=True)
    # trainer.save_model(save_dir=BASE_DIR)
    trainer.test(num_datasets=np.inf, plotting=True)
