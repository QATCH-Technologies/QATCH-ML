from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.linear_model import LogisticRegression
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy.signal import hilbert, savgol_filter
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle
FEATURES = [
    'Relative_time',
    'Dissipation',
    'Dissipation_rolling_mean',
    'Dissipation_rolling_median',
    'Dissipation_ewm',
    'Dissipation_rolling_std',
    'Dissipation_diff',
    'Dissipation_pct_change',
    'Dissipation_rate',
    'Dissipation_ratio_to_mean',
    'Dissipation_ratio_to_ewm',
    'Dissipation_envelope',
    'Time_shift'
]
DATA_TO_LOAD = 50
IGNORE_BEFORE = 50


class QForecasterDataprocessor:
    @staticmethod
    def load_content(data_dir: str) -> list:
        """
        Walk through data_dir and return a list of tuples.
        Each tuple contains the path to a CSV file (excluding those ending in "_poi.csv" or "_lower.csv")
        and its corresponding POI file (with '_poi.csv' replacing '.csv').
        """
        print(f"[INFO] Loading content from {data_dir}")
        loaded_content = []
        for root, _, files in tqdm(os.walk(data_dir), desc='Loading files...'):
            for f in files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    poi_file = f.replace(".csv", "_poi.csv")
                    loaded_content.append(
                        (os.path.join(root, f), os.path.join(root, poi_file))
                    )
        return loaded_content

    @staticmethod
    def find_time_delta(df: pd.DataFrame) -> int:
        """
        Compute the first index at which the difference in Relative_time
        deviates significantly from its expanding rolling mean.
        Returns -1 if no significant change is found.
        """
        time_df = pd.DataFrame()
        time_df["Delta"] = df["Relative_time"].diff()
        threshold = 0.032
        rolling_avg = time_df["Delta"].expanding(min_periods=2).mean()
        time_df["Significant_change"] = (
            time_df["Delta"] - rolling_avg).abs() > threshold
        change_indices = time_df.index[time_df["Significant_change"]].tolist()
        return change_indices[0] if change_indices else -1

    @staticmethod
    def reassign_region(fill):
        """
        Reassign numeric fill values to string region labels.
        """
        if fill == 0:
            return 'no_fill'
        elif fill in [1, 2, 3]:
            return 'init_fill'
        elif fill == 4:
            return 'ch_1'
        elif fill == 5:
            return 'ch_2'
        elif fill == 6:
            return 'full_fill'
        else:
            return fill  # fallback if needed

    @staticmethod
    def compute_additional_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a series of additional features (e.g., rolling statistics,
        differences, ratios, and the signal envelope) for the Dissipation column.
        Also computes a 'Time_shift' column based on the first significant change in time.
        """
        window = 10
        span = 10
        run_length = len(df)
        window_length = int(np.ceil(0.01 * run_length))
        # Ensure window_length is odd and at least 3
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 3:
            window_length = 3
        polyorder = 2 if window_length > 2 else 1

        df['Dissipation'] = savgol_filter(df['Dissipation'].values,
                                          window_length=window_length,
                                          polyorder=polyorder)
        df['Dissipation_rolling_mean'] = df['Dissipation'].rolling(
            window=window, min_periods=1).mean()
        df['Dissipation_rolling_median'] = df['Dissipation'].rolling(
            window=window, min_periods=1).median()
        df['Dissipation_ewm'] = df['Dissipation'].ewm(
            span=span, adjust=False).mean()
        df['Dissipation_rolling_std'] = df['Dissipation'].rolling(
            window=window, min_periods=1).std()
        df['Dissipation_diff'] = df['Dissipation'].diff()
        df['Dissipation_pct_change'] = df['Dissipation'].pct_change()
        df['Relative_time_diff'] = df['Relative_time'].diff().replace(0, np.nan)
        df['Dissipation_rate'] = df['Dissipation_diff'] / df['Relative_time_diff']
        df['Dissipation_ratio_to_mean'] = df['Dissipation'] / \
            df['Dissipation_rolling_mean']
        df['Dissipation_ratio_to_ewm'] = df['Dissipation'] / df['Dissipation_ewm']
        df['Dissipation_envelope'] = np.abs(hilbert(df['Dissipation'].values))

        # Drop the Resonance_Frequency column if present.
        if 'Resonance_Frequency' in df.columns:
            df.drop(columns=['Resonance_Frequency'], inplace=True)

        t_delta = QForecasterDataprocessor.find_time_delta(df)
        if t_delta == -1:
            df['Time_shift'] = 0
        else:
            df.loc[t_delta:, 'Time_shift'] = 1

        return df

    @staticmethod
    def _process_fill(df: pd.DataFrame, poi_file: str, check_unique: bool = False, file_name: str = None) -> pd.DataFrame:
        """
        Helper to process fill information from the poi_file.
        Reads the poi CSV (with no header) and adds a Fill column to df.
        If the poi file does not contain a header, the method treats the first column
        as change indices (adding 1 to Fill from that index onward).
        Optionally, if check_unique is True, the method returns None when the number
        of unique fill values is not 7.
        """
        fill_df = pd.read_csv(poi_file, header=None)
        if "Fill" in fill_df.columns:
            df["Fill"] = fill_df["Fill"]
        else:
            df["Fill"] = 0
            change_indices = sorted(fill_df.iloc[:, 0].values)
            for idx in change_indices:
                df.loc[idx:, "Fill"] += 1

        df["Fill"] = pd.Categorical(df["Fill"]).codes
        if check_unique:
            unique_fill = sorted(df["Fill"].unique())
            if len(unique_fill) != 7:
                print(f"[WARNING] File {file_name} does not have 7 unique Fill values; skipping."
                      if file_name else "[WARNING] File does not have 7 unique Fill values; skipping.")
                return None

        df["Fill"] = df["Fill"].apply(QForecasterDataprocessor.reassign_region)
        mapping = {'no_fill': 0, 'init_fill': 1,
                   'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
        df["Fill"] = df["Fill"].map(mapping)
        return df

    @staticmethod
    def load_and_preprocess_data_split(data_dir: str, required_runs: int = 20):
        """
        Load and preprocess data from all files in data_dir. Each file (and its matching
        POI file) is processed to compute additional features and fill information.
        Files are then categorized into 'short_runs' and 'long_runs' based on whether
        a significant time delta is detected. Before final concatenation, if the number
        of rows in a run exceeds IGNORE_BEFORE, the first IGNORE_BEFORE rows are dropped.
        Returns two DataFrames: one for short runs and one for long runs.
        """
        short_runs, long_runs = [], []
        content = QForecasterDataprocessor.load_content(data_dir)
        random.shuffle(content)

        for file, poi_file in content:
            # Exit early if both groups have reached the required number of runs.
            if len(short_runs) >= required_runs and len(long_runs) >= required_runs:
                break

            df = pd.read_csv(file)
            required_cols = ["Relative_time",
                             "Resonance_Frequency", "Dissipation"]
            if df.empty or not all(col in df.columns for col in required_cols):
                continue

            df = df[required_cols]
            df = QForecasterDataprocessor.compute_additional_features(df)
            df = QForecasterDataprocessor._process_fill(
                df, poi_file, check_unique=True, file_name=file)
            if df is None:
                continue

            if len(df) > IGNORE_BEFORE:
                df = df.iloc[IGNORE_BEFORE:]

            delta_idx = QForecasterDataprocessor.find_time_delta(df)
            if delta_idx == -1:
                if len(short_runs) < required_runs:
                    short_runs.append(df)
            else:
                if len(long_runs) < required_runs:
                    long_runs.append(df)

        if len(short_runs) < required_runs or len(long_runs) < required_runs:
            raise ValueError(f"Not enough runs found. Required: {required_runs} short and {required_runs} long, "
                             f"found: {len(short_runs)} short and {len(long_runs)} long.")

        training_data_short = pd.concat(short_runs).sort_values(
            "Relative_time").reset_index(drop=True)
        training_data_long = pd.concat(long_runs).sort_values(
            "Relative_time").reset_index(drop=True)
        return training_data_short, training_data_long

    @staticmethod
    def load_and_preprocess_single(data_file: str, poi_file: str):
        """
        Load and preprocess a single data file (and its corresponding POI file).
        The method ensures required sensor columns exist, processes the fill information,
        and prints a head sample of the resulting DataFrame.
        """
        df = pd.read_csv(data_file)
        required_cols = ["Relative_time", "Resonance_Frequency", "Dissipation"]
        if df.empty or not all(col in df.columns for col in required_cols):
            raise ValueError(
                "Data file is empty or missing required sensor columns.")
        df = df[required_cols]
        df = QForecasterDataprocessor._process_fill(df, poi_file)
        if df is None:
            raise ValueError("Error processing fill information.")
        print("[INFO] Preprocessed single-file data sample:")
        print(df.head())
        return df

    @staticmethod
    def viterbi_decode(prob_matrix, transition_matrix):
        T, N = prob_matrix.shape
        dp = np.full((T, N), -np.inf)
        backpointer = np.zeros((T, N), dtype=int)
        dp[0, 0] = np.log(prob_matrix[0, 0])
        for t in range(1, T):
            for j in range(N):
                allowed_prev = [0] if j == 0 else [j-1, j]
                best_state = allowed_prev[0]
                best_score = dp[t-1, best_state] + \
                    np.log(transition_matrix[best_state, j])
                for i in allowed_prev:
                    if transition_matrix[i, j] <= 0:
                        continue
                    score = dp[t-1, i] + np.log(transition_matrix[i, j])
                    if score > best_score:
                        best_score = score
                        best_state = i
                dp[t, j] = np.log(prob_matrix[t, j]) + best_score
                backpointer[t, j] = best_state
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(dp[T-1])
        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]
        return best_path

    @staticmethod
    def compute_dynamic_transition_matrix(training_data, num_states=5, smoothing=1e-6):
        states = training_data["Fill"].values
        transition_counts = np.zeros((num_states, num_states))
        for i in range(num_states):
            transition_counts[i, i] = smoothing
            if i + 1 < num_states:
                transition_counts[i, i+1] = smoothing
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            if next_state == current_state or next_state == current_state + 1:
                transition_counts[current_state, next_state] += 1
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return transition_counts / row_sums


class QForecaster:
    def __init__(self, numerical_features, categorical_features=None, target='Fill'):
        """
        Args:
            numerical_features (list): List of names for numerical features.
            categorical_features (list): List of names for categorical features.
            target (str): Name of the target column.
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.target = target

    def _build_preprocessors(self, X):
        """Create and fit preprocessors for numerical and categorical data."""
        # Numerical preprocessors
        num_imputer = SimpleImputer(strategy='mean')
        X_num = num_imputer.fit_transform(X[self.numerical_features])
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
        preprocessors = {'num_imputer': num_imputer, 'scaler': scaler}

        # Categorical preprocessors (if applicable)
        if self.categorical_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_cat = cat_imputer.fit_transform(X[self.categorical_features])
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            X_cat = encoder.fit_transform(X_cat)
            preprocessors.update(
                {'cat_imputer': cat_imputer, 'encoder': encoder})
            X_processed = np.hstack([X_num, X_cat])
        else:
            X_processed = X_num

        return X_processed, preprocessors

    def _apply_preprocessors(self, X, preprocessors):
        """Apply already fitted preprocessors to transform new data."""
        X_num = preprocessors['num_imputer'].transform(
            X[self.numerical_features])
        X_num = preprocessors['scaler'].transform(X_num)

        if self.categorical_features:
            X_cat = preprocessors['cat_imputer'].transform(
                X[self.categorical_features])
            X_cat = preprocessors['encoder'].transform(X_cat)
            X_processed = np.hstack([X_num, X_cat])
        else:
            X_processed = X_num
        return X_processed

    def _tune_parameters(self, dtrain, base_params, max_evals=10):
        """Tune hyperparameters using hyperopt and return best params and rounds."""
        space = {
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1)
        }

        def objective(hyperparams):
            params = base_params.copy()
            params['max_depth'] = int(hyperparams['max_depth'])
            params['min_child_weight'] = int(hyperparams['min_child_weight'])
            params['learning_rate'] = hyperparams['learning_rate']
            params['subsample'] = hyperparams['subsample']
            params['colsample_bytree'] = hyperparams['colsample_bytree']

            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=200,
                nfold=5,
                metrics={'aucpr'},
                early_stopping_rounds=10,
                seed=42,
                verbose_eval=False
            )
            best_score = cv_results['test-aucpr-mean'].max()
            best_rounds = len(cv_results)
            return {'loss': -best_score, 'status': STATUS_OK, 'num_rounds': best_rounds}

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest,
                    max_evals=max_evals, trials=trials)
        # Ensure parameters are integer where needed
        best['max_depth'] = int(best['max_depth'])
        best['min_child_weight'] = int(best['min_child_weight'])
        params = base_params.copy()
        params.update(best)
        best_trial = min(trials.results, key=lambda x: x['loss'])
        optimal_rounds = best_trial['num_rounds']
        return params, optimal_rounds

    def train_model_native(self, training_data, tune=True):
        """
        Train the primary XGBoost model.

        Args:
            training_data (DataFrame): Data for training.
            tune (bool): Whether to perform hyperparameter tuning.

        Returns:
            model: Trained xgboost model.
            preprocessors (dict): Fitted preprocessors for later use.
            params (dict): Parameters used for training.
        """
        features = self.numerical_features + self.categorical_features
        X = training_data[features].copy()
        y = training_data[self.target].values

        # Build and apply preprocessors
        X_processed, preprocessors = self._build_preprocessors(X)
        dtrain = xgb.DMatrix(X_processed, label=y)

        base_params = {
            'objective': 'multi:softprob',
            'num_class': 5,
            'eval_metric': 'aucpr',
            'seed': 42,
            'device': 'cuda',
        }

        if tune:
            params, optimal_rounds = self._tune_parameters(dtrain, base_params)
        else:
            params = base_params.copy()
            params.update({'max_depth': 5, 'learning_rate': 0.1})
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=200,
                nfold=5,
                metrics={'aucpr'},
                early_stopping_rounds=10,
                seed=42,
                verbose_eval=False
            )
            optimal_rounds = len(cv_results)

        model = xgb.train(params, dtrain, num_boost_round=optimal_rounds)
        return model, preprocessors, params

    def predict_native(self, model, preprocessors, X):
        """
        Generate predictions using the trained model.

        Args:
            model: Trained xgboost model.
            preprocessors (dict): Preprocessors to transform the data.
            X (DataFrame): Input features.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        X_processed = self._apply_preprocessors(X.copy(), preprocessors)
        dmatrix = xgb.DMatrix(X_processed)
        prob_matrix = model.predict(dmatrix)
        return prob_matrix

    def train_meta_classifier(self, training_data, model_short, preprocessors_short, model_long, preprocessors_long, meta_model_type='xgb'):
        """
        Train a meta-classifier that learns to combine the outputs of two base models.
        It uses the predicted probability distributions (for each Fill class) as features.

        Args:
            training_data (DataFrame): Data used to train the meta-classifier.
            model_short: First base model.
            preprocessors_short (dict): Preprocessors used with the first model.
            model_long: Second base model.
            preprocessors_long (dict): Preprocessors used with the second model.
            meta_model_type (str): Either 'xgb' or 'logistic'.

        Returns:
            Trained meta-classifier.
        """
        features = self.numerical_features + self.categorical_features
        X_input = training_data[features]

        prob_matrix_short = self.predict_native(
            model_short, preprocessors_short, X_input)
        prob_matrix_long = self.predict_native(
            model_long, preprocessors_long, X_input)

        # Concatenate predictions to form meta-features
        X_meta = np.hstack([prob_matrix_short, prob_matrix_long])
        y_meta = training_data[self.target].values

        if meta_model_type == 'xgb':
            meta_clf = XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric='auc')
        elif meta_model_type == 'logistic':
            meta_clf = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError("Unsupported meta_model_type")

        meta_clf.fit(X_meta, y_meta)
        return meta_clf


class QForecasterPredictor:
    def __init__(self):
        pass


class QForecastSimluator:
    def __init__(self):
        pass
