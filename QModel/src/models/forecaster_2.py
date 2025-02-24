import os
import time
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import hilbert
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Global constants for default configuration:
DEFAULT_FEATURES = [
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
IGNORE_BEFORE = 50


class DissipationEnsembleModel:
    def __init__(self, config=None):
        """
        Initialize the ensemble model with an optional configuration dictionary.
        Expected keys include:
          - features: list of numerical feature names.
          - ignore_before: number of early rows to ignore.
          - data_to_load: (optional) number of files to load.
        """
        if config is None:
            config = {}
        self.features = config.get("features", DEFAULT_FEATURES)
        self.ignore_before = config.get("ignore_before", IGNORE_BEFORE)
        self.data_to_load = config.get("data_to_load", 100)
        # Placeholders for models and preprocessors.
        self.model_short = None
        self.model_long = None
        self.preprocessors_short = None
        self.preprocessors_long = None
        self.params_short = None
        self.params_long = None
        self.transition_matrix_short = None
        self.transition_matrix_long = None

    # ---------------------------
    # Feature Engineering & Helpers
    # ---------------------------
    @staticmethod
    def compute_additional_features(df: pd.DataFrame) -> pd.DataFrame:
        window = 10
        span = 10
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
        if 'Resonance_Frequency' in df.columns:
            df.drop(columns=['Resonance_Frequency'], inplace=True)
        t_delta = DissipationEnsembleModel.find_time_delta(df)
        if t_delta == -1:
            df['Time_shift'] = 0
        else:
            df.loc[t_delta:, 'Time_shift'] = 1
        return df

    @staticmethod
    def find_time_delta(df: pd.DataFrame) -> int:
        time_df = pd.DataFrame()
        time_df["Delta"] = df["Relative_time"].diff()
        threshold = 0.032
        rolling_avg = time_df["Delta"].expanding(min_periods=2).mean()
        time_df["Significant_change"] = (
            time_df["Delta"] - rolling_avg).abs() > threshold
        change_indices = time_df.index[time_df["Significant_change"]].tolist()
        return change_indices[0] if change_indices else -1

    @staticmethod
    def reassign_region(Fill):
        if Fill == 0:
            return 'no_fill'
        elif Fill in [1, 2, 3]:
            return 'init_fill'
        elif Fill == 4:
            return 'ch_1'
        elif Fill == 5:
            return 'ch_2'
        elif Fill == 6:
            return 'full_fill'
        else:
            return Fill

    # ---------------------------
    # Data Loading & Preprocessing
    # ---------------------------
    def load_content(self, data_dir: str) -> list:
        print(f"[INFO] Loading content from {data_dir}")
        loaded_content = []
        for data_root, _, data_files in tqdm(os.walk(data_dir), desc='Loading files...'):
            for f in data_files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    matched_poi_file = f.replace(".csv", "_poi.csv")
                    loaded_content.append((os.path.join(data_root, f),
                                           os.path.join(data_root, matched_poi_file)))
        return loaded_content

    def load_and_preprocess_data_split(self, data_dir: str, required_runs=20):
        """
        Loads CSV files from the provided directory, computes features,
        and splits the data into short and long runs based on time delta detection.
        """
        short_runs = []
        long_runs = []
        content = self.load_content(data_dir)
        random.shuffle(content)

        for file, poi_file in content:
            if len(short_runs) >= required_runs and len(long_runs) >= required_runs:
                break

            df = pd.read_csv(file)
            if df.empty or not all(col in df.columns for col in ["Relative_time", "Resonance_Frequency", "Dissipation"]):
                continue

            df = df[["Relative_time", "Resonance_Frequency", "Dissipation"]]
            df = self.compute_additional_features(df)

            Fill_df = pd.read_csv(poi_file, header=None)
            if "Fill" in Fill_df.columns:
                df["Fill"] = Fill_df["Fill"]
            else:
                df["Fill"] = 0
                change_indices = sorted(Fill_df.iloc[:, 0].values)
                for idx in change_indices:
                    df.loc[idx:, "Fill"] += 1

            df["Fill"] = pd.Categorical(df["Fill"]).codes
            unique_Fill = sorted(df["Fill"].unique())
            if len(unique_Fill) != 7:
                print(
                    f"[WARNING] File {file} does not have 7 unique Fill values; skipping.")
                continue

            df['Fill'] = df['Fill'].apply(self.reassign_region)
            mapping = {'no_fill': 0, 'init_fill': 1,
                       'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
            df['Fill'] = df['Fill'].map(mapping)

            delta_idx = self.find_time_delta(df)
            if delta_idx == -1:
                if len(short_runs) < required_runs:
                    short_runs.append(df)
            else:
                if len(long_runs) < required_runs:
                    long_runs.append(df)

            if len(df) > self.ignore_before:
                df = df.iloc[self.ignore_before:]

        if len(short_runs) < required_runs or len(long_runs) < required_runs:
            raise ValueError(f"Not enough runs found. Required: {required_runs} short and {required_runs} long, " +
                             f"found: {len(short_runs)} short and {len(long_runs)} long.")

        training_data_short = pd.concat(short_runs).sort_values(
            "Relative_time").reset_index(drop=True)
        training_data_long = pd.concat(long_runs).sort_values(
            "Relative_time").reset_index(drop=True)
        return training_data_short, training_data_long

    # ---------------------------
    # Training & Prediction
    # ---------------------------
    def train_models(self, training_data_dir: str, required_runs=20, tune=True):
        """
        Loads training data, computes dynamic transition matrices,
        and trains both the short-run and long-run models.
        """
        training_data_short, training_data_long = self.load_and_preprocess_data_split(
            training_data_dir, required_runs)
        self.transition_matrix_short = self.compute_dynamic_transition_matrix(
            training_data_short)
        self.transition_matrix_long = self.compute_dynamic_transition_matrix(
            training_data_long)
        self.model_short, self.preprocessors_short, self.params_short = self.train_model_native(
            training_data_short, tune=tune)
        self.model_long, self.preprocessors_long, self.params_long = self.train_model_native(
            training_data_long, tune=tune)
        # Optionally, save the trained models:
        self.model_short.save_model('model_short.json')
        self.model_long.save_model('model_long.json')
        return (self.model_short, self.model_long)

    def train_model_native(self, training_data, numerical_features=None, categorical_features=None, target='Fill', tune=True):
        if numerical_features is None:
            numerical_features = self.features
        features = numerical_features + \
            (categorical_features if categorical_features else [])
        X = training_data[features].copy()
        y = training_data[target].values

        # Preprocess numerical features.
        num_imputer = SimpleImputer(strategy='mean')
        X_num = num_imputer.fit_transform(X[numerical_features])
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)

        # Optionally preprocess categorical features.
        if categorical_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_cat = cat_imputer.fit_transform(X[categorical_features])
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            X_cat = encoder.fit_transform(X_cat)
            X_processed = np.hstack([X_num, X_cat])
        else:
            X_processed = X_num

        dtrain = xgb.DMatrix(X_processed, label=y)
        base_params = {
            'objective': 'multi:softprob',
            'num_class': 5,
            'eval_metric': 'aucpr',
            'seed': 42
        }

        if tune:
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
                params['min_child_weight'] = int(
                    hyperparams['min_child_weight'])
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
            best = fmin(fn=objective, space=space,
                        algo=tpe.suggest, max_evals=10, trials=trials)
            best['max_depth'] = int(best['max_depth'])
            best['min_child_weight'] = int(best['min_child_weight'])
            params = base_params.copy()
            params.update(best)
            best_trial = min(trials.results, key=lambda x: x['loss'])
            optimal_rounds = best_trial['num_rounds']
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
        preprocessors = {'num_imputer': num_imputer, 'scaler': scaler}
        if categorical_features:
            preprocessors.update(
                {'cat_imputer': cat_imputer, 'encoder': encoder})
        return model, preprocessors, params

    def predict_native(self, model, preprocessors, X, numerical_features=None, categorical_features=None):
        if numerical_features is None:
            numerical_features = self.features
        X = X.copy()
        X_num = preprocessors['num_imputer'].transform(X[numerical_features])
        X_num = preprocessors['scaler'].transform(X_num)
        if categorical_features:
            X_cat = preprocessors['cat_imputer'].transform(
                X[categorical_features])
            X_cat = preprocessors['encoder'].transform(X_cat)
            X_processed = np.hstack([X_num, X_cat])
        else:
            X_processed = X_num
        dmatrix = xgb.DMatrix(X_processed)
        prob_matrix = model.predict(dmatrix)
        return prob_matrix

    def compute_dynamic_transition_matrix(self, training_data, num_states=5, smoothing=1e-6):
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

    def predict_ensemble(self, X, ensemble_weight=None):
        """
        Compute ensemble predictions from the two models.
        If no weight is provided, a heuristic is used based on time delta detection.
        """
        prob_matrix_short = self.predict_native(
            self.model_short, self.preprocessors_short, X)
        prob_matrix_long = self.predict_native(
            self.model_long, self.preprocessors_long, X)

        if ensemble_weight is None:
            delta_idx = self.find_time_delta(X)
            w_short = 0.7 if delta_idx != -1 else 0.3
            w_long = 1.0 - w_short
        else:
            w_short, w_long = ensemble_weight

        ensemble_prob_matrix = w_short * prob_matrix_short + w_long * prob_matrix_long
        ensemble_transition_matrix = w_short * \
            self.transition_matrix_short + w_long * self.transition_matrix_long
        ensemble_prediction = self.viterbi_decode(
            ensemble_prob_matrix, ensemble_transition_matrix)
        return ensemble_prediction

    # ---------------------------
    # Live Prediction & Visualization
    # ---------------------------
    def simulate_serial_stream(self, loaded_data, batch_size=100):
        num_rows = len(loaded_data)
        for start_idx in range(0, num_rows, batch_size):
            yield loaded_data.iloc[start_idx:start_idx+batch_size]

    def live_prediction_loop(self, loaded_data, batch_size=100, delay=0.1, ensemble_weight=None):
        accumulated_data = pd.DataFrame(columns=loaded_data.columns)
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        stream_generator = self.simulate_serial_stream(
            loaded_data, batch_size=batch_size)
        batch_num = 0

        for batch in stream_generator:
            batch_num += 1
            print(
                f"\n[INFO] Streaming batch {batch_num} with {len(batch)} data points...")
            time.sleep(delay)
            accumulated_data = pd.concat(
                [accumulated_data, batch], ignore_index=True)
            accumulated_data = self.compute_additional_features(
                accumulated_data)
            if len(accumulated_data) > self.ignore_before and batch_num == 1:
                accumulated_data = accumulated_data.iloc[self.ignore_before:]
            X_live = accumulated_data[self.features]
            ensemble_prediction = self.predict_ensemble(
                X_live, ensemble_weight)
            indices = np.arange(len(accumulated_data))
            ax.cla()
            ax.plot(indices, accumulated_data["Fill"], label="Actual", color='blue',
                    marker='o', linestyle='--', markersize=3)
            ax.plot(indices, ensemble_prediction, label="Ensemble Prediction",
                    color='red', marker='x', linestyle='-', markersize=3)
            delta_idx = self.find_time_delta(accumulated_data)
            if delta_idx != -1:
                ax.axvline(x=delta_idx, color='purple',
                           linestyle=':', label="Time Delta Detected")
            ax.set_title("Ensemble Predicted vs Actual Fill")
            ax.set_xlabel("Data Point Index")
            ax.set_ylabel("Fill Category (numeric)")
            ax.legend()
            ax.set_ylim(-0.5, 4.5)
            plt.pause(delay)
            print(
                f"[INFO] Updated ensemble live monitor after batch {batch_num}. Total points: {len(accumulated_data)}")

        plt.ioff()
        plt.show()

    # ---------------------------
    # Preprocessor Serialization Helpers
    # ---------------------------
    @staticmethod
    def save_params(params, filename):
        def convert(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            if isinstance(item, np.generic):
                return item.item()
            return item
        serializable = {k: convert(v) for k, v in params.items()}
        with open(filename, "w") as f:
            json.dump(serializable, f)

    @staticmethod
    def save_preprocessors(preprocessors, filename):
        processors_serialized = {}
        for name, processor in preprocessors.items():
            proc_dict = {"type": type(processor).__name__,
                         "init_params": processor.get_params()}
            if hasattr(processor, "mean_"):
                proc_dict["mean_"] = processor.mean_.tolist()
            if hasattr(processor, "scale_"):
                proc_dict["scale_"] = processor.scale_.tolist()
            if hasattr(processor, "var_"):
                proc_dict["var_"] = processor.var_.tolist()
            if hasattr(processor, "n_samples_seen_"):
                proc_dict["n_samples_seen_"] = processor.n_samples_seen_
            if hasattr(processor, "statistics_"):
                proc_dict["statistics_"] = processor.statistics_.tolist(
                ) if processor.statistics_ is not None else None
            if hasattr(processor, "categories_"):
                proc_dict["categories_"] = [cat.tolist()
                                            for cat in processor.categories_]
            processors_serialized[name] = proc_dict
        with open(filename, "w") as f:
            json.dump(processors_serialized, f)
        print(f"[INFO] Preprocessors saved to {filename}")

    @staticmethod
    def load_preprocessors(filename):
        with open(filename, "r") as f:
            processors_serialized = json.load(f)
        processors = {}
        for name, proc_dict in processors_serialized.items():
            proc_type = proc_dict["type"]
            init_params = proc_dict["init_params"]
            if proc_type == "StandardScaler":
                from sklearn.preprocessing import StandardScaler
                processor = StandardScaler(**init_params)
                processor.mean_ = np.array(proc_dict.get("mean_"))
                processor.scale_ = np.array(proc_dict.get("scale_"))
                processor.var_ = np.array(proc_dict.get("var_"))
                processor.n_samples_seen_ = proc_dict.get("n_samples_seen_")
            elif proc_type == "SimpleImputer":
                from sklearn.impute import SimpleImputer
                processor = SimpleImputer(**init_params)
                stats = proc_dict.get("statistics_")
                processor.statistics_ = np.array(
                    stats) if stats is not None else None
            elif proc_type == "OneHotEncoder":
                from sklearn.preprocessing import OneHotEncoder
                processor = OneHotEncoder(**init_params)
                cats = proc_dict.get("categories_")
                processor.categories_ = [
                    np.array(cat) for cat in cats] if cats is not None else None
            else:
                raise ValueError(f"Unknown processor type: {proc_type}")
            processors[name] = processor
        print(f"[INFO] Preprocessors loaded from {filename}")
        return processors


# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    # Create the ensemble model instance.
    config = {"features": DEFAULT_FEATURES, "ignore_before": 50}
    ensemble = DissipationEnsembleModel(config)

    # Training phase (using your training data directory).
    training_data_dir = r"content\training_data\full_fill"  # Update as needed.
    ensemble.train_models(training_data_dir, required_runs=20, tune=True)

    # Load models and preprocessors if needed:
    ensemble.model_short = xgb.Booster()
    ensemble.model_short.load_model('model_short.json')
    ensemble.model_long = xgb.Booster()
    ensemble.model_long.load_model('model_long.json')
    ensemble.preprocessors_short = ensemble.load_preprocessors(
        "preprocessors_short.json")
    ensemble.preprocessors_long = ensemble.load_preprocessors(
        "preprocessors_long.json")

    # Simulate live predictions on test data.
    test_dir = r"content\test_data"
    test_content = ensemble.load_content(test_dir)
    random.shuffle(test_content)
    for data_file, poi_file in test_content:
        df_test = pd.read_csv(data_file)
        delay = df_test['Relative_time'].max(
        ) / len(df_test["Relative_time"].values)
        # Or use a dedicated single-file loader.
        loaded_data = pd.read_csv(data_file)
        # Optionally, precompute additional features:
        loaded_data = ensemble.compute_additional_features(loaded_data)
        ensemble.live_prediction_loop(
            loaded_data, batch_size=100, delay=delay/2)
        print(
            "\n[INFO] Ensemble live prediction simulation complete for one test run.")
    print("\n[INFO] All ensemble live prediction simulations complete.")
