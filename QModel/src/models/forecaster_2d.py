import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch, savgol_filter
from scipy.stats import skew, kurtosis, zscore
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Global parameters for data processing
# Example thresholds for filtering based on Relative_time
SPIN_UP_TIME = (0.1, 1.0)
DOWNSAMPLE_FACTOR = 1       # Example: use every row; adjust as needed

# =============================================================================
# Data Loading & Feature Extraction Component
# =============================================================================


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
        required_columns = ["Dissipation",
                            "Resonance_Frequency", "Relative_time"]
        if not all(column in df.columns for column in required_columns):
            raise ValueError(
                f"Input DataFrame must contain the following columns: {required_columns}")

        sigma = 2
        df['Dissipation_DoG'] = gaussian_filter1d(
            df['Dissipation'], sigma=sigma, order=1)

        # Rolling Baseline for Trends
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['DoG_baseline'] = df['Dissipation_DoG'].rolling(
            window=baseline_window, center=True, min_periods=1).median()
        df['DoG_shift'] = df['Dissipation_DoG'] - df['DoG_baseline']

        # One-Class SVM for Anomaly Detection
        X_dog = df['DoG_shift'].values.reshape(-1, 1)
        ocsvm = OneClassSVM(nu=0.05, kernel='rbf',
                            gamma='scale', shrinking=False)
        ocsvm.fit(X_dog)
        df['DoG_SVM_Score'] = ocsvm.decision_function(X_dog)

        # Spectral Features via Welch's method
        fs = 1 / np.mean(np.diff(df['Relative_time']))
        f, Pxx = welch(df["DoG_SVM_Score"], fs=fs, nperseg=min(256, len(df)))
        df["DoG_SVM_Spectral_Entropy"] = - \
            np.sum((Pxx / np.sum(Pxx)) * np.log2(Pxx / np.sum(Pxx)))
        df["DoG_SVM_Dominant_Frequency"] = f[np.argmax(Pxx)]

        # Smoothing & Signal Filtering using Savitzky–Golay filter
        window_size = min(11, len(df) - 1) if len(df) > 11 else 5
        df["DoG_SVM_Smooth"] = savgol_filter(
            df["DoG_SVM_Score"], window_size, polyorder=2)

        # Rolling Statistics for Monitoring
        df["DoG_SVM_Skew"] = df["DoG_SVM_Score"].rolling(
            baseline_window, min_periods=1).apply(skew, raw=True)
        df["DoG_SVM_Kurtosis"] = df["DoG_SVM_Score"].rolling(
            baseline_window, min_periods=1).apply(kurtosis, raw=True)
        df["DoG_SVM_Variability"] = df["DoG_SVM_Score"].rolling(
            baseline_window, min_periods=1).std()

        # Anomaly Monitoring with Z-Score
        df["DoG_SVM_Zscore"] = zscore(df["DoG_SVM_Score"], nan_policy="omit")
        df["DoG_SVM_Anomaly"] = (np.abs(df["DoG_SVM_Zscore"]) > 2).astype(int)

        # Clustering Anomalies (DBSCAN) to Reduce Noise
        anomaly_indices = df[df["DoG_SVM_Anomaly"] == 1].index
        if len(anomaly_indices) > 0:
            dbscan = DBSCAN(eps=5, min_samples=2)
            clustered_labels = dbscan.fit_predict(
                anomaly_indices.values.reshape(-1, 1))
            clustered_anomalies = np.zeros(len(df))
            for cluster in set(clustered_labels):
                if cluster != -1:
                    first_index = anomaly_indices[clustered_labels == cluster][0]
                    clustered_anomalies[first_index] = 1
            df["DoG_SVM_Anomaly"] = clustered_anomalies

        # Difference Factor Computation
        if "Difference" not in df.columns:
            df["Difference"] = [0] * len(df)

        xs = df["Relative_time"]
        i = next((x for x, t in enumerate(xs) if t > 0.5), None)
        j = next((x for x, t in enumerate(xs) if t > 2.5), None)
        if i is not None and j is not None:
            avg_resonance_frequency = df["Resonance_Frequency"][i:j].mean()
            avg_dissipation = df["Dissipation"][i:j].mean()
            df["ys_diss"] = (df["Dissipation"] - avg_dissipation) * \
                avg_resonance_frequency / 2
            df["ys_freq"] = avg_resonance_frequency - df["Resonance_Frequency"]
            difference_factor = 3
            df["Difference"] = df["ys_freq"] - \
                difference_factor * df["ys_diss"]

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def _process_fill(df: pd.DataFrame, poi_file: str) -> pd.DataFrame:
        """
        Process fill information from the poi_file.
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
        df["Fill"] = df["Fill"].apply(QForecasterDataprocessor.reassign_region)
        mapping = {'no_fill': 0, 'init_fill': 1,
                   'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
        df["Fill"] = df["Fill"].map(mapping)
        return df

    @staticmethod
    def load_and_preprocess_data(data_dir: str, num_datasets: int):
        """
        Loads multiple datasets from data_dir, applies fill processing and feature extraction,
        and returns a concatenated training DataFrame.
        """
        runs = []
        content = QForecasterDataprocessor.load_content(data_dir)
        random.shuffle(content)
        if num_datasets < len(content):
            content = content[:num_datasets]
        for file, poi_file in content:
            df = pd.read_csv(file)
            required_cols = ["Relative_time",
                             "Dissipation",
                             "Resonance_Frequency",
                             ]
            if df.empty or not all(col in df.columns for col in required_cols):
                continue
            df = df[required_cols]
            try:
                df = QForecasterDataprocessor._process_fill(
                    df, poi_file=poi_file)
            except FileNotFoundError:
                df = None
            if df is None:
                continue
            df = df[df["Relative_time"] >= random.uniform(
                SPIN_UP_TIME[0], SPIN_UP_TIME[1])]
            df = df.iloc[::DOWNSAMPLE_FACTOR]
            df = df.reset_index(drop=True)
            if df.empty:
                continue
            df.loc[df['Fill'] == 0, 'Fill'] = 1
            df['Fill'] -= 1  # In-place adjustment
            df = QForecasterDataprocessor.compute_additional_features(df)
            df.reset_index(drop=True, inplace=True)
            runs.append(df)
        training_data = pd.concat(runs).sort_values(
            "Relative_time").reset_index(drop=True)
        training_data.drop(columns=['Relative_time'], inplace=True)
        return training_data

# =============================================================================
# ML Component: Training and Live Prediction
# =============================================================================


class FillPredictor:
    def __init__(self, window_size=20, feature_columns=None, num_classes=5):
        """
        Initializes the FillPredictor.
        Parameters:
            window_size (int): Number of time steps for each training sample.
            feature_columns (list): List of feature column names.
            num_classes (int): Number of target classes.
        """
        self.window_size = window_size
        self.feature_columns = feature_columns  # If None, will use all except 'Fill'
        self.num_classes = num_classes
        self.model = None
        self.buffer = []  # Rolling buffer for live predictions

    def create_sequences(self, df):
        """
        Creates sliding-window sequences from the DataFrame.
        Each sample consists of a window of features with the target being the next Fill label.
        """
        data = df.copy()
        if self.feature_columns is None:
            self.feature_columns = [
                col for col in data.columns if col != 'Fill']
        X = data[self.feature_columns].values
        y = data['Fill'].values
        X_seq, y_seq = [], []
        for i in range(len(X) - self.window_size):
            X_seq.append(X[i:i+self.window_size])
            y_seq.append(y[i+self.window_size])
        return np.array(X_seq), np.array(y_seq)

    def build_model(self):
        """
        Builds an LSTM-based sequential model.
        """
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(
            self.window_size, len(self.feature_columns))))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=['accuracy'])
        self.model = model

    def train(self, df, epochs=20, batch_size=32, validation_split=0.2):
        """
        Trains the LSTM model using historical data.
        """
        X, y = self.create_sequences(df)
        if self.model is None:
            self.build_model()
        early_stop = EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_split, callbacks=[early_stop])
        return history

    def update_buffer(self, new_data):
        """
        Adds a new observation to the rolling buffer.
        new_data: list/array of feature values (order must match self.feature_columns).
        """
        self.buffer.append(new_data)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]

    def predict_next(self):
        """
        Predicts the next Fill label based on the current rolling buffer.
        """
        if len(self.buffer) < self.window_size:
            print("Buffer not filled yet! Waiting for more data...")
            return None
        window = np.array(self.buffer).reshape(
            1, self.window_size, len(self.feature_columns))
        pred = self.model.predict(window)
        predicted_class = np.argmax(pred, axis=1)[0]
        return predicted_class


# =============================================================================
# Main: Load Data, Train Model, and Simulate Live Predictions
# =============================================================================
if __name__ == '__main__':
    # Set your data directory and number of datasets to load
    # Replace with your actual data directory
    data_dir = 'content/training_data/full_fill'
    num_datasets = 5                  # Adjust as needed

    print("Loading and preprocessing training data...")
    training_data = QForecasterDataprocessor.load_and_preprocess_data(
        data_dir, num_datasets)
    print("Training data shape:", training_data.shape)

    # Define feature columns as all except 'Fill'
    feature_columns = [col for col in training_data.columns if col != "Fill"]

    # Instantiate and train the predictor
    predictor = FillPredictor(
        window_size=20, feature_columns=feature_columns, num_classes=5)
    print("Training the model...")
    predictor.train(training_data, epochs=20,
                    batch_size=32, validation_split=0.2)
    print("Training completed.")

    # Simulate live predictions by iterating over the training data.
    print("\nSimulating live predictions:")
    for idx, row in training_data.iterrows():
        new_obs = list(row[feature_columns])
        predictor.update_buffer(new_obs)
        pred = predictor.predict_next()
        if pred is not None:
            print(f"At index {idx}: Predicted next Fill label: {pred}")
