from scipy.signal import find_peaks  # Make sure SciPy is available
from sklearn.svm import OneClassSVM
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from typing import Any, Dict, List, Optional, Tuple
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

FEATURES = [
    'Dissipation',
    'Dissipation_rolling_mean',
    'Dissipation_rolling_median',
    'Dissipation_ewm',
    'Dissipation_rolling_std',
    'Dissipation_diff',
    'Dissipation_pct_change',
    'Dissipation_ratio_to_mean',
    'Dissipation_ratio_to_ewm',
    'Dissipation_envelope'
]
NUM_CLASSES = 4
TARGET = "Fill"
DOWNSAMPLE_FACTOR = 5
SPIN_UP_TIME = (1.2, 1.4)
BASELINE_WINDOW = 100


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
                f"Input DataFrame must contain the following columns: {required_columns}"
            )

        sigma = 2
        df['Dissipation_DoG'] = gaussian_filter1d(
            df['Dissipation'], sigma=sigma, order=1)

        # Estimate a baseline using a rolling median
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['DoG_baseline'] = df['Dissipation_DoG'].rolling(
            window=baseline_window, center=True, min_periods=1
        ).median()
        df['DoG_shift'] = df['Dissipation_DoG'] - df['DoG_baseline']

        # Anomaly detection with One-Class SVM
        X_dog = df['DoG_shift'].values.reshape(-1, 1)
        ocsvm = OneClassSVM(nu=0.05, kernel='rbf',
                            gamma='scale', shrinking=False)
        ocsvm.fit(X_dog)
        df['DoG_SVM_Score'] = ocsvm.decision_function(X_dog)
        # --- Difference Factor Computation --- #
        if "Difference" not in df.columns:
            df['Difference'] = [0] * len(df)

        xs = df["Relative_time"]
        i = next((x for x, t in enumerate(xs) if t > 0.5), None)
        j = next((x for x, t in enumerate(xs) if t > 2.5), None)

        if i is not None and j is not None:
            avg_resonance_frequency = df["Resonance_Frequency"][i:j].mean()
            avg_dissipation = df["Dissipation"][i:j].mean()

            df["ys_diss"] = (df["Dissipation"] -
                             avg_dissipation) * avg_resonance_frequency / 2
            df["ys_freq"] = avg_resonance_frequency - \
                df["Resonance_Frequency"]
            difference_factor = 3
            df["Difference"] = df["ys_freq"] - \
                difference_factor * df["ys_diss"]

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def _process_fill(df: pd.DataFrame, poi_file: str) -> pd.DataFrame:
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
        df["Fill"] = df["Fill"].apply(QForecasterDataprocessor.reassign_region)
        mapping = {'no_fill': 0, 'init_fill': 1,
                   'ch_1': 2, 'ch_2': 3, 'full_fill': 4}
        df["Fill"] = df["Fill"].map(mapping)
        return df

    @staticmethod
    def load_and_preprocess_data(data_dir: str, num_datasets: int):
        runs = []
        content = QForecasterDataprocessor.load_content(data_dir)
        random.shuffle(content)

        if num_datasets < len(content):
            content = content[:num_datasets]

        for file, poi_file in content:
            df = pd.read_csv(file)
            required_cols = ["Relative_time", "Dissipation"]
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
            df = df.reset_index()
            init_fill_point = QForecasterDataprocessor.init_fill_point(
                df, BASELINE_WINDOW, 10)
            df = df.iloc[init_fill_point:]
            if df is None or df.empty:
                continue
            df.loc[df['Fill'] == 0, 'Fill'] = 1
            # Method 2 (in-place subtraction):
            df['Fill'] -= 1
            df = QForecasterDataprocessor.compute_additional_features(df)
            df.reset_index(inplace=True)

            runs.append(df)

        training_data = pd.concat(runs).sort_values(
            "Relative_time").reset_index(drop=True)
        training_data.drop(columns=['Relative_time'], inplace=True)
        return training_data

    @staticmethod
    def init_fill(df):
        baseline_diff = df['Difference'].iloc[:100].mean()
        baseline_rf = df['Resonance_Frequency'].iloc[:100].mean()
        baseline_diss = df['Dissipation'].iloc[:100].mean()
        baseline_dog = df['DoG_SVM_Score'].iloc[:100].mean()

        std_diff = df['Difference'].iloc[:100].std()
        std_rf = df['Resonance_Frequency'].iloc[:100].std()
        std_diss = df['Dissipation'].iloc[:100].std()

        threshold_diff = baseline_diff + std_diff
        threshold_rf = baseline_rf - std_rf
        threshold_diss = baseline_diss + std_diss * 3

        baselines = {
            'Difference': baseline_diff,
            'Resonance_Frequency': baseline_rf,
            'Dissipation': baseline_diss,
            'DoG_SVM_Score': baseline_dog
        }

        # Only consider rows where 'Relative_time' > 3.0 (preserving original indices)
        df_after3 = df[df['Relative_time'] > 3.0]

        # Iterate through df_after3 to find the first candidate shift point
        for idx, row in df_after3.iterrows():
            # if (row['Difference'] > threshold_diff and
            #     row['Resonance_Frequency'] < threshold_rf and
            #         row['Dissipation'] > threshold_diss):
            if (row['Dissipation'] > threshold_diss):
                pos = df_after3.index.get_loc(idx)
                window_start = max(pos - 2, 0)
                window_end = min(pos + 3, len(df_after3))  # non-inclusive end
                dog_window = df_after3['DoG_SVM_Score'].iloc[window_start:window_end]
                if (dog_window < baseline_dog).sum() >= 3:
                    return idx, baselines, df_after3
        return -1, baselines, df_after3

    @staticmethod
    def post_point(init_fill_index, baselines, df: pd.DataFrame, stable_window=3):
        baseline_dog = baselines['DoG_SVM_Score']

        # Get the candidate's integer location in the filtered DataFrame
        # try:
        #     candidate_pos = df_after3.index.get_loc(init_fill_index)
        # except KeyError:
        #     return -1, -1
        tolerance = 0.001
        # Iterate forward from the candidate's position to find an initial stable window
        print(init_fill_index, len(df))
        for pos in range(init_fill_index, len(df) - stable_window + 1):
            window_scores = df['DoG_SVM_Score'].iloc[pos: pos +
                                                     stable_window].mean()
            # Check if all scores in the window are at or above baseline
            print(window_scores, baseline_dog, abs(
                window_scores - baseline_dog))
            if abs(window_scores - baseline_dog) <= tolerance:
                stable_start = df.index[pos]
                stable_end_pos = pos + stable_window
                while stable_end_pos < len(df) and df['DoG_SVM_Score'].iloc[stable_end_pos] >= baseline_dog:
                    stable_end_pos += 1
                # The last index that still met the baseline
                stable_end = df.index[stable_end_pos - 1]
                return stable_start, stable_end
        return -1, -1

    @staticmethod
    def ch1_point(df: pd.DataFrame) -> int:
        pass

    @staticmethod
    def ch2_point(df: pd.DataFrame) -> int:
        pass

    @staticmethod
    def ch3_point(df: pd.DataFrame) -> int:
        pass

    @staticmethod
    def exit_point(df: pd.DataFrame) -> int:
        pass
###############################################################################
# Trainer Class: Handles model training, hyperparameter tuning, and saving.
###############################################################################


class QForecasterPredictor:
    """Predictor class for QForecaster that handles model loading, data accumulation, and prediction.

    This class loads a pre-trained XGBoost model along with its associated preprocessors and transition
    matrix. It provides methods to apply preprocessing to input data, generate predictions using the
    model and Viterbi decoding, and maintain a history of predictions to assess stability.
    """

    def __init__(
        self,
        batch_threshold: int = 60
    ) -> None:
        """Initializes the QForecasterPredictor.

        Args:
            numerical_features (List[str], optional): List of feature names used for numerical processing.
                Defaults to FEATURES.
            target (str, optional): The target variable name. Defaults to 'Fill'.
            save_dir (Optional[str], optional): Directory path from which to load model files and preprocessors.
                Defaults to None.
            batch_threshold (int, optional): The rate at which to process batches of data. Defaults to 60.
        """
        self.batch_threshold: int = batch_threshold
        self.accumulated_data: Optional[pd.DataFrame] = None
        self.batch_num: int = 0

        self.init_fill_points = []
        self.ch1_fill_points = []
        self.ch2_fill_points = []
        self.ch3_fill_points = []
        self.exit_fill_points = []

        self.init_fill_point = -1
        self.post_fill_point = -1
        self.ch1_fill_point = -1
        self.ch2_fill_point = -1
        self.exit_fill_point = -1

        self.baselines = None
        self.df_after3 = None

    def update_predictions(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Accumulates new data, generates predictions, and assesses prediction stability.

        Once the candidate shift point (init_fill_point) and the stable return point
        (post_fill_point) are found, they are stored and never updated in subsequent batches.
        """
        # Accumulate new data
        if self.accumulated_data is None:
            self.accumulated_data = pd.DataFrame(columns=new_data.columns)
        # Remove columns that are completely empty (all NA) from new_data
        new_data = new_data.dropna(axis=1, how='all')
        self.accumulated_data = pd.concat(
            [self.accumulated_data, new_data], ignore_index=True
        )
        current_count = len(self.accumulated_data)

        # Compute additional features
        self.accumulated_data = QForecasterDataprocessor.compute_additional_features(
            self.accumulated_data
        )
        if self.init_fill_point == -1:
            init_fill_point, baselines, df_after3 = QForecasterDataprocessor.init_fill(
                self.accumulated_data
            )
            self.init_fill_point = init_fill_point
            self.baselines = baselines
            self.df_after3 = df_after3

        if self.post_fill_point == -1 and self.init_fill_point > -1:
            post_fill_point, _ = QForecasterDataprocessor.post_point(
                self.init_fill_point, self.baselines, self.accumulated_data, stable_window=5
            )
            print(f'Setting post point to: {post_fill_point}')
            self.post_fill_point = post_fill_point

        self.batch_num += 1
        return {
            "status": "completed",
            "accumulated_data": self.accumulated_data,
            "accumulated_count": current_count,
            "init_fill_point": self.init_fill_point,
            "post_fill_point": self.post_fill_point,
            "ch1_fill_point": self.ch1_fill_point,
            "ch2_fill_point": self.ch2_fill_point,
            "exit_fill_point": self.exit_fill_point,
        }

    def reset_accumulator(self) -> None:
        """Resets the accumulated data and batch counter.

        This method clears the internal DataFrame that accumulates new data and resets the batch number to zero.
        """
        self.accumulated_data = None
        self.batch_num = 0

    def _return_waiting_state(self, current_count: int) -> Dict[str, Any]:
        return {
            "status": "waiting",
            "accumulated_data": self.accumulated_data,
            "accumulated_count": current_count
        }


def simulate_serial_stream_from_loaded(loaded_data):
    """
    Simulate a serial data stream by yielding random chunks from loaded_data.

    Each chunk will have a random size between 100 and 200 rows.

    Args:
        loaded_data (pd.DataFrame): The loaded dataset to stream.

    Yields:
        pd.DataFrame: A chunk of data with a random number of rows.
    """
    num_rows = len(loaded_data)
    start_idx = 0
    while start_idx < num_rows:
        batch_size = random.randint(10, 30)
        yield loaded_data.iloc[start_idx:start_idx + batch_size]
        start_idx += batch_size


class QForecasterSimulator:
    def __init__(self, predictor: QForecasterPredictor, dataset: pd.DataFrame, poi_file: pd.DataFrame = None, ignore_before=0, delay=1.0):
        """
        Simulator class to stream data in random chunk sizes and update predictions live.
        Args:
            predictor (QForecasterPredictor): Predictor object.
            dataset (pd.DataFrame): Dataset to simulate streaming.
            poi_file (pd.DataFrame, optional): DataFrame containing Points Of Interest indices.
            ignore_before (int): Number of initial rows to ignore.
            delay (float): Delay (in seconds) between processing batches.
        """
        self.predictor = predictor
        self.dataset = dataset
        self.ignore_before = ignore_before
        self.delay = delay

        # If a POI file is provided, extract the actual indices.
        if poi_file is not None:
            self.actual_poi_indices = poi_file
        else:
            self.actual_poi_indices = np.array([])

        # Setup a live plot with a single axis.
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.ion()  # Enable interactive mode.
        plt.show()

    def run(self):
        """
        Runs the simulation by iterating through the dataset in random chunk sizes,
        updating predictions, and plotting the current state.
        """
        self.predictor.reset_accumulator()
        batch_number = 0
        for batch_data in simulate_serial_stream_from_loaded(self.dataset):
            batch_number += 1
            # print(
            #     f"[INFO] Processing batch {batch_number} with {len(batch_data)} rows.")

            # Update predictions using the new batch.
            results = self.predictor.update_predictions(batch_data)

            # Update the live plot.
            self.plot_results(results, batch_number=batch_number)

            # Simulate processing delay.
            plt.pause(self.delay)

        plt.ioff()  # Turn off interactive mode.
        plt.show()

    def plot_results(self, results: dict, batch_number: int, column_to_plot='DoG_SVM_Score'):
        """
        Updates the live plot with two subplots:
        - Top subplot: Normalized curve for the specified column (e.g., DoG_SVM_Score).
        - Bottom subplot: Curve for the 'Dissipation' column data.

        Also overlays background shading for predicted class regions, marks trough clusters,
        and plots anomalous points. The baseline for trough detection is calculated using the
        first 100 data points.
        """
        accumulated_data = results.get("accumulated_data")
        if accumulated_data is None or accumulated_data.empty:
            print("[WARN] No accumulated data available for plotting.")
            return

        x = np.arange(len(accumulated_data))
        self.fig.clear()
        init_fill_point = results['init_fill_point']
        post_fill_point = results['post_fill_point']
        # --- Top Plot: Primary Data (e.g., DoG_SVM_Score) ---
        ax1 = self.fig.add_subplot(211)
        data = accumulated_data[column_to_plot].values
        ax1.plot(x, data, color="red", linewidth=1.5, label=column_to_plot)
        if hasattr(self, "actual_poi_indices") and self.actual_poi_indices.size > 0:
            valid_actual_indices = [
                int(idx) for idx in self.actual_poi_indices if idx < len(accumulated_data)]
            if valid_actual_indices:
                y_actual = data[valid_actual_indices]
                ax1.scatter(valid_actual_indices, y_actual, color="#2ca02c", marker="o", s=50,
                            label=f"Actual POI {self.actual_poi_indices.size}")

        # --- Bottom Plot: Dissipation Data ---
        if "Dissipation" in accumulated_data.columns:
            ax2 = self.fig.add_subplot(212)
            dissipation_data = accumulated_data["Dissipation"].values
            ax2.plot(x, dissipation_data, color="blue",
                     linewidth=1.5, label="Dissipation")
            if hasattr(self, "actual_poi_indices") and self.actual_poi_indices.size > 0:
                valid_actual_indices = [
                    int(idx) for idx in self.actual_poi_indices if idx < len(accumulated_data)]
                if valid_actual_indices:
                    y_actual = dissipation_data[valid_actual_indices]
                    ax2.scatter(valid_actual_indices, y_actual, color="#2ca02c", marker="o", s=50,
                                label=f"Actual POI {self.actual_poi_indices.size}")
            ax2.set_title(
                f"Dissipation Data (Batch {batch_number})", fontsize=14, weight="medium")
            ax2.set_xlabel("Data Index", fontsize=12)
            ax2.set_ylabel("Dissipation", fontsize=12)
            ax2.tick_params(axis="both", which="major", labelsize=10)
            ax2.legend(frameon=False, fontsize=10)
        else:
            print("[WARN] 'Dissipation' column not found in accumulated_data.")

        if init_fill_point > -1:
            ax1.axvline(init_fill_point, color='y')
            ax2.axvline(init_fill_point, color='y')

        if post_fill_point > -1:
            ax1.axvline(post_fill_point, color='orange')
            ax2.axvline(post_fill_point, color='orange')
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


TESTING = True
TRAINING = False
# Main execution block.
if __name__ == '__main__':
    SAVE_DIR = r'QModel\SavedModels\forecaster_v3'
    if TRAINING:
        train_dir = r'content\long_tails'
        training_data = QForecasterDataprocessor.load_and_preprocess_data(
            train_dir, num_datasets=100)

        qft = QForecasterTrainer(FEATURES, TARGET, SAVE_DIR)
        qft.train_model(training_data=training_data, tune=True)
        qft.save_models()

    if TESTING:
        test_dir = r"content\test_data"
        test_content = QForecasterDataprocessor.load_content(test_dir)
        random.shuffle(test_content)
        for data_file, poi_file in test_content:
            dataset = pd.read_csv(data_file)
            end = np.random.randint(0, len(dataset))
            end = random.randint(min(end, len(dataset)),
                                 max(end, len(dataset)))
            random_slice = dataset.iloc[0:len(dataset)-1]

            # Load the POI file which is a flat list of 6 indices from the original dataset.
            poi_df = pd.read_csv(poi_file, header=None)
            poi_indices_original = poi_df.to_numpy().flatten()

            # Use these indices to extract the corresponding relative times from the original dataset.
            original_relative_times = dataset.loc[poi_indices_original,
                                                  'Relative_time'].values

            predictor = QForecasterPredictor()
            delay = dataset['Relative_time'].max() / len(random_slice)
            random_slice = random_slice[random_slice["Relative_time"] >= random.uniform(
                SPIN_UP_TIME[0], SPIN_UP_TIME[1])]
            # Downsample the data.
            random_slice = random_slice.iloc[::DOWNSAMPLE_FACTOR]

            # Get the downsampled relative times.
            downsampled_times = random_slice["Relative_time"].values

            # For each original relative time, find the index in the downsampled data
            # that has the closest relative time.
            mapped_poi_indices = []
            for orig_time in original_relative_times:
                idx = np.abs(downsampled_times - orig_time).argmin()
                mapped_poi_indices.append(idx)
            mapped_poi_indices = np.array(mapped_poi_indices)

            # Create the simulator, passing the mapped POI indices.
            simulator = QForecasterSimulator(
                predictor,
                random_slice,
                poi_file=mapped_poi_indices,
                ignore_before=50,
                delay=delay
            )

            # Run the simulation.
            simulator.run()
