import pandas as pd
import os
import csv
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt, detrend
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from QConstants import *


class QDataPipeline:

    def __init__(self, data_filepath: str = None, multi_class: bool = False) -> None:
        """
        Initializes the QDataPipeline object.

        Parameters:
        - data_filepath (str): The path to the CSV file containing the dataset.
        - multi_class (bool): Flag indicating whether the classification problem is multi-class. Defaults to False.

        Raises:
        - ValueError: If `data_filepath` is not provided.
        """
        if data_filepath is not None:
            self.data_filepath = data_filepath
            self.dataframe = pd.read_csv(
                self.data_filepath
            )  # Load the CSV into a DataFrame
            self.multi_class = multi_class  # Set the multi_class flag
        else:
            raise ValueError(
                f"[QDataPipeline.__init__] Filepath required, found {data_filepath}."
            )

    def preprocess(self, poi_filepath: str = None) -> None:
        # STEP 0
        # Drop unecessary columns.
        self.__dataframe__.drop(
            columns=[
                "Date",
                "Time",
                "Ambient",
                "Temperature",
                "Peak Magnitude (RAW)",
            ],
            inplace=True,
        )

        # STEP 1
        # Compute the difference curve of this dataframe.
        self.compute_difference()

        # OPTIONAL
        # For training datasets, add the POI flags to the dataframe.
        if poi_filepath is not None:
            self.add_class(poi_filepath)

        # Compute the 0.5% smoothing quantity for this dataset.
        smooth_win = int(0.005 * len(self.__dataframe__["Relative_time"].values))
        if smooth_win % 2 == 0:
            smooth_win += 1
        if smooth_win <= 1:
            smooth_win = 2

        # STEP 3
        # Compute the super gradient of the dissipation, difference, cumulative,
        # and resonance frequency curves.
        self.super_gradient("Dissipation")
        self.super_gradient("Difference")
        self.super_gradient("Cumulative")
        self.super_gradient("Resonance_Frequency")

        # STEP 4
        # Smooth the dissipation, difference, and resonance frequency curves
        # using dynamic window size and polyorder 1
        self.compute_smooth(column="Dissipation", winsize=smooth_win, polyorder=1)
        self.compute_smooth(column="Difference", winsize=smooth_win, polyorder=1)
        self.compute_smooth(
            column="Resonance_Frequency", winsize=smooth_win, polyorder=1
        )

        # STEP 5
        # Compute the gradient of the smooothed dissipation, difference, and resonance
        # frequency curves.
        self.compute_gradient(column="Dissipation")
        self.compute_gradient(column="Difference")
        self.compute_gradient(column="Resonance_Frequency")

        # STEP 6
        # Filter for noise among all curves eliminating lowpass jitter in the signal
        # for each curve.
        self.noise_filter("Cumulative")
        self.noise_filter("Dissipation")
        self.noise_filter("Difference")
        self.noise_filter("Resonance_Frequency")
        self.noise_filter("Difference_gradient")
        self.noise_filter("Dissipation_gradient")
        self.noise_filter("Resonance_Frequency_gradient")

        # STEP 7
        # Detrend Cumulative, Dissipation, Resonance Frequency, and Difference curves.
        self.remove_trend("Cumulative")
        self.remove_trend("Dissipation")
        self.remove_trend("Resonance_Frequency")
        self.remove_trend("Difference")

    def find_time_delta(self) -> int:
        """
        Finds the first significant time delta change in the 'Relative_time' column of the dataframe.

        Returns:
        - idx (int): The index of the first significant change in time delta.
                    Returns -1 if no significant change is found.
        """
        # Calculate the time difference (delta) between consecutive rows in the 'Relative_time' column
        time_df = pd.DataFrame()
        time_df["Delta"] = self.dataframe["Relative_time"].diff()

        # Define the threshold for detecting a significant change
        threshold = 0.032

        # Calculate the expanding (cumulative) mean of the time deltas
        rolling_avg = time_df["Delta"].expanding(min_periods=2).mean()

        # Identify where the absolute difference between the delta and its rolling average exceeds the threshold
        time_df["Significant_change"] = (
            time_df["Delta"] - rolling_avg
        ).abs() > threshold

        # Get the indices where a significant change occurs
        change_indices = time_df.index[time_df["Significant_change"]].tolist()

        # Determine if there is any significant change
        has_significant_change = len(change_indices) > 0

        # If a significant change is found, return the first index; otherwise, return -1
        idx = change_indices[0] if has_significant_change else -1

        return idx

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the DataFrame containing the loaded data.

        Returns:
            pd.DataFrame: The loaded data.
        """
        return self.dataframe

    def super_gradient(self, column: str) -> None:
        """
        Computes the smoothed first derivative of the specified column using the Savitzky-Golay filter.

        The resulting gradient is normalized and stored in a new column with the suffix '_super'.

        Parameters:
            column (str): The name of the column to compute the gradient for.

        Modifies:
            Adds a new column to the dataframe with the name '<column>_super'.
        """
        # Generate the name for the new column where the gradient will be stored
        name = f"{column}_super"

        # Extract the data from the specified column
        data = self.dataframe[column]

        # Determine the window size for the Savitzky-Golay filter, ensuring it's an odd number and at least 3
        window = int(len(data) * 0.01)
        if window % 2 == 0:
            window += 1
        if window <= 1:
            window = 3

        # Apply the Savitzky-Golay filter to smooth the data
        smoothed_data = savgol_filter(
            x=data, window_length=window, polyorder=1, deriv=0
        )

        # Calculate the first derivative (gradient) of the smoothed data
        gradient = savgol_filter(
            x=smoothed_data, window_length=window, polyorder=1, deriv=1
        )

        # Normalize the gradient and store it in the new column
        self.dataframe[name] = self.normalize_data(gradient)

    def standardize(self, column):
        # define standard scaler
        scaler = StandardScaler()
        df = self.__dataframe__
        # transform data
        self.__dataframe__[column] = scaler.fit_transform(
            np.array(df[column]).reshape(-1, 1)
        )

    def remove_trend(self, column: str) -> None:
        """
        Removes the linear trend from the specified column of the dataframe.

        The detrended data is stored in a new column with the suffix '_detrend'.

        Parameters:
            column (str): The name of the column from which the trend will be removed.

        Modifies:
            Adds a new column to the dataframe with the name '<column>_detrend'.
        """
        # Generate the name for the new column where the detrended data will be stored
        name = f"{column}_detrend"

        # Remove the linear trend from the specified column and store it in the new column
        self.dataframe[name] = detrend(data=self.dataframe[column].values)

    def interpolate(self, num_rows: int = 20) -> None:
        """
        *** THIS FUNCTION IS CURRENTLY TOO COSTLY TO INTEGRATE
        Interpolates the dataframe rows between specific points if the 'Relative_time' exceeds 90.

        Parameters:
            num_rows (int): The number of rows to interpolate between each pair of points. Defaults to 20.

        Modifies:
            Updates the dataframe with the interpolated rows, adding them after the specified start index.
        """
        df = self.dataframe

        # Check if 'Relative_time' exceeds 90 and find the start index
        if max(df["Relative_time"].values) > 90:
            start_index = df.index[df["Relative_time"] > 90].tolist()[0]

            # Initialize an empty DataFrame to store the result
            result = pd.DataFrame(columns=df.columns)

            # Iterate through the dataframe to interpolate rows where needed
            for i in range(len(df) - 1):
                # Append the current row to the result DataFrame
                result = pd.concat(
                    [result, pd.DataFrame(df.iloc[i]).transpose()],
                    ignore_index=True,
                )

                # If the current index is greater than or equal to start_index, interpolate rows
                if i >= start_index:
                    self.interpolation_size += 1
                    start_row = df.iloc[i]
                    end_row = df.iloc[i + 1]

                    # Prepare a DataFrame to store interpolated rows
                    interpolated_rows = pd.DataFrame()

                    for col in df.columns:
                        if col in ["Class_1", "Class_2"]:
                            # If the original value is 0, interpolate to 0
                            if start_row[col] == 0:
                                interpolated_rows[col] = np.repeat(0, num_rows, axis=0)
                            else:
                                # Otherwise, repeat the start value and fill the remaining with 0
                                interpolated_rows[col] = np.concatenate(
                                    (
                                        np.repeat(start_row[col], 1, axis=0),
                                        np.repeat(0, num_rows - 1, axis=0),
                                    )
                                )
                        else:
                            # Linearly interpolate other columns
                            interpolated_rows[col] = np.linspace(
                                start_row[col], end_row[col], num_rows
                            )

                    # Append the interpolated rows to the result DataFrame
                    result = pd.concat([result, interpolated_rows], ignore_index=True)

            # Append the last row of the original DataFrame to the result
            result = pd.concat(
                [result, pd.DataFrame(df.iloc[-1]).transpose()],
                ignore_index=True,
            )

            # Update the original dataframe with the interpolated result
            self.dataframe = result

    def noise_filter(self, column: str) -> None:
        """
        Applies a Butterworth low-pass filter to remove noise from the specified column of the dataframe.

        The filtered data replaces the original data in the specified column.

        Parameters:
            column (str): The name of the column to filter.

        Raises:
            ValueError: If the column data is empty.
        """
        # Extract the data from the specified column
        data = self.dataframe[column]

        # Validate that the column is not empty
        if data.size == 0:
            raise ValueError(
                "[QModelPredict noise_filter()]: `predictions` cannot be an empty array."
            )

        # Define Butterworth low-pass filter parameters
        fs = 300  # Sampling frequency
        normal_cutoff = 2 / (0.5 * fs)  # Normalize the cutoff frequency

        # Get the filter coefficients for a 2nd order low-pass Butterworth filter
        b, a = butter(2, Wn=normal_cutoff, btype="lowpass", analog=False)

        # Apply the filter to the data using filtfilt for zero-phase filtering
        filtered = filtfilt(b, a, data)

        # Update the dataframe with the filtered data
        self.dataframe[column] = filtered

        # Ensure that no values are negative by setting any negative values to 0
        self.dataframe[column] = self.dataframe[column].apply(lambda x: max(0, x))

    def save_dataframe(self, new_filepath: str = None) -> None:
        """
        Saves the DataFrame to a CSV file.

        Parameters:
            new_filepath (str, optional): The path where the CSV file will be saved.
                                          If not provided, the DataFrame will be saved to the original file path.

        Raises:
            ValueError: If the original file path is not set and no new file path is provided.
        """
        # Determine the file path to save the DataFrame
        filepath = new_filepath if new_filepath else self.data_filepath

        # Check if the file path is valid
        if not filepath:
            raise ValueError(
                "[QDataPipeline save_dataframe()]: No file path provided to save the DataFrame."
            )

        # Save the DataFrame to the specified CSV file
        self.dataframe.to_csv(filepath, index=False)

    def compute_smooth(self, column, winsize=5, polyorder=1):
        self.__dataframe__[column] = savgol_filter(
            self.__dataframe__[column], winsize, polyorder
        )

    def compute_difference(self):
        # Ensure the required columns are present
        required_columns = ["Dissipation", "Resonance_Frequency"]
        if not all(column in self.__dataframe__.columns for column in required_columns):
            raise ValueError(
                f"[QDataPipeline.compute_difference] Input CSV must contain the following columns: {required_columns}"
            )

        # Calculate the average value of 'Dissipation' and 'Resonance_Frequency' columns
        xs = self.__dataframe__["Relative_time"]
        i = next(x + 0 for x, t in enumerate(xs) if t > 0.5)
        j = next(x + 1 for x, t in enumerate(xs) if t > 2.5)
        avg_resonance_frequency = self.__dataframe__["Resonance_Frequency"][i:j].mean()
        avg_dissipation = self.__dataframe__["Dissipation"][i:j].mean()
        # Compute the ys_diss, ys_freq, and ys_diff
        self.__dataframe__["ys_diss"] = (
            (self.__dataframe__["Dissipation"] - avg_dissipation)
            * avg_resonance_frequency
            / 2
        )
        self.__dataframe__["ys_freq"] = (
            avg_resonance_frequency - self.__dataframe__["Resonance_Frequency"]
        )
        diff_factor = 1.5
        self.__dataframe__["Difference"] = (
            self.__dataframe__["ys_freq"] - diff_factor * self.__dataframe__["ys_diss"]
        )

        # Drop the intermediate columns if not needed
        self.__dataframe__["Resonance_Frequency"] = self.__dataframe__["ys_freq"]
        self.__dataframe__["Dissipation"] = self.__dataframe__["ys_diss"]
        self.__dataframe__["Cumulative"] = savgol_filter(
            self.__dataframe__["Dissipation"].values
            + self.__dataframe__["Resonance_Frequency"].values,
            25,
            1,
            1,
        )
        self.__dataframe__.drop(columns=["ys_diss", "ys_freq"], inplace=True)

    def normalize_df(self):
        self.__dataframe__ = (self.__dataframe__ - self.__dataframe__.min()) / (
            self.__dataframe__.max() - self.__dataframe__.min()
        )

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def fill_nan(
        self,
    ):
        self.__dataframe__.replace([np.inf, -np.inf], 0)
        self.__dataframe__.replace(np.nan, 0)

    def trim_head(self):
        data = self.__dataframe__["Dissipation"]
        factor = int(len(data) * 0.03)
        data = data.values[factor:]
        savgol_filter(data, factor, 1)

        # The start of the critical region for initial fill.
        start = 0
        # Slopes to determine the start of this region.
        start_slopes = []

        # A buffer for the first 1% of the input data set.
        # Ensures any sensor interference does not affect the slope
        # calculation.
        buffer = int(len(data) * 0.01)
        start_slopes = np.where(
            np.arange(start + 1, len(data)) < buffer,
            0,
            (data[start + 1 :] - data[start]) / np.arange(start + 1, len(data) - start),
        )

        # Compute where there is a significant positive change in the start slopes.
        # This index gets returned as the start index of the critical region.
        start_slopes = self.normalize_data(start_slopes)
        start_tmp = 1

        for i in range(1, len(start_slopes)):
            if start_slopes[i] < start_slopes[start] + 0.1:
                start_tmp = i
        self.__dataframe__ = self.__dataframe__.loc[start + start_tmp :]
        # self.__dataframe__ = self.__dataframe__.reset_index(drop=True)
        self.head_trim = start + start_tmp - factor

    def compute_gradient(self, column):
        # Compute the gradient of the column
        gradient = self.__dataframe__[column].diff()
        # Create the new column name
        gradient_column_name = column + "_gradient"

        # Add the gradient as a new column to the dataframe
        self.__dataframe__[gradient_column_name] = gradient

    def add_class(self, poi_filepath=None):
        if poi_filepath is None:
            raise ValueError(
                "[QDataPipeline.add_class] Adding class for training requires POI reference file, found None."
            )
        if os.path.exists(poi_filepath):
            # Read POI file to get the indices
            with open(poi_filepath, "r") as poi_csv:
                reader = csv.reader(poi_csv)
                indices = [int(row[0]) - 2 for row in reader]
            if self.__multi_class__:
                self.__dataframe__[M_TARGET] = 0
                for poi, idx in enumerate(indices):
                    self.__dataframe__.loc[idx, M_TARGET] = poi + 1
            else:
                self.__dataframe__["Class_1"] = 0
                self.__dataframe__["Class_2"] = 0
                self.__dataframe__["Class_3"] = 0
                self.__dataframe__["Class_4"] = 0
                self.__dataframe__["Class_5"] = 0
                self.__dataframe__["Class_6"] = 0
                for poi, idx in enumerate(indices):
                    self.__dataframe__.loc[idx, "Class_" + str(poi + 1)] = 1

        else:
            raise ValueError(
                f"[QDataPipeline.add_class] POI reference file does not exist at {poi_filepath}."
            )
