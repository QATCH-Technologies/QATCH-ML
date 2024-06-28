import pandas as pd
import os
import csv
import numpy as np
import statistics as stats
from scipy.signal import savgol_filter, butter, filtfilt, argrelextrema
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

DOWNSAMPLE_AFTER = 90
DOWNSAMPLE_COUNT = 20
DOWNSAMPLE_THRESHOLD = 0.6


class QDataPipeline:
    """
    A class to handle data preprocessing for machine learning models.

    Attributes:
        __filepath__ (str): Path to the CSV file containing the data.
        __dataframe__ (DataFrame): DataFrame containing the loaded data.
    """

    def __init__(self, filepath=None):
        """
        Initializes the QDataPipeline with the given file path.

        Args:
            filepath (str, optional): Path to the input CSV file. Defaults to None.

        Raises:
            ValueError: If no file path is provided.
        """
        if filepath is not None:
            self.__filepath__ = filepath
            self.__dataframe__ = pd.read_csv(self.__filepath__)
        else:
            raise ValueError(
                f"[QDataPipeline.__init__] filepath required, found {filepath}."
            )

    def get_dataframe(self):
        """
        Returns the DataFrame containing the loaded data.

        Returns:
            DataFrame: The loaded data.
        """
        return self.__dataframe__

    def scale(self, column):
        # define standard scaler
        scaler = StandardScaler()
        df = self.__dataframe__

        # transform data
        self.__dataframe__[column] = scaler.fit_transform(
            np.array(df[column]).reshape(-1, 1)
        )

    def interpolate(self):
        data = self.__dataframe__
        timestamps = self.__dataframe__["Relative_time"].values
        time_diff = np.diff(timestamps)
        if timestamps[-1] > DOWNSAMPLE_AFTER:
            start = next(
                x + 0 for x, t in enumerate(timestamps) if t > DOWNSAMPLE_AFTER
            )
            time_diff = time_diff[start:]
            interp_idxs = np.where(time_diff > DOWNSAMPLE_THRESHOLD)
            for idx in interp_idxs:
                data[idx] = np.NaN
                data.index = data.index + 1  # shifting index
                data = data.sort_index()
            # for idx in interp_idxs[::-1]:
            #     interp_times = np.linspace(
            #         timestamps[idx], timestamps[idx + 1], DOWNSAMPLE_COUNT
            #     )
            #     for i, t in enumerate(interp_times):
            #         interp_result = np.interp(
            #             x=timestamps[t], xp=timestamps[start:], fp=data[start:]
            #         )
            #         np.insert(data)
            print(len(data), len(self.__dataframe__))
            data.interpolate(axis=1, inplace=True)
            plt.figure(figsize=(10, 10))
            plt.plot(self.__dataframe__["Dissipation"], color="b")
            plt.plot(data["Dissipation"], color="r")

            plt.show()

    def noise_filter(self, column):
        # Validate `predictions`
        data = self.__dataframe__[column]
        if data.size == 0:
            raise ValueError(
                "[QModelPredict noise_filter()]: `predictions` cannot be an empty array."
            )

        # Butterworth low-pass filter parameters
        fs = 300
        normal_cutoff = 2 / (0.5 * fs)
        # Get the filter coefficients
        b, a = butter(2, Wn=[normal_cutoff], btype="lowpass", analog=False)

        # Apply the filter using filtfilt
        filtered_predictions = filtfilt(b, a, data)

        self.__dataframe__[column] = filtered_predictions
        self.__dataframe__[column] = abs(
            self.__dataframe__[column]
            - savgol_filter(self.__dataframe__[column], 25, 1)
        )
        # xs = self.__dataframe__["Relative_time"]
        # i = next(x + 0 for x, t in enumerate(xs) if t > 0.5)
        # j = next(x + 1 for x, t in enumerate(xs) if t > 2.5)
        # baseline = (
        #     self.__dataframe__[column][i:j].max()
        #     - self.__dataframe__[column][i:j].min()
        # )
        # self.__dataframe__[column] = np.where(
        #     self.__dataframe__[column] < 2 * baseline, 0, self.__dataframe__[column]
        # )
        # extrema_indices = argrelextrema(self.__dataframe__[column].values, np.greater)
        # extrema_data = [data.values[i] for i in extrema_indices]
        # interpolation_function = interp1d(
        #     extrema_indices[0], extrema_data[0], fill_value="extrapolate"
        # )
        # new_x = np.arange(len(data))
        # self.__dataframe__["Extrema"] = abs(interpolation_function(new_x))
        # plt.figure()
        # plt.plot(self.__dataframe__[column], c="black")
        # plt.plot(self.__dataframe__["Extrema"], c="y")
        # plt.axhline(np.std(self.__dataframe__[column]), c="g")
        # plt.axhline(2 * baseline, c="r")
        # plt.show()
        # np.std(self.__dataframe__[column])

    def save_dataframe(self, new_filepath=None):
        """
        Saves the DataFrame to a CSV file.

        Args:
            new_filepath (str, optional): Path to save the CSV file. Defaults to None.

        If no new file path is provided, it saves to the original file path.
        """
        if new_filepath:
            self.__dataframe__.to_csv(new_filepath, index=False)
        else:
            self.__dataframe__.to_csv(self.__filepath__, index=False)

    def compute_smooth(self, column, winsize=5, polyorder=1):
        self.__dataframe__[column] = savgol_filter(
            self.__dataframe__[column], winsize, polyorder
        )

    def compute_difference(self):
        """
        Computes the 'Difference' column based on 'Dissipation' and 'Resonance_Frequency'.

        Raises:
            ValueError: If the required columns are not present in the DataFrame.
        """
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
        # dynamic_factor = self.__dataframe__["ys_freq"] / self.__dataframe__["ys_diss"]
        # mask = np.where(
        #     dynamic_factor > 1.1,
        # )
        # diff_factor = np.array(dynamic_factor)[mask].mean()
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

    def fill_nan(self):
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
        return start + start_tmp - factor

    def compute_gradient(self, column):
        # Compute the gradient of the column
        gradient = self.__dataframe__[column].diff()
        # Create the new column name
        gradient_column_name = column + "_gradient"

        # Add the gradient as a new column to the dataframe
        self.__dataframe__[gradient_column_name] = gradient

    def add_class(self, poi_filepath=None):
        """
        Adds a 'Class' column based on points of interest (POI) from a reference file.

        Args:
            poi_filepath (str, optional): Path to the POI CSV file. Defaults to None.

        Raises:
            ValueError: If no POI file path is provided or the file does not exist.
        """
        if poi_filepath is None:
            raise ValueError(
                "[QDataPipeline.add_class] Adding class for training requires POI reference file, found None."
            )
        if os.path.exists(poi_filepath):
            # Read POI file to get the indices
            with open(poi_filepath, "r") as poi_csv:
                reader = csv.reader(poi_csv)
                indices = [int(row[0]) - 2 for row in reader]

            # self.__dataframe__["Class"] = 0
            # for poi, idx in enumerate(indices):
            #     self.__dataframe__.loc[idx, "Class"] = poi + 1
            self.__dataframe__["Class"] = 0
            self.__dataframe__.loc[indices, "Class"] = 1
        else:
            raise ValueError(
                f"[QDataPipeline.add_class] POI reference file does not exist at {poi_filepath}."
            )
