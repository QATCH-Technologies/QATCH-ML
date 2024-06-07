"""
File: dynamic_region_analysis.py
Author: Paul MacNichol
Version: 1.3
Last Updated: 2024-06-07
License: N/A
Contact: paulmacnichol@gmail.com

Description:
This script implements a data processing pipeline for analyzing dissipation data. It performs several preprocessing steps, including downsampling, trimming, and identifying inflection regions, to extract points of interest (POIs) from the data.

Dependencies:
- numpy (version x.y.z): NumPy is used for numerical operations, such as array manipulation and mathematical computations.
- matplotlib (version x.y.z): Matplotlib is used for data visualization, specifically for generating plots of the dissipation data and identified inflection regions.
- pandas (version x.y.z): Pandas is used for handling tabular data structures, particularly for loading and manipulating datasets stored in CSV format.

Functions:
- load_data_from_directory(directory_path): Loads CSV files from the specified directory path and returns a list of data dictionaries.
- downsample(original_data, interval): Downsamples the original data by a specified interval.
- smooth(data, window_size=1): Smooths the input data using a moving average filter.
- differentiate(data): Computes the derivative of the input data using finite differences.
- determine_threshold(derivative_data, threshold_factor=0.1): Determines a threshold value based on the maximum value of the derivative data.
- find_indices_above_threshold(data, threshold): Finds the indices where the input data exceeds a specified threshold.
- trim_data(data, data_map): Trims the head and tail of the input data based on the dissipation curve.
- find_region(data, data_map, head_trim, tail_trim): Identifies and returns regions of interest within the trimmed data based on gradient analysis.

Usage:
To use this script, simply execute it in a Python environment. Ensure that the necessary dependencies are installed beforehand.

Example:
python dynamic_region_analysis.py

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import load_data_from_directory, info

__author__ = "Paul MacNichol"
__version__ = "1.3"
__last_updated__ = "2024-06-07"
__license__ = "N/A"
__contact__ = "paulmacnichol@gmail.com"

""" Path to the training data directory. """
CONTENT_DIRECTORY = "content/training_data_v2"
""" Head offset for trim curve. """
HEAD_OFFSET = 50
""" Tail offset for trim curve. """
TAIL_OFFSET = 50
""" Percentage of original data to downsample."""
DOWNSAMPLING_FACTOR = 0.01
""" The amount percentage padding for inflection regions. """
PADDING_FACTOR = 0.05
""" Plotting flag (set to True to display verbose plots)"""
PLOTTING = False


def smooth(data, window_size=1):
    """
    Smooths the input data using a simple moving average.

    Args:
        data (array-like): The input data to be smoothed. This should be a one-dimensional array or list of numerical values.
        window_size (int, optional): The size of the smoothing window. This determines the number of points over which to average
                                     the data. A larger window size results in a smoother data. Defaults to 1, which means
                                     no smoothing.

    Returns:
        numpy.ndarray: The smoothed data. This is a one-dimensional array of the same length as the input data, where each point
                       is the average of the points in the input data within the window centered at that point.
    """
    # Validate input types
    if not isinstance(data, (list, np.ndarray, pd.Series)):
        raise TypeError(f"data must be a list or a numpy array was {type(data)}")
    if not isinstance(window_size, int):
        raise TypeError("window_size must be an integer")

    # Ensure window_size is positive
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    # Create a smoothing window of the specified size.
    # Each element in the window has a value of 1/window_size, so the sum of the window elements is 1.
    # This ensures the average of the windowed data is calculated correctly.
    smoothing_window = np.ones(window_size) / window_size

    # Convolve the input data with the smoothing window.
    # The 'same' mode ensures the output data has the same length as the input data.
    # Convolution effectively slides the smoothing window across the data, averaging the points within the window.
    smoothed_data = np.convolve(data, smoothing_window, mode="same")

    # Return the smoothed data as a NumPy array.
    return smoothed_data


def differentiate(data):
    """
    Computes the numerical derivative of the input data using finite differences.

    Args:
        data (array-like): The input data to be differentiated. This should be a one-dimensional array or list of numerical values.

    Returns:
        numpy.ndarray: The derivative of the input data. This is a one-dimensional array of the same length as the input data,
                       where each point represents the rate of change of the data at that point.
    """
    # Validate input type
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("data must be a list or a numpy array")

    # Compute the numerical derivative of the data using finite differences.
    # np.gradient calculates the gradient of the data, providing an approximation of the derivative at each point.
    derivative_data = np.gradient(data)

    # Return the derivative data as a NumPy array.
    return derivative_data


def determine_threshold(derivative_data, threshold_factor=0.1):
    """
    Determines a threshold value based on the maximum value of the derivative data.

    Args:
        derivative_data (array-like): The input derivative data from which to determine the threshold.
                                      This should be a one-dimensional array or list of numerical values.
        threshold_factor (float, optional): A factor to multiply with the maximum value of the derivative data
                                            to compute the threshold. Defaults to 0.1.

    Returns:
        float: The computed threshold value. This is a single numerical value obtained by multiplying the maximum
               value of the derivative data by the threshold factor.
    """
    # Validate input types
    if not isinstance(derivative_data, (list, np.ndarray)):
        raise TypeError("derivative_data must be a list or a numpy array")
    if not isinstance(threshold_factor, (int, float)):
        raise TypeError("threshold_factor must be a numerical value")

    # Ensure threshold_factor is non-negative
    if threshold_factor < 0:
        raise ValueError("threshold_factor must be non-negative")

    # Determine the threshold by multiplying the threshold_factor with the maximum value of the derivative data.
    threshold = threshold_factor * np.max(derivative_data)

    # Return the computed threshold value.
    return threshold


def find_indices_above_threshold(data, threshold):
    """
    Finds the indices of the input data where the values exceed the given threshold.

    Args:
        data (array-like): The input data to be checked against the threshold.
                           This should be a one-dimensional array or list of numerical values.
        threshold (float): The threshold value to compare each element of the data against.

    Returns:
        numpy.ndarray: An array of indices where the values in the input data exceed the threshold.
                       The returned array contains the indices of the elements in the input data that are greater than the threshold.
    """
    # Validate input types
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("data must be a list or a numpy array")
    if not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a numerical value")

    # Find indices where data values exceed the threshold.
    # np.where returns the indices of the elements that satisfy the condition (data > threshold).
    above_threshold_indices = np.where(data > threshold)[0]

    # Return the indices as a NumPy array.
    return above_threshold_indices


def trim_data(data, data_map):
    """
    Processes the input data to identify and trim regions of rapid increase.

    This function smooths the input data, computes its derivative, determines a threshold for rapid increase, and identifies
    the start and end points of regions where the derivative exceeds the threshold. Optionally, it plots the results.

    Args:
        data (array-like): The input signal data to be processed. This should be a one-dimensional array or list of numerical values.
        data_map (array-like): A mapping of the data indices to another domain, such as time or another relevant metric.
                               This should be a one-dimensional array or list of the same length as `data`.

    Returns:
        tuple: A tuple containing the start and end points (indices) of the region of rapid increase in the data.
    """
    # 1. Preprocessing: Smooth the input data to reduce noise.
    smoothed_data = smooth(data)

    # 2. Differentiation: Compute the derivative of the smoothed data.
    derivative_data = differentiate(smoothed_data)

    # 3. Thresholding: Determine a threshold value based on the derivative data.
    threshold = determine_threshold(derivative_data)

    # Identify indices where the derivative exceeds the threshold.
    rapid_increase_indices = find_indices_above_threshold(derivative_data, threshold)

    # 4. Identify Start and End Points: Find the start and end indices of the region of rapid increase.
    start_point = rapid_increase_indices[0]
    end_point = rapid_increase_indices[-1]

    # Optional plotting for visualization.
    if PLOTTING:
        t = np.arange(len(data))
        plt.figure(figsize=(10, 6))
        plt.plot(t, data, label="Original Signal")
        plt.plot(t, smoothed_data, label="Smoothed Signal")
        plt.plot(t, derivative_data, label="Derivative Signal")
        plt.axhline(threshold, color="r", linestyle="--", label="Threshold")
        plt.scatter(
            t[rapid_increase_indices],
            derivative_data[rapid_increase_indices],
            color="g",
            label="Rapid Increase",
        )
        plt.axvline(t[start_point], color="m", linestyle="--", label="Start Point")
        plt.axvline(t[end_point], color="m", linestyle="--", label="End Point")
        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.title("Detection of Rapid Increase")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print the mapped start and end points
        print("Start point:", data_map[start_point])
        print("End point:", data_map[end_point])

    # Return the start and end points as a tuple.
    return start_point, end_point


def find_region(data, data_map, head_trim, tail_trim):
    """
    Identifies and returns regions of interest within the trimmed input data based on gradient analysis.

    Args:
        data (array-like): The input data to be analyzed. This should be a one-dimensional array or list of numerical values.
        data_map (array-like): A mapping of the data indices to another domain, such as time or another relevant metric.
                               This should be a one-dimensional array or list of the same length as `data`.
        head_trim (int): The number of data points to trim from the start of the input data.
        tail_trim (int): The number of data points to trim from the end of the input data.

    Returns:
        list of tuples: A list containing three tuples, each representing a region of interest. Each tuple contains the start
                        and end points (mapped indices) of a region.
    """
    # Trim the data based on head_trim and tail_trim values
    data = data[head_trim:tail_trim]

    # Ensure there is enough data left after trimming
    if len(data) < 1:
        raise ValueError("Data set is too small!")

    # Time index array for plotting
    t = np.arange(len(data))

    # Compute the gradient of the trimmed data
    gradient = np.gradient(data)

    # Identify the start and end points of the first region
    start_first_region = 0
    end_first_region = None
    for i in range(head_trim, len(gradient) - 1):
        avg_grad = np.mean(gradient[head_trim:i])
        if avg_grad - gradient[i - 1] < 0:  # Check if gradient is decreasing
            end_first_region = i
            break

    # If no end point for the first region is found, set it to the last data point
    if end_first_region is None:
        end_first_region = len(data) - 1

    # Identify the start and end points of the second and third regions
    start_second_region = end_first_region
    end_third_region = tail_trim
    start_third_region = 0
    for i in range(len(gradient) - 1, -1, -1):
        avg_grad = np.mean(gradient[i : len(gradient) - 1])
        if avg_grad - gradient[i - 1] > 0:  # Check if gradient is increasing
            start_third_region = i - int(len(gradient) * PADDING_FACTOR)
            break

    # If no start point for the third region is found, set it to head_trim
    if start_third_region is None:
        start_third_region = head_trim

    end_second_region = start_third_region

    # Optional plotting for visualization
    if PLOTTING:
        plt.figure(figsize=(10, 6))
        plt.plot(t, data, label="Data", color="b")
        plt.plot(t, gradient, label="Gradient", color="r", linestyle="--")
        plt.axvspan(
            start_first_region,
            end_first_region,
            color="orange",
            alpha=0.3,
            label="First Region",
        )
        plt.axvspan(
            start_second_region,
            end_second_region,
            color="green",
            alpha=0.3,
            label="Second Region",
        )
        plt.axvspan(
            start_third_region,
            len(data) - 1,
            color="blue",
            alpha=0.3,
            label="Third Region",
        )
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Data and Gradient with Identified Regions")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Compute the regions' start and end points mapped to the original data
    regions = [
        (
            data_map[start_first_region] + data_map[head_trim],
            data_map[end_first_region] + data_map[head_trim],
        ),
        (
            data_map[start_second_region] + data_map[head_trim],
            data_map[end_second_region] + data_map[head_trim],
        ),
        (
            data_map[start_third_region] + data_map[head_trim],
            data_map[end_third_region],
        ),
    ]

    # Return the identified regions
    return regions


def downsample(original_data, interval):
    """
    Downsamples the original data by a given interval.

    Args:
        original_data (pandas.DataFrame): The original data to be downsampled. This should be a pandas DataFrame.
        interval (int): The downsampling interval, i.e., the number of data points to skip between samples.

    Returns:
        tuple: A tuple containing the downsampled data and a mapping dictionary. The downsampled data is a subset
               of the original data, and the mapping dictionary maps indices of the downsampled data to the
               corresponding indices in the original data.
    """
    # Downsample the original data using the specified interval
    downsampled_data = original_data.iloc[
        int(len(original_data) * DOWNSAMPLING_FACTOR) :: interval
    ]

    # Create a mapping dictionary to map indices of downsampled data to original data
    mapping = {i: i * interval for i in range(len(downsampled_data))}

    # Optionally plot the original and downsampled data
    if PLOTTING:
        plt.figure(figsize=(10, 5))
        plt.plot(
            original_data.index,
            original_data["Dissipation"],
            label="Original Data",
            linestyle="-",
            color="b",
        )
        plt.scatter(
            downsampled_data.index,
            downsampled_data["Dissipation"],
            label="Downsampled Data",
            linestyle="-",
            marker="o",
            color="r",
        )

        # Add titles and labels to the plot
        plt.title("Original and Downsampled Data")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Return the downsampled data and mapping dictionary
    return downsampled_data, mapping


if __name__ == "__main__":
    correct_regions, incorrect_regions = [], []
    correct_plots, incorrect_plots = [], []

    # Load CSV files from content directory path. (Can change to production data in the future.)
    content = load_data_from_directory(CONTENT_DIRECTORY)

    for data_set in content:
        # Downsample data set.
        downsampled_data, downsampled_data_map = downsample(
            data_set["RAW"], int(len(data_set["RAW"]) * DOWNSAMPLING_FACTOR)
        )

        # Extract dissipation points and POIs from downsampled dataset.
        downsampled_dissipation_data = downsampled_data["Dissipation"]
        poi_data = data_set["POI"][0]

        # Trim head and tail of dataset based on dissipation curve.
        head_trim, tail_trim = trim_data(
            downsampled_dissipation_data, downsampled_data_map
        )

        # Adjust each POI based on the amount trimmed from the head of the dataset.
        adj_poi_data = [poi - head_trim for poi in poi_data]

        # Determine inflection regions based on dissipation data within the trim bounds.
        try:
            inflection_regions = find_region(
                downsampled_dissipation_data, downsampled_data_map, head_trim, tail_trim
            )
        except ValueError as e:
            print(e)
            continue

        # Count correct POIs within inflection regions
        correct_poi_count = 0
        for p in range(0, len(poi_data)):
            if p < 4 and (
                poi_data[p] > inflection_regions[0][0]
                and poi_data[p] < inflection_regions[0][1]
            ):
                correct_poi_count += 1
            if (
                p == 4
                and adj_poi_data[p]
                and (
                    poi_data[p] > inflection_regions[1][0]
                    and poi_data[p] < inflection_regions[1][1]
                )
            ):
                correct_poi_count += 1
            if p == 5 and (
                poi_data[p] > inflection_regions[2][0]
                and poi_data[p] < inflection_regions[2][1]
            ):
                correct_poi_count += 1

        # Plot the data with inflection regions and actual POIs
        time = np.arange(len(data_set["RAW"]))
        plt.plot(downsampled_dissipation_data, label="Data")
        plt.axvline(
            x=time[downsampled_data_map[head_trim]],
            color="r",
            linestyle="--",
            label="Head Trim",
        )
        plt.axvline(
            x=time[downsampled_data_map[tail_trim] - 1],
            color="g",
            linestyle="--",
            label="Tail Trim",
        )
        span_colors = ["r", "b", "orange"]
        for i, pts in enumerate(inflection_regions):
            plt.axvspan(
                time[pts[0]],
                time[pts[1]],
                color=span_colors[i],
                alpha=0.3,
            )
        plt.scatter(
            poi_data,
            data_set["RAW"]["Dissipation"][poi_data],
            color="red",
            label="Actual POIs",
            marker="x",
        )
        plt.xlabel("Time")
        plt.ylabel("Dissipation")
        plt.title("Dissipation Over Time with Inflection Point Regions")
        plt.legend()
        plt.grid(True)

        # Output POI counts and accuracy
        info(f"correct poi count {correct_poi_count}")
        info(f"POI count {len(poi_data)}\n")

        # Determine if the regions were predicted correctly
        if correct_poi_count == len(poi_data):
            correct_regions.append(data_set)
            correct_plots.append(plt)
        else:
            incorrect_regions.append(data_set)
            incorrect_plots.append(plt)

    # Calculate and output accuracy
    accuracy = len(correct_regions) / (len(correct_regions) + len(incorrect_regions))
    info(f"Accuracy at predicting inflection regions: {accuracy}")

    # Show plots for incorrectly predicted regions
    for p in incorrect_plots:
        p.show()
    # info(f"Incorrect data sets were : {incorrect_regions}")
