import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from multi_model import read_data, load_data_from_directory
from scipy.signal import find_peaks

DATA = "content/training_data_with_points/W0802_F5_DI6s_good_3rd.csv"
POI = "content/training_data_with_points/W0802_F5_DI6s_good_3rd_poi.csv"

""" Path to the training data directory. """
CONTENT_DIRECTORY = "content/training_data_with_points"
HEAD_OFFSET = 50
TAIL_OFFSET = 50


def trim_data(data, sensitivity=0.1):
    # Calculate the gradient of the curve
    gradient = np.gradient(data)
    # Find peaks in the absolute gradient to detect edges
    peaks, _ = find_peaks(
        np.abs(gradient), height=sensitivity * np.max(np.abs(gradient))
    )

    # Check if any peaks are detected
    if len(peaks) > 1:
        start_index = peaks[0]
        end_index = peaks[-1]
    else:
        # If no peaks are detected, return the original data
        return data, (0, len(data) - 1)

    # Return the trimmed data and indices
    if start_index - HEAD_OFFSET > 0:
        start_index = start_index - HEAD_OFFSET
    if end_index + TAIL_OFFSET < len(data):
        end_index = end_index + TAIL_OFFSET

    print(start_index, end_index)
    return (start_index, end_index)


def systematic_downsampling(data, interval):
    downsampled_data = data[::interval]
    mapping = {i: i * interval for i in range(len(downsampled_data))}

    return downsampled_data, mapping


def find_inflection_points(data, window_size=10, threshold=0.1):
    down_data, down_map = systematic_downsampling(data, int(len(data) * 0.01))
    plt.figure(figsize=(10, 5))
    plt.plot(down_data, label="Original Data", linestyle="-", marker="o")

    # Adding titles and labels
    plt.title("Original and Smoothed Data")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate first derivative
    first_derivative = np.gradient(down_data)

    # Calculate second derivative
    second_derivative = np.gradient(first_derivative)

    # Find peaks in the second derivative
    peaks = []
    while len(peaks) < 3:
        peaks = (
            np.where(
                (second_derivative[:-1] > threshold)
                & (second_derivative[1:] < -threshold)
            )[0]
            + 1
        )
        threshold = threshold / 2

    # Divide the data into three sections
    section_length = len(data) // 3

    # Determine inflection points from each section
    inflection_regions = []
    for i in range(3):
        start_index = i * section_length
        end_index = min((i + 1) * section_length, len(data))
        section_peaks = [peak for peak in peaks if start_index <= peak < end_index]
        if section_peaks:
            peak = np.random.choice(section_peaks)
            left_index = down_map[max(0, peak - window_size // 2)]
            right_index = down_map[min(len(data) - 1, peak + window_size // 2)]
            inflection_regions.append((left_index, right_index))

    return inflection_regions


if __name__ == "__main__":
    correct = []
    incorrect = []
    content = load_data_from_directory(CONTENT_DIRECTORY)
    for data in content:
        diss_data = data["RAW"]["Dissipation"]
        pois = data["POI"][0]
        start_idx, end_idx = trim_data(diss_data)
        for poi in pois:
            poi = poi - start_idx
        transformed_data = np.log(diss_data + 1)
        inflection_regions = find_inflection_points(transformed_data[start_idx:end_idx])
        correct_count = 0
        for poi in pois:
            for region in inflection_regions:
                if poi in range(region[0], region[1]):
                    correct_count += 1

        if correct_count == 6:
            correct.append(data)
        else:
            incorrect.append(data)
        print("Inflection point regions:")
        for region in inflection_regions:
            print(region)

        time = np.arange(len(diss_data))
        plt.plot(diss_data, label="Data")

        plt.axvline(x=time[start_idx], color="r", linestyle="--", label="Head Trim")
        plt.axvline(x=time[end_idx - 1], color="g", linestyle="--", label="Tail Trim")
        for left, right in inflection_regions:
            plt.axvspan(
                time[left + start_idx],
                time[right + start_idx],
                color="orange",
                alpha=0.3,
            )

        # Scatter plot points of interest
        plt.scatter(
            pois,
            diss_data[pois],
            color="red",
            label="POIs",
            marker="x",
        )
        plt.xlabel("Time")
        plt.ylabel("Dissipation")
        plt.title("Dissipation Over Time with Inflection Point Regions")
        plt.legend()
        plt.grid(True)
        plt.show()
