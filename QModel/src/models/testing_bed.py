import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
import pandas as pd

# Load data
df = pd.read_csv("content/training_data/01053/W10_QV1881_C9_3rd.csv")
dissipation = df["Dissipation"].values
time = df["Relative_time"].values
dissipation = savgol_filter(dissipation, window_length=30, polyorder=3)

# Detect peaks (local maxima) and troughs (local minima)
peaks, _ = find_peaks(dissipation)
troughs, _ = find_peaks(-dissipation)

# Combine peaks and troughs, and sort them by their indices
envelope_indices = np.sort(np.concatenate([peaks, troughs]))
envelope_values = dissipation[envelope_indices]

# Linear interpolation between peaks and troughs
interpolator = interp1d(envelope_indices, envelope_values,
                        kind='linear', fill_value="extrapolate")
interpolated_envelope = interpolator(np.arange(len(dissipation)))
ie_peaks, _ = find_peaks(interpolated_envelope)
ie_troughs, _ = find_peaks(-interpolated_envelope)

# Function to identify clusters of consecutive peak-trough pairs


def identify_clusters_of_pairs(peaks, troughs, distance_threshold=50):
    clusters = []
    current_cluster = []

    # Pair peaks with their closest troughs and group consecutive pairs
    i = 0
    while i < len(peaks):
        peak = peaks[i]
        # Find the nearest trough
        closest_trough = min(troughs, key=lambda trough: abs(peak - trough))

        # Check if the peak and trough are close enough to the last pair in the current cluster
        if current_cluster and (abs(peak - current_cluster[-1][0]) < distance_threshold and abs(closest_trough - current_cluster[-1][1]) < distance_threshold):
            current_cluster.append((peak, closest_trough))
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [(peak, closest_trough)]
        i += 1

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


# Identify clusters of consecutive peak-trough pairs
clusters = identify_clusters_of_pairs(ie_peaks, ie_troughs)

# Plot the results
plt.plot(interpolated_envelope, label='Rough Envelope (Linear Interpolation)',
         color='orange', linestyle='--')
plt.scatter(
    ie_peaks, interpolated_envelope[ie_peaks], color='green', label='Peaks')
plt.scatter(
    ie_troughs, interpolated_envelope[ie_troughs], color='red', label='Troughs')

# Highlight clusters of pairs
for cluster in clusters:
    for peak, trough in cluster:
        plt.plot([peak, trough], [interpolated_envelope[peak],
                 interpolated_envelope[trough]], color='purple', linewidth=2)

plt.xlabel('Index')
plt.ylabel('Dissipation')
plt.title(
    'Rough Envelope of Dissipation Curve with Identified Clusters of Peak-Trough Pairs')
plt.legend()
plt.show()
