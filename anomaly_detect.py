import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d


from simple_model import read_data

HEAD_OFFSET = 1600
TAIL_OFFSET = 250

data = read_data(
    "content/training_data_with_points/W0802_F5_DI6s_good_3rd.csv",
    "content/training_data_with_points/W0802_F5_DI6s_good_3rd_poi.csv",
)
dissipation_raw = data["Dissipation"].values
dissipation = data["Dissipation"].values[
    HEAD_OFFSET : len(data["Dissipation"]) - TAIL_OFFSET
]
pois = data["POIs"].values
time = np.arange(len(dissipation))
# Step 1: Preprocessing
# No preprocessing required in this example

# Step 2: Smoothing
smoothed_data = savgol_filter(dissipation, window_length=5, polyorder=3)
# Step 3: Trend Removal
trend = np.polyfit(time, smoothed_data, 3)  # Fit a 3rd degree polynomial
detrended_data = smoothed_data - np.polyval(trend, time)
# Reshape data for LOF (required for sklearn)
lof_data = dissipation.reshape(-1, 1)

# Fit the LOF model
lof = LocalOutlierFactor(contamination=0.06)
anomalies = lof.fit_predict(lof_data)
print(anomalies)
# Visualization
plt.figure(figsize=(10, 6))
plt.plot(time, dissipation, label="Original Data")
plt.plot(time, smoothed_data, label="Smoothed Data")
plt.plot(time, detrended_data, label="Detrended Data")

# Highlight anomalies
plt.scatter(
    time[anomalies == -1],
    detrended_data[anomalies == -1],
    color="red",
    label="Anomalies",
)

plt.xlabel("Time")
plt.ylabel("Dissipation")
plt.title("Original Dissipation Data with Smoothed Data and Anomalies Detected by LOF")
plt.legend()
plt.grid(True)
plt.show()


idxs = []
for pt in time:
    if anomalies[pt] == -1:
        idxs.append(pt)
    else:
        idxs.append(0)

data = np.asarray(idxs)
data = np.reshape(data, (-1, 1))
kmeans = KMeans(n_clusters=7)
kmeans.fit(data)
plt.scatter(data, time, c=kmeans.labels_)
plt.show()

clusters = {}
for i in range(len(kmeans.labels_)):
    if kmeans.labels_[i] > 0:
        if clusters.get(kmeans.labels_[i]) is None:
            clusters[kmeans.labels_[i]] = [i]
        else:
            clusters[kmeans.labels_[i]].append(i)
print(pois)

leftmost = []
for k, v in clusters.items():
    leftmost.append(HEAD_OFFSET + v[0])
pois_idxs = np.argwhere(pois == 1).flatten()
# Interpolate dissipation curve to match predicted and actual data points
f_dissipation = interp1d(np.arange(len(dissipation_raw)), dissipation_raw)
interpolated_dissipation_predicted = f_dissipation(leftmost)
interpolated_dissipation_actual = f_dissipation(pois_idxs)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(
    np.arange(len(dissipation_raw)),
    dissipation_raw,
    color="green",
    label="Dissipation Curve",
)
plt.scatter(
    leftmost, interpolated_dissipation_predicted, color="red", label="Predicted"
)
plt.scatter(pois_idxs, interpolated_dissipation_actual, color="blue", label="Actual")
plt.xlabel("Time")
plt.ylabel("Dissipation")
plt.title("Predicted and Actual Data over Dissipation Curve")
plt.legend()
plt.grid(True)
plt.show()
