import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

DIRECTORY = "content/training_data_with_points"
RANDOM_STATE = 42
N_CLUSTERS = 4


def load_content(data_dir, size):
    content = []

    for root, dirs, files in os.walk(data_dir):
        for file in tqdm(files, desc="<<Loading Data>>"):
            if (
                file.endswith(".csv")
                and not file.endswith("_poi.csv")
                and not file.endswith("_lower.csv")
            ):
                content.append(os.path.join(root, file))

    train_content, test_content = train_test_split(
        content, test_size=size, random_state=42, shuffle=True
    )
    training_dfs = []
    for file in tqdm(train_content, desc="<<Building Training Dataset>>"):
        df = pd.read_csv(file)
        df.drop(
            columns=[
                "Date",
                "Time",
                "Ambient",
                "Temperature",
                "Peak Magnitude (RAW)",
            ],
            inplace=True,
        )
        training_dfs.append(df)
    test_dfs = []
    for file in tqdm(test_content, desc="<<Building Test Dataset>>"):
        df = pd.read_csv(file)
        df.drop(
            columns=[
                "Date",
                "Time",
                "Ambient",
                "Temperature",
                "Peak Magnitude (RAW)",
            ],
            inplace=True,
        )
        test_dfs.append(df)
    return training_dfs, test_dfs


def extract(df):
    means = df.mean().values
    stds = df.std().values
    r_time = df["Relative_time"].max()
    return (means, stds)


def extract_features(dataframes):
    features = []
    for df in dataframes:
        ft = extract(df)
        features.append(np.concatenate(ft))
    return np.array(features)


def cluster_data(features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(features)
    return labels, kmeans


def evaluate_clustering(features, labels):
    score = silhouette_score(features, labels)
    return score


def get_representative_samples(labels, dataframes, n_clusters):
    representative_samples = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) > 0:
            representative_sample_index = cluster_indices[0]
            representative_samples.append(dataframes[representative_sample_index])
    return representative_samples


def display_samples(samples, clusters):
    cmap = plt.get_cmap("tab10")  # 'tab10' is a colormap with distinct colors
    color_map = [cmap(i) for i in np.linspace(0, 1, clusters)]
    for i, sample in enumerate(samples):
        plt.figure()
        plt.title(f"Sample from Cluster {i + 1}")
        plt.plot(sample["Dissipation"], color=color_map[i], label="Dissipation")
        plt.legend()
        plt.show()


def determine_optimal_clusters(features):
    inertia = []
    silhouette_scores = []
    K = range(2, 11)  # Check for cluster numbers from 2 to 10

    for k in tqdm(K, desc="<<Determining Optimal K>>"):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(features)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features, labels))

    # Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(K, inertia, "bx-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method For Optimal k")
    plt.show()

    # Silhouette Method
    plt.figure(figsize=(10, 5))
    plt.plot(K, silhouette_scores, "bx-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method For Optimal k")
    plt.show()

    # Determine the best number of clusters
    optimal_k = K[np.argmax(silhouette_scores)]
    return optimal_k


train_dfs, test_dfs = load_content(DIRECTORY, 0.2)
features = extract_features(train_dfs)
optimal_k = determine_optimal_clusters(features)
print(f"Optimal number of clusters: {optimal_k}")

labels, kmeans = cluster_data(features, optimal_k)
representative_samples = get_representative_samples(labels, train_dfs, optimal_k)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="viridis")
plt.title("Clustering of CSV Files")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

score = evaluate_clustering(features, labels)
print(f"Silhouette Score: {score}")
display_samples(representative_samples, optimal_k)
for df in test_dfs:
    features = []
    features.append(np.array(np.concatenate(extract(df))))
    result = kmeans.predict(features)
    plt.figure()
    plt.title(f"Run Classified as Type {result[0] + 1}")
    plt.plot(
        representative_samples[result[0]]["Dissipation"],
        color="grey",
        label=f"Representative Type {result[0] + 1}",
    )
    plt.plot(df["Dissipation"], color="red", label="Classified Run")
    plt.legend()
    plt.show()
