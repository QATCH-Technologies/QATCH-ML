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
import shutil
from QDataPipline import QDataPipeline
TRAIN_DIR = "content/dropbox_dump"
ALL_DIR = 'content/dropbox_dump'
RANDOM_STATE = 42
D_SAMPLE_SIZE = 256

def load_content(data_dir):
    content = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if (
                file.endswith(".csv")
                and not file.endswith("_poi.csv")
                and not file.endswith("_lower.csv")
            ):
                content.append(os.path.join(root, file))
    dfs_S = []
    dfs_L = []
    directories_S = []
    directories_L = []
    for file in tqdm(content, desc="<<Building Dataset>>"):
        qdp = QDataPipeline(filepath=file, multi_class=True)
        delta = qdp.find_time_delta()
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
        if delta == -1:
            dfs_S.append(df)
            directories_S.append(os.path.dirname(file))
        else:
            dfs_L.append(df)
            directories_L.append(os.path.dirname(file))
    return (dfs_S, directories_S), (dfs_L, directories_L)

def downsample_array(array, num_points=256):
    if len(array) <= num_points:
        return array
    
    indices = np.linspace(0, len(array) - 1, num_points, dtype=int)
    downsampled_array = array[indices]
    return downsampled_array

def extract(df):
    dissipation = df['Dissipation'].values
    dissipation_downsampled = downsample_array(dissipation, D_SAMPLE_SIZE)
    return dissipation_downsampled


def extract_features(dataframes):
    features = []
    for df in dataframes:
        ft = extract(df)
        features.append(ft)
    print(features)
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


train_dfs_S, train_dfs_L = load_content(TRAIN_DIR)
all_dfs_S, all_dfs_L = load_content(ALL_DIR)
features_S = extract_features(train_dfs_S[0])
features_L = extract_features(train_dfs_L[0])
optimal_k_S = determine_optimal_clusters(features_S)
optimal_k_L = determine_optimal_clusters(features_L)
print(f"Optimal number of clusters short: {optimal_k_S}")
print(f"Optimal number of clusters long: {optimal_k_L}")
labels_S, kmeans_S = cluster_data(features_S, optimal_k_S)
labels_L, kmeans_L = cluster_data(features_L, optimal_k_L)
representative_samples_S = get_representative_samples(labels_S, train_dfs_S[0], optimal_k_S)
representative_samples_L = get_representative_samples(labels_L, train_dfs_L[0], optimal_k_L)
pca = PCA(n_components=2)
reduced_features_S = pca.fit_transform(features_S)
reduced_features_L = pca.fit_transform(features_L)
plt.scatter(reduced_features_S[:, 0], reduced_features_S[:, 1], c=labels_S, cmap="viridis")
plt.title("Clustering of Short CSV Files")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
plt.scatter(reduced_features_L[:, 0], reduced_features_L[:, 1], c=labels_L, cmap="viridis")
plt.title("Clustering of Long CSV Files")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
score_S = evaluate_clustering(features_S, labels_S)
score_L = evaluate_clustering(features_L, labels_L)
print(f"Silhouette Score (Short): {score_S}")
print(f"Silhouette Score (Long): {score_L}")
display_samples(representative_samples_S, optimal_k_S)
display_samples(representative_samples_L, optimal_k_L)
for (df, dir) in zip(all_dfs_S[0], all_dfs_S[1]):
    features = []
    features.append(extract(df))
    result = kmeans_S.predict(features)
    target_folder_path = os.path.join('content', f"type_{result[0]}_S")
    print(f"[INFO] Copying file {dir} to {target_folder_path}")
    shutil.copytree(dir,  target_folder_path, dirs_exist_ok=True)
for (df, dir) in zip(all_dfs_L[0], all_dfs_L[1]):
    features = []
    features.append(extract(df))
    result = kmeans_L.predict(features)
    target_folder_path = os.path.join('content', f"type_{result[0]}_L")
    print(f"[INFO] Copying file {dir} to {target_folder_path}")
    shutil.copytree(dir,  target_folder_path, dirs_exist_ok=True)
