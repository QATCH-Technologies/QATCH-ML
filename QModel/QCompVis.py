from PIL import Image
import os
import matplotlib.pyplot as plt
from QConstants import *
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_content(data_dir):
    print(f"[INFO] Loading content from {data_dir}")
    content = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if (
                file.endswith(".csv")
                and not file.endswith("_poi.csv")
                and not file.endswith("_lower.csv")
            ):
                content.append(os.path.join(root, file))
    return content

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

def load_images(file):
    images = []
    for file in tqdm(content, desc='<<Building Images>>'):
        df = pd.read_csv(file)
        dissipation = df['Dissipation']
        # Render the plot to a NumPy array
        fig, ax = plt.subplots()
        ax.plot(dissipation)
        ax.axis("off")
        canvas = FigureCanvas(fig)
        canvas.draw()
        # Convert the canvas to a NumPy array
        image_array = np.frombuffer(canvas.renderer.buffer_rgba(), dtype=np.uint8)

        image = Image.fromarray(image_array).convert("L")
        image = np.array(image).flatten()
        images.append(image)
        plt.close()
    return images

def plot_cluster_images(images, labels, num_clusters):
    fig, axes = plt.subplots(num_clusters, 1, figsize=(10, 10))
    for cluster in range(num_clusters):
        cluster_images = [images[i] for i in range(len(images)) if labels[i] == cluster]
        axes[cluster].set_title(f'Cluster {cluster}')
        for i, img in enumerate(cluster_images[:10]):  # Show up to 10 images per cluster
            ax = plt.subplot(num_clusters, 10, cluster * 10 + i + 1)
            ax.axis('off')
    plt.show()

if __name__ == "__main__": 
    content = load_content(DROPBOX_DUMP_PATH)
    images = load_images(content)
    scaler = StandardScaler()
    images_scaled = scaler.fit_transform(images)
    optimal_k = 2
    # optimal_k = determine_optimal_clusters(images_scaled)
    print(f'Optimal clusters: {optimal_k}')
    labels, kmeans = cluster_data(images_scaled, optimal_k)
    plot_cluster_images(images, labels, optimal_k)
    representative_samples = get_representative_samples(labels, images_scaled[0], optimal_k)
    pca = PCA(n_components=2)
    reduced_features_S = pca.fit_transform(images_scaled)
    plt.scatter(reduced_features_S[:, 0], reduced_features_S[:, 1], c=labels, cmap="viridis")
    plt.title("Clustering Images")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()
    score = evaluate_clustering(images_scaled, labels)
    print(f"Silhouette Score: {score}")