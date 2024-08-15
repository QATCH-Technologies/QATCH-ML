import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras_preprocessing.image import img_to_array
from keras.models import Model
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import math
from QConstants import *
import joblib


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


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


def load_images(file, size):
    print("[INFO] Loading images")
    images = []
    for i, file in enumerate(tqdm(content, desc="<<Loading Images>>")):
        if i >= size:
            print("[INFO] Breaking early")
            break
        df = pd.read_csv(file)
        dissipation = df["Dissipation"]
        # Render the plot to a NumPy array
        fig, ax = plt.subplots()
        ax.plot(dissipation)
        ax.axis("off")
        image = fig2img(fig)
        images.append(image)
        plt.close()
    return images


# Preprocess images
def preprocess_images(pil_images, target_size=(224, 224)):
    print("[INFO] Preprocessing images")
    processed_images = []
    for img in tqdm(pil_images, desc="<<Preprocessing Images>>"):
        img = img.resize(target_size)
        if img.mode != "RGB":  # Check if the image is not already in RGB mode
            img = img.convert("RGB")  # Convert grayscale to RGB
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        processed_images.append(img_array)
    return np.vstack(processed_images)


# Feature extraction using pre-trained CNN
def extract_features(images, model):
    print(f"[INFO] Extracting features using model {model}")
    features = model.predict(images)
    return features.reshape((features.shape[0], -1))


# Dimensionality reduction
def reduce_dimensionality(features, n_components=50):
    print(f"[INFO] Dimensionality reduction with {n_components} components")
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features


# Clustering
def perform_clustering(features, n_clusters=5):
    print(f"[INFO] Clustering with {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(features)
    return labels, kmeans


# Visualization
def visualize_clusters(features, labels):
    print(f"[INFO] Visualizing")
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="viridis")
    plt.title("Clusters of 2D Line Plot Images")
    plt.show()


def display_cluster_samples(pil_images, labels, n_clusters=5):
    fig, axes = plt.subplots(1, n_clusters, figsize=(15, 5))
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        sample_index = cluster_indices[0]  # Select the first image in the cluster

        # Get the original PIL image
        sample_image = pil_images[sample_index]
        print(sample_image)
        # # Convert back to original scale if necessary
        # if sample_image.mode == "F":  # Check if the image is in float mode
        #     sample_image = sample_image.convert("RGB")  # Ensure it's in RGB format
        axes[i].imshow(sample_image)
        axes[i].axis("off")
        axes[i].set_title(f"Cluster {i}")
    plt.show()


def display_cluster_images(pil_images, labels, n_clusters=5):
    fig, axes = plt.subplots(n_clusters, 1, figsize=(15, n_clusters * 5))

    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        n_images = len(cluster_indices)

        # Determine grid size
        n_cols = math.ceil(math.sqrt(n_images))
        n_rows = math.ceil(n_images / n_cols)

        # Create a new figure for each cluster
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
        ax = ax.flatten()  # Flatten the axes array for easy indexing

        for j, idx in enumerate(cluster_indices):
            sample_image = pil_images[idx]

            if sample_image.mode != "RGB":  # Convert to RGB if necessary
                sample_image = sample_image.convert("RGB")

            ax[j].imshow(sample_image)
            ax[j].axis("off")
            ax[j].set_title(f"Image {j+1}")

        # Hide any unused subplots in the grid
        for j in range(n_images, len(ax)):
            ax[j].axis("off")

        plt.suptitle(f"Cluster {i}", fontsize=20)
        plt.show()


def find_optimal_clusters(features, min_k=2, max_k=10):
    print(f"[INFO] Finding optimal_k with max_k={max_k}")
    inertia = []
    silhouette_scores = []
    k_values = range(min_k, max_k + 1)

    for k in tqdm(k_values, desc="<<Optimal K>>"):

        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(features)

        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features, labels))

    # Plot the Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, inertia, "bx-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method For Optimal K")
    plt.show()

    # Plot the Silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, silhouette_scores, "bx-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis For Optimal K")
    plt.show()

    # Return the k with the highest   score
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette analysis: {optimal_k}")
    return optimal_k


# Main pipeline function
def pipeline(pil_images, min_k, max_k=10):
    # Load pre-trained VGG16 model + higher level layers
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Preprocess images
    processed_images = preprocess_images(pil_images)

    # Extract features
    features = extract_features(processed_images, model)

    # Reduce dimensionality
    reduced_features = reduce_dimensionality(features, n_components=50)
    optimal_k = find_optimal_clusters(reduced_features, min_k=min_k, max_k=max_k)

    # Perform clustering
    labels, kmeans = perform_clustering(reduced_features, n_clusters=optimal_k)

    # Visualize clusters
    visualize_clusters(reduced_features, labels)

    display_cluster_samples(pil_images, labels, n_clusters=optimal_k)
    display_cluster_images(pil_images, labels, n_clusters=optimal_k)

    return labels, kmeans


if __name__ == "__main__":
    content = load_content("content/cluster_train")
    images = load_images(content, len(content))
    labels, kmeans_model = pipeline(images, min_k=5, max_k=30)
    joblib.dump(kmeans_model, "cluster.joblib")
