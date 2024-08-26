import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras_preprocessing.image import img_to_array
from keras.models import Model
from PIL import Image
from tqdm import tqdm
import os
import joblib
import math
import io
from QDataPipeline import QDataPipeline
import pandas as pd
from sklearn.decomposition import PCA


class QClusterer:
    """A class to perform image clustering using a VGG16 model and KMeans.

    Attributes:
        model (Model): The VGG16 model for feature extraction.
        kmeans (KMeans): The KMeans clustering model.
    """

    def __init__(self, model_path: str = None) -> None:
        if model_path:
            print(
                f"[INFO] Operating in prediction mode using KMeans model from {model_path}"
            )
            self.kmeans = joblib.load(model_path)
        else:
            print(f"[INFO] Operating in training mode.")
            self.kmeans = None

    def load_content(self, data_directory: str = None) -> list:
        """Loads content from a directory.

        Args:
            data_dir (str): Path to the directory containing image files.

        Returns:
            list: List of file paths.
        """
        print(f"[STATUS] Loading content from {data_directory}")
        content = []
        for root, _, files in os.walk(data_directory):
            for file in files:
                if (
                    file.endswith(".csv")
                    and not file.endswith("_poi.csv")
                    and not file.endswith("_lower.csv")
                ):
                    content.append(os.path.join(root, file))
        return content

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def extract_feature(self, file):

        poi_file = file.replace(".csv", "_poi.csv")
        pois = pd.read_csv(poi_file, header=None).values
        df = pd.read_csv(file)
        poi_lst = []
        t = df["Relative_time"].values.flatten()
        for p in pois:
            poi_lst.append(p[0])
        t = self.normalize(t)
        rel_time = t[poi_lst]
        return rel_time

    def find_optimal_clusters(
        self, features: np.ndarray = None, min_k: int = 2, max_k: int = 10
    ) -> int:
        """Finds the optimal number of clusters using silhouette score.

        Args:
            features (np.ndarray): Array of extracted features.
            min_k (int, optional): Minimum number of clusters to test. Defaults to 2.
            max_k (int, optional): Maximum number of clusters to test. Defaults to 10.

        Returns:
            int: Optimal number of clusters.
        """
        print(f"[INFO] Finding optimal_k with max_k={max_k}")
        silhouette_scores = []
        k_values = range(min_k, max_k + 1)

        for k in tqdm(k_values, desc="<<Optimal K>>"):
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(features)
            silhouette_scores.append(silhouette_score(features, labels))

        plt.figure(figsize=(10, 5))
        plt.plot(k_values, silhouette_scores, "bx-")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis For Optimal K")
        plt.show()
        optimal_k = k_values[np.argmax(silhouette_scores)]
        print(
            f"[INFO] Optimal number of clusters based on silhouette analysis: {optimal_k}"
        )
        return optimal_k

    def perform_clustering(
        self, features: np.ndarray = None, n_clusters: int = 2
    ) -> np.ndarray:
        """Performs KMeans clustering.

        Args:
            features (np.ndarray): Array of extracted features.
            n_clusters (int): Number of clusters.

        Returns:
            np.ndarray: Cluster labels.
        """
        print(f"[INFO] Clustering with {n_clusters} clusters")
        self.kmeans = KMeans(n_clusters=n_clusters)
        labels = self.kmeans.fit_predict(features)
        return labels

    def visualize_clusters(
        self, features: np.ndarray = None, labels: np.ndarray = None
    ) -> None:
        """Visualizes clusters using a scatter plot.

        Args:
            features (np.ndarray): Array of extracted features.
            labels (np.ndarray): Cluster labels.
        """
        print(f"[STATUS] Visualizing")
        pca = PCA(n_components=2)
        transform = pca.fit_transform(features)
        plt.scatter(transform[:, 0], transform[:, 1], c=labels, cmap="viridis")
        plt.title(f"Clusters {max(labels)} POIs")
        plt.show()

    def train(
        self,
        data_dir: str = None,
        min_k: int = 2,
        max_k: int = 10,
        plotting: bool = False,
    ) -> np.ndarray:
        content = self.load_content(data_dir)
        features = []
        for file in tqdm(content, desc="<<Extracting Features>>"):
            f = self.extract_feature(file)
            features.append(f)
        # print(features)
        optimal_k = self.find_optimal_clusters(features, min_k=3, max_k=30)
        labels = self.perform_clustering(features, n_clusters=optimal_k)
        self.visualize_clusters(features, labels)

    def predict_label(self, csv_path: str = None) -> int:
        pass

    def save_model(self, model_path: str = None) -> None:
        """Saves the trained KMeans model to a file.

        Args:
            model_path (str): Path where the KMeans model should be saved.
        """
        if self.kmeans is None:
            raise ValueError("Model is not trained. Please train the model first.")
        print(f"[STATUS] Saving model as {model_path}")
        joblib.dump(self.kmeans, model_path)


# Example usage
if __name__ == "__main__":
    # Example Usage
    qcr = QClusterer()
    labels = qcr.train("content/training_data/test_clusters", min_k=3, max_k=10)
    # qcr.save_model("QModel/SavedModels/cluster.joblib")

    # For prediction
    # qcp = QClusterer("QModel/SavedModels/cluster.joblib")
    # label = qcp.predict_label(
    #     "content/dropbox_dump/01154/MM231106W10_Y60P_PROBLEM_D10_3rd.csv"
    # )
    # print(f"The image belongs to cluster {label}")
