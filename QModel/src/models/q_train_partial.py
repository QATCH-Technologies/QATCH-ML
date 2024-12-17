from collections import Counter
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from q_data_pipeline import QDataPipeline


def extract_features(image_paths, model):
    """
    Extract features from images using a pre-trained model.
    Args:
        image_paths (list): List of image file paths.
        model (Model): Pre-trained feature extraction model.

    Returns:
        np.ndarray: Extracted features for each image.
    """
    features = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=(256, 256))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Extract features
        feature = model.predict(img_array)
        features.append(feature.flatten())
    return np.array(features)


def cluster_images(features, n_clusters):
    """
    Cluster extracted features using K-Means.
    Args:
        features (np.ndarray): Extracted features from images.
        n_clusters (int): Number of clusters.

    Returns:
        KMeans: Fitted KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans


def map_clusters_to_categories(train_paths, train_labels, categories):
    """
    Map clusters to categories based on majority voting.
    Args:
        train_paths (list): Paths to training images.
        train_labels (np.ndarray): Cluster labels for training images.
        categories (list): List of category names.

    Returns:
        dict: Mapping of cluster labels to category names.
    """
    # Extract the true category from the file paths
    true_labels = []
    for path in train_paths:
        for category in categories:
            if category in path:
                true_labels.append(category)
                break

    # Count occurrences of categories in each cluster
    cluster_to_category = {}
    for cluster in np.unique(train_labels):
        cluster_indices = np.where(train_labels == cluster)[0]
        cluster_categories = [true_labels[idx] for idx in cluster_indices]
        most_common_category = Counter(cluster_categories).most_common(1)[0][0]
        cluster_to_category[cluster] = most_common_category

    return cluster_to_category


def evaluate_predictions(test_paths, test_labels, cluster_to_category, categories):
    """
    Evaluate the quality of predictions by comparing them to ground truth labels.
    Args:
        test_paths (list): Paths to testing images.
        test_labels (np.ndarray): Cluster labels for testing images.
        cluster_to_category (dict): Mapping of cluster labels to category names.
        categories (list): List of category names.

    Returns:
        None
    """
    # Extract ground truth categories
    true_labels = []
    for path in test_paths:
        for category in categories:
            if category in path:
                true_labels.append(category)
                break

    # Map cluster labels to categories
    predicted_categories = [cluster_to_category[label]
                            for label in test_labels]

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_categories, labels=categories)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=categories, yticklabels=categories)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(true_labels,
          predicted_categories, target_names=categories))
    print("Raw Counts of Correct and Incorrect Predictions:")
    for idx, category in enumerate(categories):
        correct = cm[idx, idx]
        incorrect = sum(cm[idx, :]) - correct
        print(
            f"Category '{category}': Correct = {correct}, Incorrect = {incorrect}")

    # Silhouette Score
    silhouette = silhouette_score(test_features, test_labels)
    print(f"Silhouette Score: {silhouette:.3f}")


def predict_and_map(image_paths, model, kmeans, cluster_to_category):
    """
    Predict cluster labels for images and map them to categories.
    Args:
        image_paths (list): List of image file paths.
        model (Model): Pre-trained feature extraction model.
        kmeans (KMeans): Fitted KMeans clustering model.
        cluster_to_category (dict): Mapping of cluster labels to category names.

    Returns:
        pd.DataFrame: DataFrame containing image paths, cluster labels, and mapped categories.
    """
    features = extract_features(image_paths, model)
    cluster_labels = kmeans.predict(features)
    categories = [cluster_to_category[label] for label in cluster_labels]

    results = pd.DataFrame({
        "Image Path": image_paths,
        "Cluster Label": cluster_labels,
        "Category": categories
    })
    return results


def prepare_images(output_dirs, image_dir):
    """
    Normalize data, generate graphs, and save them as images in the specified directory.
    Args:
        output_dirs (dict): A dictionary containing the paths to output directories.
        image_dir (str): The directory to save generated images.
    """
    os.makedirs(image_dir, exist_ok=True)

    for label, (category, output_dir) in enumerate(output_dirs.items()):
        category_dir = os.path.join(image_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".csv") and not file.endswith("_poi.csv"):
                    data_path = os.path.join(root, file)
                    qdp = QDataPipeline(data_filepath=data_path)
                    t_delta = qdp.find_time_delta()
                    if t_delta > 0:

                        downsampling_factor = qdp.get_downsampling_factor()
                        print(
                            f"[INFO] Applying downsampling with factor of {downsampling_factor}")
                        qdp.downsample(k=t_delta, factor=downsampling_factor)
                    data_df = qdp.get_dataframe()

                    if "Dissipation" in data_df.columns:
                        # Normalize and generate graph
                        data_df["normalized_dissipation"] = normalize_data(
                            data_df)
                        graph_path = os.path.join(
                            category_dir, f"{os.path.splitext(file)[0]}.jpg")
                        generate_graph(data_df, graph_path)


def normalize_data(data):
    """
    Normalize the dissipation column in the dataset.
    Args:
        data (pd.DataFrame): The dataset containing a dissipation column.

    Returns:
        pd.Series: The normalized dissipation column.
    """
    dissipation = data["Dissipation"]
    normalized = (dissipation - dissipation.min()) / \
        (dissipation.max() - dissipation.min())
    return normalized


def generate_graph(data_df, save_path):
    """
    Generate a graph from the dissipation column and save as an image.
    The graph contains only the line with no legends, titles, or annotations.

    Args:
        data_df (pd.DataFrame): The dataset containing a dissipation column.
        save_path (str): The path to save the graph image.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(data_df["normalized_dissipation"], color="blue",
             linewidth=2)  # Line plot with custom styling
    plt.axis("off")  # Turn off axes
    # Save without extra white space
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    # Paths to the output directories
    output_dirs = {
        "full_fill": "content/dropbox_dump",
        "no_fill": "content/no_fill",
        "channel_1_partial": "content/channel_1",
        "channel_2_partial": "content/channel_2",
    }
    categories = list(output_dirs.keys())
    image_dir = "./image_data"

    # Step 1: Prepare image data (if not already prepared)
    prepare_images(output_dirs, image_dir)

    # Step 2: Load images and split into training and testing datasets
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))

    # Split data into training and testing sets
    train_paths, test_paths = train_test_split(
        image_paths, test_size=0.2, random_state=42)

    # Step 3: Feature extraction
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    feature_extractor = Model(inputs=base_model.input,
                              outputs=base_model.output)

    # Extract features for training and testing sets
    train_features = extract_features(train_paths, feature_extractor)
    test_features = extract_features(test_paths, feature_extractor)

    # Step 4: Clustering
    n_clusters = 4  # Set the number of clusters
    kmeans = cluster_images(train_features, n_clusters)

    # Map clusters to categories
    train_labels = kmeans.predict(train_features)
    cluster_to_category = map_clusters_to_categories(
        train_paths, train_labels, categories)
    print(f"Cluster to Category Mapping: {cluster_to_category}")
    # Predict cluster labels for test data
    test_labels = kmeans.predict(test_features)

    # Evaluate predictions
    evaluate_predictions(test_paths, test_labels,
                         cluster_to_category, categories)
    # Predict and map categories for test data
    # test_results = predict_and_map(
    #     test_paths, feature_extractor, kmeans, cluster_to_category)
    # print(test_results)

    # # Save results
    # test_results.to_csv(
    #     "test_clustering_results_with_categories.csv", index=False)

    # # Example prediction for a new image
    # new_image_path = test_paths[0]
    # new_image_results = predict_and_map(
    #     [new_image_path], feature_extractor, kmeans, cluster_to_category)
    # print(new_image_results)
