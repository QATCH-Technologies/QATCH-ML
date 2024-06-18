import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter


class QDataVisualizer:
    def __init__(self, single_dataset=None, training_dataset=None):
        self.single_dataset = single_dataset
        self.training_dataset = training_dataset

    def plot_data(self, pois, predictions):
        """
        Plots the dataset with Points of Interest (POIs) and predictions.

        Args:
            pois (list): A list of Points of Interest.
            predictions (list): A list of predictions corresponding to the dataset.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.dataset[:, 0],
            self.dataset[:, 1],
            c=predictions,
            cmap="viridis",
            label="Data Points",
        )
        plt.scatter(pois[:, 0], pois[:, 1], c="red", label="POIs", marker="x")
        plt.title("Data with Points of Interest and Predictions")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

    def plot_correlations(self):
        """
        Plots a heatmap of the correlation matrix of the dataset.
        """
        corr = pd.DataFrame(self.dataset).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Matrix Heatmap")
        plt.show()

    def plot_poi_distribution(self, pois):
        """
        Plots the distribution of Points of Interest (POIs).

        Args:
            pois (list): A list of Points of Interest.
        """
        poi_counts = Counter(pois)
        plt.figure(figsize=(10, 6))
        plt.bar(poi_counts.keys(), poi_counts.values(), color="blue")
        plt.title("POI Distribution")
        plt.xlabel("POI")
        plt.ylabel("Frequency")
        plt.show()

    def plot_tsne(self, labels=None):
        """
        Plots a 2D t-SNE visualization of the dataset.

        Args:
            labels (list, optional): A list of labels for coloring the data points.
        """
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(self.dataset)

        plt.figure(figsize=(10, 6))
        if labels is not None:
            plt.scatter(
                tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis"
            )
            plt.colorbar()
        else:
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        plt.title("t-SNE Plot")
        plt.xlabel("TSNE Component 1")
        plt.ylabel("TSNE Component 2")
        plt.show()

    def plot_imbalance(self, labels):
        """
        Plots the class imbalance of the dataset.

        Args:
            labels (list): A list of labels for the dataset.
        """
        label_counts = Counter(labels)
        plt.figure(figsize=(10, 6))
        plt.bar(label_counts.keys(), label_counts.values(), color="green")
        plt.title("Class Imbalance")
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.show()

    def plot_confusion(self, true_labels, predicted_labels):
        """
        Plots the confusion matrix for the given true and predicted labels.

        Args:
            true_labels (list): The true labels of the dataset.
            predicted_labels (list): The predicted labels of the dataset.
        """
        cm = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()
