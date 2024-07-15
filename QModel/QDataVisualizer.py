import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter


class QDataVisualizer:
    def __init__(self, dataset=None, poi=None):
        self.__dataset__ = dataset
        self.__poi__ = poi

    def with_dissipation(self, *arrays):
        if "Dissipation" not in self.__dataset__.columns:
            raise ValueError("The dataframe does not have a 'Dissipation' column")

        # Extract the Dissipation column from the dataframe
        dissipation = self.__dataset__["Dissipation"]

        # Plot each array against the Dissipation column
        plt.figure()
        plt.plot(dissipation, label="Dissipation")
        for i, array in enumerate(arrays):
            if len(array) != len(dissipation):
                raise ValueError(
                    f"Array {i+1} length ({len(array)}) does not match the length of 'Dissipation' column ({len(dissipation)})"
                )
            plt.plot(array, label=f"Array {i+1}")

        plt.xlabel("Dissipation")
        plt.ylabel("Array Values")
        plt.title("Arrays vs. Dissipation")
        plt.legend()
        plt.grid(True)
        plt.show()

    def with_resonance(self):
        pass

    def with_difference(self):
        pass

    def plot_correlation(self):
        corr_matrix = self.__dataset__.corr()
        plt.figure()
        sns.heatmap(corr_matrix, cmap="magma", robust=True, square=True, cbar=True)
        plt.title("Feature Correlation")
        plt.show()
