# plot decision tree
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt


class QModelVisuzalizer:
    def __init__(
        self, model, training_dataset=None, valid_dataset=None, test_dataset=None
    ):
        self.__model__ = model
        self.__training_dataset__ = training_dataset
        self.__valid_dataset__ = valid_dataset
        self.__test_dataset__ = test_dataset

    def plot_training_history(self):
        pass

    def plot_xgbtree(self):
        plot_tree(self.__model__)
        plt.show()
