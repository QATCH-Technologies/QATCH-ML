"""
Module for Data Processing and Machine Learning Pipeline with XGBoost.

This module includes functions and classes for preprocessing data, performing dimensionality
reduction, and building machine learning pipelines using XGBoost. It provides utilities for
normalizing data, visualizing feature correlations, and applying dimensionality reduction
techniques such as PCA and t-SNE. It also includes functionality for resampling datasets to
balance class distributions using SMOTE and Random Under-Sampling.

The module integrates various components such as:
    - `normalize`: Normalizes numpy arrays to the range [0, 1].
    - `correlation_view`: Visualizes Pearson correlation matrices.
    - `tsne_view`: Applies and visualizes t-SNE for dimensionality reduction.
    - `pca_view`: Applies and visualizes PCA for dimensionality reduction.
    - `resample_df`: Resamples a DataFrame using SMOTE and Random Under-Sampling.
    - `load_content`: Loads paths of CSV files from a specified directory while excluding
    certain file patterns.
    - `xgb_pipeline`: Processes multiple datasets for XGBoost preprocessing.

Additionally, the module handles the training and testing of models using the `QMultiModel` and
`QPredictor` classes, and supports visualization of predictions versus actual values.

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - imbalanced-learn
    - scikit-learn
    - tqdm
    - joblib

Classes:
    - `QDataPipeline`: For preprocessing data.
    - `QMultiModel`: For building and training multi-target models.
    - `QPredictor`: For making predictions with a trained model.

Usage:
The module script can be run directly to perform end-to-end training and testing of models.
Configuration and file paths are set through global variables and are dependent on the specific
project setup.

Example:
    python QTrainMulti.py

Author(s):
    Paul MacNichol (paulmacnichol@gmail.com)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import joblib
from QConstants import M_TARGET, TRAINING, FEATURES, TESTING, PLOTTING
from q_data_pipeline import QDataPipeline
from q_multi_model import QMultiModel, QPredictor


def normalize(arr: np.ndarray = None) -> np.ndarray:
    """
    Normalize a numpy array to the range [0, 1].

    Args:
        arr (numpy.ndarray): The array to be normalized.

    Returns:
        numpy.ndarray: The normalized array with values scaled to the range [0, 1].

    Raises:
        ValueError: If `arr` is empty or all values in `arr` are the same.
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def correlation_view(dataframe: pd.DataFrame = None) -> None:
    """
    Visualize the Pearson correlation matrix of features in a DataFrame.

    This function creates a heatmap of the Pearson correlation coefficients between the features
    of the provided DataFrame. The heatmap is displayed using the `RdBu` colormap.

    Args:
        df (pandas.DataFrame): The DataFrame containing the features for which
                            the correlation matrix will be computed and visualized.

    Returns:
        None: This function does not return any value. It displays the heatmap plot directly.

    Raises:
        ValueError: If the DataFrame is empty or does not contain numeric data.
    """
    colormap = plt.cm.RdBu

    plt.figure(figsize=(14, 12))
    plt.title("Pearson Correlation of Features", y=1.05, size=15)
    sns.heatmap(
        dataframe.astype(float).corr(),
        linewidths=0.1,
        vmax=1.0,
        square=True,
        cmap=colormap,
        linecolor="white",
        annot=True,
    )
    plt.show()


def tsne_view(x: np.ndarray, y: np.ndarray) -> None:
    """
    Visualize data using t-SNE for dimensionality reduction and plotting.

    This function applies t-Distributed Stochastic Neighbor Embedding (t-SNE) to reduce the
    dimensionality of the data to 2 dimensions and creates a scatter plot. The points in the
    scatter plot are colored according to the target values provided.

    Args:
        x (numpy.ndarray or pandas.DataFrame): The input data to be reduced in dimensionality.
                                              Should have more than 2 dimensions.
        y (numpy.ndarray or pandas.Series): The target values corresponding to the data points.
                                            These are used to color the scatter plot.

    Returns:
        None: This function does not return any value. It displays the scatter plot directly.

    Raises:
        ValueError: If `x` and `y` have incompatible dimensions or `x` does not have more than
                    2 dimensions.
    """
    print("[INFO] building t-SNE")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    x_tsne = tsne.fit_transform(x)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        x_tsne[:, 0], x_tsne[:, 1], c=y, cmap="viridis", edgecolor="k"
    )
    plt.title("t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(scatter, label="Target Value")
    plt.show()


def pca_view(x: np.ndarray, y: np.ndarray) -> None:
    """
    Visualize data using PCA for dimensionality reduction and plotting.

    This function applies Principal Component Analysis (PCA) to reduce the dimensionality of the
    data to 2 dimensions and creates a scatter plot. The points in the scatter plot are colored
    according to the target values provided.

    Args:
        x (numpy.ndarray or pandas.DataFrame): The input data to be reduced in dimensionality.
                                              Should have more than 2 dimensions.
        y (numpy.ndarray or pandas.Series): The target values corresponding to the data points.
                                            These are used to color the scatter plot.

    Returns:
        None: This function does not return any value. It displays the scatter plot directly.

    Raises:
        ValueError: If `x` and `y` have incompatible dimensions or `x` does not have more than
                    2 dimensions.
    """
    print("[INFO] building PCA")
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap="viridis", edgecolor="k")
    plt.title("PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Target Value")
    plt.show()


def resample_df(data: pd.DataFrame, target: str, droppable: list) -> pd.DataFrame:
    """
    Resample a DataFrame using SMOTE and Random Under-Sampling.

    This function performs resampling on the provided DataFrame to balance the class distribution.
    It first applies SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority
    class, then applies Random Under-Sampling to balance the classes, and finally returns the
    resampled DataFrame.

    Args:
        data (pandas.DataFrame): The input DataFrame to be resampled.
        target (str): The name of the column in `data` that contains the
        target variable to be resampled.
        droppable (list of str): A list of column names to be excluded
        from the resampling process.

    Returns:
        pandas.DataFrame: A DataFrame with the same columns as the input `data`
                            (excluding `droppable`) and the resampled target column.

    Raises:
        KeyError: If `target` or any columns in `droppable` are not present in `data`.
    """
    print(f"[INFO] resampling target='{target}'")
    y = data[target].values
    x = data.drop(columns=droppable)

    over = SMOTE(sampling_strategy="auto")
    under = RandomUnderSampler(sampling_strategy="auto")
    steps = [
        ("o", over),
        ("u", under),
    ]
    pipeline = Pipeline(steps=steps)

    x, y = pipeline.fit_resample(x, y)

    resampled_df = pd.DataFrame(x, columns=data.drop(columns=droppable).columns)
    resampled_df[target] = y
    return resampled_df


def load_content(data_dir: str) -> list:
    """
    Load paths of CSV files from a specified directory, excluding certain file patterns.

    This function recursively traverses the given directory to find and collect paths of
    CSV files, excluding those with filenames ending in "_poi.csv" or "_lower.csv".

    Args:
        data_dir (str): The path to the directory where CSV files will be searched.

    Returns:
        list of str: A list of file paths for CSV files that do not end with "_poi.csv"
        or "_lower.csv".

    Raises:
        OSError: If the provided `data_dir` is not a valid directory or cannot be accessed.
    """
    print(f"[INFO] Loading content from {data_dir}")
    loaded_content = []

    for data_root, _, data_files in os.walk(data_dir):
        for f in data_files:
            if (
                f.endswith(".csv")
                and not f.endswith("_poi.csv")
                and not f.endswith("_lower.csv")
            ):
                loaded_content.append(os.path.join(data_root, f))
    return loaded_content


def xgb_pipeline(training_content: list) -> pd.DataFrame:
    """
    Process multiple datasets for XGBoost preprocessing.

    This function iterates over a list of dataset file paths, processes each CSV file (excluding
    those ending with "_poi.csv"), and applies preprocessing using the `QDataPipeline` class.
    It concatenates the processed data into a single DataFrame and performs resampling.

    Args:
        train_content (list of str): A list of file paths to the CSV datasets to be processed.

    Returns:
        pandas.DataFrame: A DataFrame containing the resampled data after preprocessing.

    Raises:
        FileNotFoundError: If the file paths do not exist or cannot be accessed.
        ValueError: If the DataFrame contains NaN values after preprocessing.
    """
    print(f"[INFO] XGB Preprocessing on {len(training_content)} datasets")
    xgb_df = pd.DataFrame()
    for f in tqdm(training_content, desc="<<Processing XGB>>"):
        if f.endswith(".csv") and not f.endswith("_poi.csv"):
            qdp_pipeline = QDataPipeline(f, multi_class=True)
            matched_poi_file = f.replace(".csv", "_poi.csv")
            if not os.path.exists(matched_poi_file):
                continue
            t_delta = qdp_pipeline.find_time_delta()
            actual = pd.read_csv(matched_poi_file, header=None).values

            if (
                t_delta > 0
                and max(qdp_pipeline.__dataframe__["Relative_time"]) < 1800
                and max(actual) < len(qdp_pipeline.__dataframe__) - 1
            ):

                qdp_pipeline.preprocess(poi_filepath=matched_poi_file)

                indices = qdp_pipeline.__dataframe__.index[
                    qdp_pipeline.__dataframe__["Class"] != 0
                ].tolist()
                # print(f"actual: {actual}")
                qdp_pipeline.downsample(t_delta, 20)
                has_nan = qdp_pipeline.__dataframe__.isna().any().any()
                if not has_nan:
                    xgb_df = pd.concat([xgb_df, qdp_pipeline.get_dataframe()])

    resampled_df = resample_df(xgb_df, M_TARGET, M_TARGET)
    return resampled_df


if __name__ == "__main__":
    print("[INFO] QTrainMulti.py script start")
    vals = [0, 1, 2]
    for t in vals:

        model_name = f"QMultiType_long"
        print(f"[INFO] Training {model_name}")
        TRAIN_PATH = r"C:\Users\QATCH\dev\QATCH-ML\content\training_data"

        if TRAINING:
            train_content = load_content(TRAIN_PATH)
            training_set = xgb_pipeline(train_content)
            print("[INFO] Building multi-target model")
            qmodel = QMultiModel(
                dataset=training_set, predictors=FEATURES, target_features=M_TARGET
            )
            qmodel.tune()
            qmodel.train_model()
            qmodel.save_model(model_name)
        if TESTING:
            cluster_model = joblib.load(
                f"C:\\Users\\QATCH\\dev\\QATCH-ML\\QModel\\SavedModels\\label_{t}.pkl"
            )
            data_df = pd.DataFrame()
            content = []
            qmp = QPredictor(
                f"C:\\Users\\QATCH\\dev\\QATCH-ML\\QModel\\SavedModels\\{model_name}.json"
            )
            TEST_PATH = f"content/label_{t}/test"
            for root, dirs, files in os.walk(TEST_PATH):
                for file in files:
                    content.append(os.path.join(root, file))

            for filename in tqdm(content, desc="<<Testing on files>>"):
                if (
                    filename.endswith(".csv")
                    and not filename.endswith("_poi.csv")
                    and not filename.endswith("_lower.csv")
                ):
                    data_file = filename
                    poi_file = filename.replace(".csv", "_poi.csv")
                    actual_indices = pd.read_csv(poi_file, header=None).values
                    qdp = QDataPipeline(data_file)
                    time_delta = qdp.find_time_delta()

                    qdp.preprocess(poi_filepath=None)
                    print("[INFO] Predicting using multi-target model")
                    predictions = qmp.predict(data_file)
                    if PLOTTING:
                        palette = sns.color_palette("husl", 6)
                        df = pd.read_csv(data_file)
                        dissipation = normalize(df["Dissipation"])
                        print("Actual, Predicted")
                        for actual, predicted in zip(actual_indices, predictions):
                            print(f"{actual[0]}, {predicted}")
                        plt.figure()

                        plt.plot(
                            dissipation,
                            color="grey",
                            label="Dissipation",
                        )
                        for i, index in enumerate(predictions):
                            plt.axvline(
                                x=index,
                                color=palette[i],
                                label=f"Predicted POI {i + 1}",
                            )
                        for i, index in enumerate(actual_indices):
                            plt.axvline(
                                x=index,
                                color=palette[i],
                                linestyle="dashed",
                                label=f"Actual POI {i + 1}",
                            )
                        plot_name = data_file.replace(TRAIN_PATH, "")
                        plt.xlabel("POIs")
                        plt.ylabel("Dissipation")
                        plt.title(f"Predicted/Actual POIs on Data: {plot_name}")

                        plt.legend()
                        plt.grid(True)
                        plt.show()
