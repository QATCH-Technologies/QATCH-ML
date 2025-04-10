#!/usr/bin/env python3
"""
run_dataset_splitter.py

This module provides a class for splitting run directories into training, testing,
and validation sets with balanced representation. It extracts features from each
run directory, uses k-means clustering to group similar runs, and performs a
stratified split based on the clustering results. In addition, it can determine the
optimal number of clusters via the silhouette score and generate a PCA-based visualization
of the clusters.

Author: Paul MacNichol (paul.macnichol@qatchtech.com)
Date: 04-02-2025

Example:
    To run the dataset splitting procedure from the command line:

        $ python run_dataset_splitter.py --source path/to/dropbox_dump \
            --output path/to/static --train-ratio 0.7 --test-ratio 0.15 --valid-ratio 0.15
"""

import os
import shutil
import logging
import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BalancedSpliter:
    """A class to split run directories into train, test, and validation sets with balanced representation.

    The class extracts features from each run directory, clusters them using k-means,
    determines the optimal number of clusters if requested, performs a stratified split,
    copies the directories to corresponding output folders, and visualizes the clustering.
    """

    def __init__(self,
                 source: Path,
                 output: Path,
                 train_ratio: float = 0.7,
                 test_ratio: float = 0.15,
                 valid_ratio: float = 0.15,
                 clusters: int = 3,
                 optimize_clusters: bool = True,
                 min_clusters: int = 2,
                 max_clusters: int = 10,
                 seed: int = 42,
                 log_level: str = "INFO"):
        """Initializes the RunDatasetSplitter.

        Args:
            source (Path): Path to the source dropbox_dump directory.
            output (Path): Path to the output directory for the splits.
            train_ratio (float): Proportion of data to assign to training.
            test_ratio (float): Proportion of data to assign to testing.
            valid_ratio (float): Proportion of data to assign to validation.
            clusters (int): Number of clusters to use for grouping similar runs if not optimizing.
            optimize_clusters (bool): Whether to determine the optimal number of clusters using silhouette score.
            min_clusters (int): Minimum number of clusters to consider when optimizing.
            max_clusters (int): Maximum number of clusters to consider when optimizing.
            seed (int): Random seed for reproducibility.
            log_level (str): Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        self.source = source
        self.output = output
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.clusters = clusters
        self.optimize_clusters = optimize_clusters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.seed = seed
        self.log_level = log_level.upper()

        logging.getLogger().setLevel(self.log_level)
        self.output_train = self.output / "train"
        self.output_test = self.output / "test"
        self.output_valid = self.output / "valid"

    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        """Parses command-line arguments.

        Returns:
            argparse.Namespace: Parsed arguments.
        """
        default_source = os.path.join("content", "dropbox_dump")
        default_output = os.path.join("content", "static")
        parser = argparse.ArgumentParser(
            description="Split processed run directories into train, test, and valid sets with balanced representation."
        )
        parser.add_argument("--source",
                            type=str,
                            default=default_source,
                            help="Path to the source dropbox_dump directory. (Default: %(default)s)")
        parser.add_argument("--output",
                            type=str,
                            default=default_output,
                            help="Path to the output directory for train, test, and valid splits. (Default: %(default)s)")
        parser.add_argument("--train-ratio",
                            type=float,
                            default=0.7,
                            help="Proportion of data to assign to training. (Default: %(default)s)")
        parser.add_argument("--test-ratio",
                            type=float,
                            default=0.15,
                            help="Proportion of data to assign to testing. (Default: %(default)s)")
        parser.add_argument("--valid-ratio",
                            type=float,
                            default=0.15,
                            help="Proportion of data to assign to validation. (Default: %(default)s)")
        parser.add_argument("--clusters",
                            type=int,
                            default=3,
                            help="Number of clusters to use for grouping similar runs if not optimizing. (Default: %(default)s)")
        parser.add_argument("--optimize-clusters",
                            default=True,
                            action="store_true",
                            help="Determine the optimal number of clusters using silhouette score.")
        parser.add_argument("--min-clusters",
                            type=int,
                            default=2,
                            help="Minimum number of clusters to consider when optimizing. (Default: %(default)s)")
        parser.add_argument("--max-clusters",
                            type=int,
                            default=10,
                            help="Maximum number of clusters to consider when optimizing. (Default: %(default)s)")
        parser.add_argument("--seed",
                            type=int,
                            default=42,
                            help="Random seed for reproducibility. (Default: %(default)s)")
        parser.add_argument("--log-level",
                            type=str,
                            default="INFO",
                            help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
        return parser.parse_args()

    @classmethod
    def from_args(cls) -> "BalancedSpliter":
        """Creates an instance of RunDatasetSplitter from command-line arguments.

        Returns:
            RunDatasetSplitter: An instance configured using parsed arguments.
        """
        args = cls.parse_arguments()
        return cls(
            source=Path(args.source),
            output=Path(args.output),
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            valid_ratio=args.valid_ratio,
            clusters=args.clusters,
            optimize_clusters=args.optimize_clusters,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            seed=args.seed,
            log_level=args.log_level,
        )

    def extract_features_from_run(self, run_dir: Path) -> Optional[List[float]]:
        """Extracts a feature vector from a given run directory.

        The feature vector is based on:
          - Mean and standard deviation of six unique POI values from the *_poi.csv file.
          - Maximum of the 'Relative_time' column.
          - Variance of the 'Dissipation' column.
          - Variance of the 'Resonance_Frequency' column.
          - Number of rows in the data file.

        Args:
            run_dir (Path): The directory of the run.

        Returns:
            Optional[List[float]]: A list of extracted features or None if extraction fails.
        """
        # Process the POI file.
        poi_files = list(run_dir.glob("*_poi.csv"))
        if not poi_files:
            logging.error(
                f"Run directory {run_dir.name} lacks a POI file; this run will be skipped.")
            return None

        poi_file = poi_files[0]
        try:
            with poi_file.open("r") as f:
                reader = csv.reader(f)
                poi_values = [int(row[0].strip())
                              for row in reader if row and row[0].strip() != ""]
            if len(poi_values) != 6 or len(set(poi_values)) != 6:
                logging.error(
                    f"POI file {poi_file.name} in run directory {run_dir.name} does not contain six unique positive integers.")
                return None
            mean_poi = np.mean(poi_values)
            std_poi = np.std(poi_values)
        except Exception as e:
            logging.error(
                f"Error reading POI file {poi_file.name} in {run_dir.name}: {e}")
            return None

        # Identify the data file.
        data_files = [f for f in run_dir.glob("*.csv")
                      if not f.name.endswith("_poi.csv") and "_lower" not in f.name and "_tec" not in f.name]
        if not data_files:
            logging.error(
                f"Run directory {run_dir.name} lacks a proper data file; this run will be skipped.")
            return None

        data_file = data_files[0]
        try:
            df = pd.read_csv(data_file)
            required_columns = ['Relative_time',
                                'Dissipation', 'Resonance_Frequency']
            if not all(col in df.columns for col in required_columns):
                logging.error(
                    f"Data file {data_file.name} in run directory {run_dir.name} is missing one or more required columns: {required_columns}.")
                return None
            max_relative_time = df['Relative_time'].max()
            var_dissipation = df['Dissipation'].var()
            var_resonance = df['Resonance_Frequency'].var()
            row_count = len(df)
        except Exception as e:
            logging.error(
                f"Error processing data file {data_file.name} in run directory {run_dir.name}: {e}")
            return None

        features = [mean_poi, std_poi, max_relative_time,
                    var_dissipation, var_resonance, row_count]
        logging.debug(f"Extracted features for run {run_dir.name}: {features}")
        return features

    def determine_optimal_clusters(self, features: List[List[float]]) -> Tuple[int, Dict[int, float]]:
        """Determines the optimal number of clusters using the silhouette score.

        Args:
            features (List[List[float]]): List of feature vectors.

        Returns:
            Tuple[int, Dict[int, float]]: The optimal number of clusters and a dictionary mapping
            each candidate cluster count to its silhouette score.
        """
        features_arr = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_arr)

        best_k = None
        best_score = -1
        scores = {}
        for k in range(self.min_clusters, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.seed)
            labels = kmeans.fit_predict(features_scaled)
            score = silhouette_score(features_scaled, labels)
            scores[k] = score
            logging.info(f"Silhouette score for {k} clusters: {score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k
        logging.info(
            f"Optimal number of clusters determined: {best_k} with silhouette score: {best_score:.3f}")
        return best_k, scores

    def stratified_split(self, run_dirs: List[Path], features: List[List[float]]
                         ) -> Tuple[List[int], List[int], List[int], List[int], KMeans, StandardScaler]:
        """Performs a stratified split based on k-means clustering.

        The runs are clustered using their feature vectors, and then the indices are split
        into training, testing, and validation sets ensuring each split has a balanced
        representation from each cluster.

        Args:
            run_dirs (List[Path]): List of run directory paths.
            features (List[List[float]]): List of feature vectors corresponding to run directories.

        Returns:
            Tuple containing:
                - List[int]: Indices for the training set.
                - List[int]: Indices for the testing set.
                - List[int]: Indices for the validation set.
                - List[int]: Cluster labels for each run.
                - KMeans: The fitted k-means clustering model.
                - StandardScaler: The scaler used for standardization.
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        features_arr = np.array(features)

        # Standardize features.
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_arr)

        # Cluster using k-means.
        kmeans = KMeans(n_clusters=self.clusters, random_state=self.seed)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Organize indices by cluster.
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            clusters.setdefault(label, []).append(idx)

        train_indices, test_indices, valid_indices = [], [], []

        # Stratify the split per cluster.
        for label, indices in clusters.items():
            random.shuffle(indices)
            n = len(indices)
            n_train = int(n * self.train_ratio)
            n_test = int(n * self.test_ratio)
            # Ensure that remaining indices go to validation.
            train_indices.extend(indices[:n_train])
            test_indices.extend(indices[n_train:n_train + n_test])
            valid_indices.extend(indices[n_train + n_test:])

        logging.info(
            f"Stratified split completed. Cluster distribution: {clusters}")
        return train_indices, test_indices, valid_indices, list(cluster_labels), kmeans, scaler

    def copy_runs(self, run_dirs: List[Path], indices: List[int], destination: Path) -> None:
        """Copies run directories corresponding to the provided indices to the destination.

        Args:
            run_dirs (List[Path]): List of run directories.
            indices (List[int]): List of indices representing which runs to copy.
            destination (Path): Destination directory to copy the runs into.
        """
        for idx in tqdm(indices, desc=f"Copying to {destination.name}"):
            src_dir = run_dirs[idx]
            dst_dir = destination / src_dir.name
            try:
                shutil.copytree(src_dir, dst_dir)
            except Exception as e:
                logging.error(
                    f"Error copying run directory {src_dir.name} to {destination}: {e}")

    def visualize_clusters(self, features: List[List[float]], cluster_labels: List[int],
                           kmeans: KMeans, scaler: StandardScaler) -> None:
        """Generates a PCA-based visualization of the clusters.

        The method standardizes the features, reduces them to 2D using PCA, and creates a scatter
        plot where each run is colored by its cluster assignment. Cluster centroids are overlaid.

        Args:
            features (List[List[float]]): List of feature vectors.
            cluster_labels (List[int]): Cluster label for each run.
            kmeans (KMeans): The fitted k-means model.
            scaler (StandardScaler): The scaler used for standardization.
        """
        features_arr = np.array(features)
        features_scaled = scaler.transform(features_arr)
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1],
                              c=cluster_labels, cmap='viridis', s=50)
        plt.title("Run Clusters Visualization (PCA Projection)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label='Cluster')

        # Transform cluster centroids to PCA space.
        centers_scaled = kmeans.cluster_centers_
        centers_pca = pca.transform(centers_scaled)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                    c='red', s=200, marker='X', label='Centroids')
        plt.legend()
        plt.show()

    def split_dataset(self) -> None:
        """Executes the dataset splitting procedure.

        This includes:
          - Creating output directories.
          - Extracting features from valid run directories.
          - Optionally determining the optimal number of clusters.
          - Performing a stratified split.
          - Copying run directories to train, test, and validation folders.
          - Visualizing the clustering results.
        """
        # Create output directories.
        for d in [self.output_train, self.output_test, self.output_valid]:
            d.mkdir(parents=True, exist_ok=True)

        # Collect feature vectors from each run directory.
        run_dirs = []
        features = []
        logging.info("Beginning feature extraction from run directories.")
        for run_dir in sorted(self.source.iterdir()):
            if run_dir.is_dir():
                feats = self.extract_features_from_run(run_dir)
                if feats is not None:
                    run_dirs.append(run_dir)
                    features.append(feats)
        logging.info(
            f"Successfully extracted features from {len(features)} run directories.")

        if not features:
            logging.error("No valid runs found. Exiting.")
            return

        # Determine number of clusters if optimization is enabled.
        if self.optimize_clusters:
            optimal_k, _ = self.determine_optimal_clusters(features)
            self.clusters = optimal_k
            logging.info(
                f"Using optimized number of clusters: {self.clusters}")

        # Perform stratified split based on clustering.
        train_idx, test_idx, valid_idx, cluster_labels, kmeans, scaler = self.stratified_split(
            run_dirs, features)
        logging.info(
            f"Total runs: {len(run_dirs)}; Train: {len(train_idx)}, Test: {len(test_idx)}, Valid: {len(valid_idx)}")

        # Copy the runs into their respective directories.
        logging.info("Commencing copying of train run directories.")
        self.copy_runs(run_dirs, train_idx, self.output_train)
        logging.info("Commencing copying of test run directories.")
        self.copy_runs(run_dirs, test_idx, self.output_test)
        logging.info("Commencing copying of valid run directories.")
        self.copy_runs(run_dirs, valid_idx, self.output_valid)

        logging.info("Dataset splitting procedure completed successfully.")

        # Visualize clusters.
        self.visualize_clusters(features, cluster_labels, kmeans, scaler)


def main():
    """Main function to run the dataset splitting procedure."""
    splitter = BalancedSpliter.from_args()
    splitter.split_dataset()


if __name__ == "__main__":
    main()
