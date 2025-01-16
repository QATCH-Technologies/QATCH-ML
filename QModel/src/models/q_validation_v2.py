"""
This will probably have data contamination which will need to be resolved for a  true evaluation of the system.
1. For each dataset in a subdiretory of content/test_data/test (listed as 0000, 00001, 00002, etc.) (ignore any sub-directories that have a capture.zip folder)
    - Make 4 new partitioned datasets from the data in the file no suffixed with _poi.csv.
        * The first is randomly chosen between index 0 and the POI 1 listed in the suffixed file _poi.csv (labeled no_fill)
        * The second is randomly chosen bewteen the index of POI 4 and the index of POI 5 in the suffixed _poi.csv file (labeled channel_1)
        * The third is randomly chosen between th index of POI 5 and POI 6 of the suffixed _poi.csv file in this directory. (labeled channel_2)
        * The final dataset will be identical to the orinal (labeled full_fill)
    - For each new partioned dataset, pass it to QPredictor from q_predictor.py
        * Collect the fill type (no_fill, channel_1, channel_2, and full_fill) as well as the predicted POI indices from the predictor.
2. Create a set of metrics based on the predictions
    - For each partitioned dataset, measure the accuracy of the predicted fill type
    - For each predicted POI measure the MAE and MAPE (with outliers and ignoring outliers) from the true POI indices in the _poi.csv file for each true fill type.
3. Build graphic visualizations for these metrics
"""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from q_predictor_v2 import QPredictor
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import tempfile


def load_and_partition_datasets(base_dir, data_percentage=0.1):
    """
    Loads and partitions datasets from the given directory and calculates accuracy.

    Parameters:
    - base_dir (str): The base directory containing the datasets.
    - data_percentage (float): The fraction of files to process. Must be between 0 and 1.

    Returns:
    - results (list): A list of dictionaries containing fill types, predicted POIs, and true POIs.
    - accuracy (float): The overall accuracy of predicted fill types.
    """
    if not (0 <= data_percentage <= 1):
        raise ValueError("data_percentage must be between 0 and 1.")

    results = []
    true_labels = []
    predicted_labels = []

    # Get all subdirectories and files
    dataset_paths = [
        (subdir, files)
        for subdir, _, files in os.walk(base_dir)
        if 'capture.zip' not in files
    ]

    # Calculate the number of datasets to load based on the fraction
    total_datasets = len(dataset_paths)
    datasets_to_load = int(data_percentage * total_datasets)

    # Randomly sample the datasets
    sampled_paths = random.sample(dataset_paths, datasets_to_load)
    # Initialize predictor
    predictor = QPredictor()
    # , redirect_stdout(StringIO())
    # Initialize progress bar
    with tqdm(total=len(sampled_paths), desc="Processing datasets") as pbar:
        for subdir, files in sampled_paths:
            main_file = next(
                (f for f in files if not f.endswith('_poi.csv')), None)
            poi_file = next((f for f in files if f.endswith('_poi.csv')), None)
            if not main_file or not poi_file:
                pbar.update(1)
                continue

            data_path = os.path.join(subdir, main_file)
            poi_path = os.path.join(subdir, poi_file)

            # Load data
            data = pd.read_csv(data_path)
            poi_indices = pd.read_csv(
                poi_path, header=None).iloc[:, 0].tolist()

            # Partition data
            partitions = {
                'no_fill': data.iloc[0:random.randint(0, poi_indices[3])],
                'channel_1_partial': data.iloc[0: random.randint(poi_indices[3], poi_indices[4])],
                'channel_2_partial': data.iloc[0: random.randint(poi_indices[4], poi_indices[5])],
                'full_fill': data
            }

            for fill_type, partition in partitions.items():
                if partition.empty:
                    continue  # Skip empty partitions

                # Write partition to a temporary CSV file
                temp_file_path = tempfile.NamedTemporaryFile(
                    delete=False, suffix='.csv').name
                partition.to_csv(temp_file_path, index=False)

                # Pass the file path to the predictor

                prediction = predictor.predict(temp_file_path, fill_type)
                predicted_type = prediction[0]
                true_labels.append(fill_type)
                predicted_labels.append(predicted_type)
                predicted_pois = unpack_pois(prediction[1], prediction[0])
                # if fill_type != "no_fill" and prediction[0] != 'no_fill':
                results.append({
                    'true_fill_type': fill_type,
                    'predicted_fill_type': predicted_type,
                    'predicted_pois': predicted_pois,
                    'true_pois': poi_indices
                })
                # Cleanup temporary file
                os.remove(temp_file_path)
        pbar.update(1)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"Overall Accuracy: {accuracy:.2%}")
    visualize_accuracy(true_labels, predicted_labels)

    return results, accuracy


def visualize_accuracy(true_labels, predicted_labels):
    """
    Visualizes the accuracy using a confusion matrix.

    Parameters:
    - true_labels (list): List of true fill type labels.
    - predicted_labels (list): List of predicted fill type labels.
    """
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels,
                          labels=np.unique(true_labels))
    labels = np.unique(true_labels)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix: True vs Predicted Fill Types')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def unpack_pois(poi_data, pred_fill_type):
    best_pois = []
    if pred_fill_type == "no_fill":
        return [-1]
    for poi in poi_data:
        best_pois.append(poi[0][0])
    return best_pois


def compute_metrics_per_poi(results):
    """
    Compute MAE and MAPE metrics on a per-POI basis.

    Parameters:
    - results (list): A list of dictionaries containing true and predicted values.

    Returns:
    - metrics (dict): A dictionary with per-POI MAE and MAPE metrics.
    """
    poi_metrics = {f'POI_{i}': {'mae': [], 'mape': []} for i in range(1, 7)}

    for result in results[0]:
        true_pois = result['true_pois']
        predicted_pois = result['predicted_pois']

        # Ensure predicted_pois and true_pois have the same length
        num_pois = min(len(true_pois), len(
            predicted_pois), 6)  # Limit to POI 1-6

        for i in range(num_pois):
            true_value = true_pois[i]
            predicted_value = predicted_pois[i]

            # Calculate MAE
            mae = abs(predicted_value - true_value)

            # Calculate MAPE (Avoid division by zero)
            mape = abs((predicted_value - true_value) / true_value) * \
                100 if true_value != 0 else None

            poi_metrics[f'POI_{i + 1}']['mae'].append(mae)
            poi_metrics[f'POI_{i + 1}']['mape'].append(mape)

    # Average the metrics for each POI
    averaged_metrics = {}
    for poi, values in poi_metrics.items():
        averaged_metrics[poi] = {
            'mae': np.mean(values['mae']) if values['mae'] else None,
            'mape': np.mean(values['mape']) if values['mape'] else None
        }

    return averaged_metrics


def visualize_poi_metrics(poi_metrics):
    """
    Visualize MAE and MAPE metrics for each POI.

    Parameters:
    - poi_metrics (dict): A dictionary with per-POI MAE and MAPE metrics.
    """
    pois = list(poi_metrics.keys())
    mae_values = [poi_metrics[poi]['mae'] for poi in pois]
    mape_values = [poi_metrics[poi]['mape'] for poi in pois]

    # MAE Bar Chart
    plt.figure(figsize=(10, 5))
    plt.bar(pois, mae_values, alpha=0.7)
    plt.title('Mean Absolute Error (MAE) by POI')
    plt.ylabel('MAE')
    plt.xlabel('POI')
    for i, v in enumerate(mae_values):
        plt.text(
            i, v + 0.02, f"{v:.2f}" if v is not None else "N/A", ha='center')
    plt.show()

    # MAPE Bar Chart
    plt.figure(figsize=(10, 5))
    plt.bar(pois, mape_values, alpha=0.7)
    plt.title('Mean Absolute Percentage Error (MAPE) by POI')
    plt.ylabel('MAPE (%)')
    plt.xlabel('POI')
    for i, v in enumerate(mape_values):
        plt.text(
            i, v + 0.02, f"{v:.2f}%" if v is not None else "N/A", ha='center')
    plt.show()


if __name__ == "__main__":
    base_dir = "content/test_data/test"
    # try:
    results = load_and_partition_datasets(base_dir, data_percentage=0.1)
    poi_metrics = compute_metrics_per_poi(results)
    visualize_poi_metrics(poi_metrics)
    print("Per-POI metrics processing complete!")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
