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
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import tempfile
from q_data_pipeline import QPartialDataPipeline
import xgboost as xgb
import matplotlib.pyplot as plt


FILL_TYPE = {"full_fill": 0, "channel_1_partial": 1,
             "channel_2_partial": 2, "no_fill": 3}
FILL_TYPE_R = {0: "full_fill", 1: "channel_1_partial",
               2: "channel_2_partial", 3: "no_fill"}


def predict(input_file_path, model_path="QModel/SavedModels/partial_qmm.json"):
    model = xgb.Booster()
    model.load_model(model_path)
    qpd = QPartialDataPipeline(data_filepath=input_file_path)
    qpd.preprocess()
    features = qpd.get_features()
    features_df = pd.DataFrame([features]).fillna(0)
    f_names = model.feature_names
    df = features_df[f_names]
    d_data = xgb.DMatrix(df)
    predicted_class_index = model.predict(d_data)
    return FILL_TYPE_R[np.argmax(predicted_class_index)]


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
                'no_fill': data.iloc[0:random.randint(0, poi_indices[2])],
                'channel_1_partial': data.iloc[0:random.randint(poi_indices[2], poi_indices[3])],
                'channel_2_partial': data.iloc[0:random.randint(poi_indices[3], poi_indices[5])],
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

                predicted_type = predict(temp_file_path)
                true_labels.append(fill_type)
                predicted_labels.append(predicted_type)
                # if fill_type != "no_fill" and prediction[0] != 'no_fill':
                results.append({
                    'true_fill_type': fill_type,
                    'predicted_fill_type': predicted_type,
                })
                # if fill_type != predicted_type:
                #     plt.figure()

                #     plt.plot(data["Dissipation"],
                #              color='r', linestyle='dotted',  label=poi_indices)
                #     for idx in poi_indices:
                #         plt.axvline(idx)
                #     plt.plot(partition['Dissipation'])
                #     plt.title(
                #         f"Predicted: {predicted_type}, True: {fill_type}")
                #     plt.legend()
                #     plt.show()
                #     print("TRUE | PREDICTED")
                #     print("---")
                #     print(fill_type, predicted_type)

                # Cleanup temporary file
                os.remove(temp_file_path)
        pbar.update(1)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"Overall Accuracy: {accuracy:.2%}")
    visualize_accuracy(true_labels, predicted_labels)


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


if __name__ == "__main__":
    base_dir = "content/test_data/test"
    load_and_partition_datasets(base_dir, data_percentage=1)
