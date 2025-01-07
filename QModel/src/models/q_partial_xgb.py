import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from q_data_pipeline import QPartialDataPipeline
from q_multi_model import QMultiModel
import xgboost as xgb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

FILL_TYPE = {"full_fill": 0, "channel_1_partial": 1,
             "channel_2_partial": 2, "no_fill": 3}
FILL_TYPE_R = {0: "full_fill", 1: "channel_1_partial",
               2: "channel_2_partial", 3: "no_fill"}


def process_file(file_path, label):
    """
    Process a single file and return its features and label.
    """
    qpd = QPartialDataPipeline(data_filepath=file_path)
    qpd.preprocess()
    features = qpd.get_features()
    return features, label


def load_and_prepare_data(dataset_paths):
    """
    Load and preprocess data from the given dataset paths using multithreading.
    """
    X, y = [], []
    tasks = []

    with ThreadPoolExecutor() as executor:
        for label, path in dataset_paths.items():
            for root, _, files in os.walk(path):
                files = [
                    f for f in files if f.endswith(".csv") and not f.endswith("_poi.csv")
                ]
                for file in files:
                    file_path = os.path.join(root, file)
                    tasks.append(executor.submit(
                        process_file, file_path, label))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing datasets"):
            try:
                features, label = future.result()
                X.append(features)
                y.append(FILL_TYPE[label])
            except Exception as e:
                print(f"Error processing file: {e}")

    # Combine features into a DataFrame and handle labels
    X_df = pd.DataFrame(X).fillna(0)

    # Combine X_df and y_encoded into a single DataFrame
    combined_df = X_df.copy()
    combined_df['Class'] = y

    predictors = X_df.columns.tolist()
    return predictors, combined_df


def train_xgboost_model(dataset, predictors, target_features, num_targets):
    qmm = QMultiModel(dataset=dataset, predictors=predictors,
                      target_features=target_features, num_targets=num_targets, eval_metric="merror")
    qmm.tune()
    qmm.train_model()
    qmm.save_model("partial_qmm")


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


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
    input(predicted_class_index)
    return FILL_TYPE_R[np.argmax(predicted_class_index)]


if __name__ == "__main__":
    dataset_paths = {
        "full_fill": "content/dropbox_dump",
        "no_fill": "content/no_fill",
        "channel_1_partial": "content/channel_1",
        "channel_2_partial": "content/channel_2",
    }

    predictors, dataset = load_and_prepare_data(dataset_paths)
    train_xgboost_model(dataset=dataset, predictors=predictors,
                        target_features="Class", num_targets=4)

    test_file = "content/dropbox_dump/00001/DD240125W1_C5_OLDBSA367_3rd.csv"
    predicted_class = predict(test_file)
    print(f"Predicted dataset type: {predicted_class}")
