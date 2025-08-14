
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import os
from v4_dp import DP
import warnings
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from sklearn.utils.class_weight import compute_class_weight


def _process_file(args):
    data_file, poi_file, window_size, stride, ignore_index = args
    data_df = pd.read_csv(data_file)
    poi_arr = pd.read_csv(poi_file, header=None).values
    feature_df = DP.gen_features(data_df)
    feature_df.drop(columns="Relative_time", inplace=True)
    results = []

    for i, poi_index in enumerate(poi_arr.flatten().tolist()):
        if i != ignore_index:
            labeled_df = DataGen.label_df(
                feature_df, poi_index=poi_index)
            poi_windows = DataGen.generate_windows(
                data_df=labeled_df,
                window_size=window_size,
                poi_index=poi_index,
                poi_num=i,
                stride=stride
            )
            results.append((f"poi_{i+1}", poi_windows))

    return results


class DataGen:
    IGNORE_INDEX = 2
    DEFAULT_WINDOW_SIZE = 128
    DEFAULT_STRIDE = 16

    def __init__(self, data_dir: str,
                 num_datasets: int = 10,
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 stride: int = DEFAULT_STRIDE):
        self._content = DP.load_content(
            data_dir=data_dir, num_datasets=num_datasets)
        self._window_size = window_size
        self._stride = stride

    def gen(self) -> dict[str, list]:
        all_windows = {"poi_1": [], "poi_2": [],
                       "poi_4": [], "poi_5": [], "poi_6": []}

        args_list = [
            (data_file, poi_file, self._window_size,
             self._stride, self.IGNORE_INDEX)
            for data_file, poi_file in self._content
        ]

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_process_file, args)
                       for args in args_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc="<Processing Datasets>"):
                results = future.result()
                for poi_key, poi_windows in results:
                    all_windows[poi_key].append(poi_windows)
                    random.shuffle(all_windows[poi_key])

        return all_windows

    @staticmethod
    def generate_windows(data_df: pd.DataFrame, window_size: int, poi_index: int, poi_num: int, stride: int):
        total_len = len(data_df)
        if poi_index < 0 or poi_index >= total_len:
            raise ValueError("POI index is out of range.")
        start_min = max(0, poi_index - window_size + 1)
        start_max = min(poi_index, total_len - window_size)

        windows = []
        for start in range(start_min, start_max + 1, stride):
            end = start + window_size
            extracted_window = data_df.iloc[start:end]
            extracted_window["POI_Num"] = poi_num
            extracted_window.reset_index(drop=True, inplace=True)
            windows.append(extracted_window)

        return windows

    @staticmethod
    def label_df(data_df: pd.DataFrame, poi_index: int):
        labeled_df = data_df.copy()
        labeled_df["label"] = 0
        if 0 <= poi_index < len(labeled_df):
            labeled_df.loc[poi_index, "label"] = 1
        else:
            raise IndexError(
                f"POI index {poi_index} is out of range for dataframe of length {len(labeled_df)}")
        return labeled_df
