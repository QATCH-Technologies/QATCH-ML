import matplotlib.pyplot as plt
from q_data_pipeline import QDataPipeline
import xgboost as xgb
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks


class QLongPredictor:
    def __init__(self):
        self.__model__ = xgb.Booster()

        self.__model__.load_model(
            r"C:\Users\QATCH\dev\QATCH-ML\QModel\SavedModels\QMultiType_long.json"
        )

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def find_and_sort_peaks(self, signal):
        # Find peaks
        peaks, properties = find_peaks(signal)
        # Get the peak heights
        peak_heights = []
        for p in peaks:
            peak_heights.append(signal[p])

        # Sort peaks by height in descending order
        sorted_indices = np.argsort(peak_heights)[::-1]
        sorted_peaks = peaks[sorted_indices]
        return sorted_peaks

    def extract_results(self, results):

        num_indices = len(results[0])
        extracted = [[] for _ in range(num_indices)]

        for sublist in results:
            for idx in range(num_indices):
                extracted[idx].append(sublist[idx])

        return extracted

    def adjustment_poi_6(self, candidates, poi_5_guess):
        return candidates[candidates >= poi_5_guess]

    def predict(self, data: QDataPipeline, t_delta: int):
        original_df = data.__dataframe__
        data.preprocess()
        data.downsample(k=t_delta, factor=20)
        df = data.get_dataframe()
        f_names = self.__model__.feature_names
        df = df[f_names]
        d_data = xgb.DMatrix(df)

        results = self.__model__.predict(d_data)
        extracted = self.extract_results(results)

        candidates_1 = self.find_and_sort_peaks(extracted[1])
        candidates_2 = self.find_and_sort_peaks(extracted[2])
        candidates_3 = self.find_and_sort_peaks(extracted[3])
        candidates_4 = self.find_and_sort_peaks(extracted[4])
        candidates_5 = self.find_and_sort_peaks(extracted[5])
        candidates_6 = self.find_and_sort_peaks(extracted[6])
        candidates_6 = self.adjustment_poi_6(candidates_6, candidates_5[0])
        confidence_1 = np.array(extracted[1])[candidates_1]
        confidence_2 = np.array(extracted[2])[candidates_2]
        confidence_3 = np.array(extracted[3])[candidates_3]
        confidence_4 = np.array(extracted[4])[candidates_4]
        confidence_5 = np.array(extracted[5])[candidates_5]
        confidence_6 = np.array(extracted[6])[candidates_6]
        remapped_1, remapped_2, remapped_3, remapped_4, remapped_5, remapped_6 = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for poi_1, poi_2, poi_3, poi_4, poi_5, poi_6 in zip(
            candidates_1,
            candidates_2,
            candidates_3,
            candidates_4,
            candidates_5,
            candidates_6,
        ):

            r_time_1 = data.__dataframe__.at[poi_1, "Relative_time"]
            r_time_2 = data.__dataframe__.at[poi_2, "Relative_time"]
            r_time_3 = data.__dataframe__.at[poi_3, "Relative_time"]
            r_time_4 = data.__dataframe__.at[poi_4, "Relative_time"]
            r_time_5 = data.__dataframe__.at[poi_5, "Relative_time"]
            r_time_6 = data.__dataframe__.at[poi_6, "Relative_time"]

            remapped_idx_1 = original_df[
                original_df["Relative_time"] == r_time_1
            ].index[0]
            remapped_idx_2 = original_df[
                original_df["Relative_time"] == r_time_2
            ].index[0]
            remapped_idx_3 = original_df[
                original_df["Relative_time"] == r_time_3
            ].index[0]
            remapped_idx_4 = original_df[
                original_df["Relative_time"] == r_time_4
            ].index[0]
            remapped_idx_5 = original_df[
                original_df["Relative_time"] == r_time_5
            ].index[0]
            remapped_idx_6 = original_df[
                original_df["Relative_time"] == r_time_6
            ].index[0]
            remapped_1.append(remapped_idx_1)
            remapped_2.append(remapped_idx_2)
            remapped_3.append(remapped_idx_3)
            remapped_4.append(remapped_idx_4)
            remapped_5.append(remapped_idx_5)
            remapped_6.append(remapped_idx_6)
        # palette = sns.color_palette("husl", 7)
        # plt.figure(figsize=(8, 8))
        # plt.plot(self.normalize(after["Dissipation"]), label="After")
        # for i, res in enumerate(extracted):
        #     prediction = np.argmax(res)
        #     r_time = data.__dataframe__.at[prediction, "Relative_time"]
        #     print(
        #         i, prediction, original_df[original_df["Relative_time"] == r_time].index
        #     )
        #     plt.axvline(x=np.argmax(res), color=palette[i], label=f"POI {i}")
        # plt.legend()
        # plt.show()
        candidates = [
            (np.array(remapped_1), confidence_1),
            (np.array(remapped_2), confidence_2),
            (np.array(remapped_3), confidence_3),
            (np.array(remapped_4), confidence_4),
            (np.array(remapped_5), confidence_5),
            (np.array(remapped_6), confidence_6),
        ]

        return candidates
