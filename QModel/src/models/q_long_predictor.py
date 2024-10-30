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
            r"C:\Users\QATCH\dev\QATCH-ML\QModel\SavedModels\QSingle4_long.json"
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

    def adjustment_poi_6(self, candidates, poi_5_guess, data):
        candidates = candidates[candidates >= poi_5_guess]
        candidates = candidates[:10]
        plt.figure(figsize=(8, 8))
        plt.plot(self.normalize(data.__dataframe__["Dissipation"]), color="black")
        plt.scatter(
            candidates, self.normalize(data.__dataframe__["Dissipation"])[candidates]
        )
        plt.plot(self.normalize(data.__dataframe__["Difference"]), color="grey")
        plt.scatter(
            candidates, self.normalize(data.__dataframe__["Difference"])[candidates]
        )
        plt.plot(
            self.normalize(data.__dataframe__["Resonance_Frequency"]), color="brown"
        )
        plt.scatter(
            candidates,
            self.normalize(data.__dataframe__["Resonance_Frequency"])[candidates],
        )
        plt.show()

        return candidates

    def predict(self, data: QDataPipeline, t_delta: int):
        original_df = data.__dataframe__
        data.preprocess()
        data.downsample(k=t_delta, factor=20)
        df = data.get_dataframe()
        f_names = self.__model__.feature_names
        df = df[f_names]
        d_data = xgb.DMatrix(df)

        results = self.__model__.predict(d_data)
        # extracted = self.extract_results(results)

        candidates_6 = self.find_and_sort_peaks(results)

        # candidates_6 = self.adjustment_poi_6(candidates_6, candidates_5[0], data)
        confidence_6 = np.array(results)[candidates_6]
        remapped_6 = []
        for poi_6 in candidates_6:
            r_time_6 = data.__dataframe__.at[poi_6, "Relative_time"]
            remapped_idx_6 = original_df[
                original_df["Relative_time"] == r_time_6
            ].index[0]
            remapped_6.append(remapped_idx_6)
        palette = sns.color_palette("husl", 7)
        # plt.figure(figsize=(8, 8))
        # plt.plot(self.normalize(after["Dissipation"]), label="After")
        # for i, res in enumerate(results):
        #     prediction = np.argmax(res)
        #     r_time = data.__dataframe__.at[prediction, "Relative_time"]
        #     print(
        #         i, prediction, original_df[original_df["Relative_time"] == r_time].index
        #     )
        #     plt.axvline(x=np.argmax(res), color=palette[i], label=f"POI {i}")
        # plt.legend()
        # plt.show()
        candidates = (np.array(remapped_6), confidence_6)

        return candidates
