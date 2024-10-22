import matplotlib.pyplot as plt
from q_data_pipeline import QDataPipeline
import xgboost as xgb
import numpy as np


class QLongPredictor:
    def __init__(self):
        self.__model__ = xgb.Booster()

        self.__model__.load_model(
            r"C:\Users\QATCH\dev\QATCH-ML\QModel\SavedModels\QMultiType_long.json"
        )

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def predict(self, data: QDataPipeline, t_delta: int):

        data.downsample(k=t_delta, factor=20)
        data.preprocess()
        after = data.__dataframe__
        df = data.get_dataframe()
        f_names = self.__model__.feature_names
        df = df[f_names]
        d_data = xgb.DMatrix(df)

        results = self.__model__.predict(d_data)
        plt.figure(figsize=(8, 8))
        # plt.plot(before["Dissipation"], label="Before")
        plt.plot(self.normalize(after["Dissipation"]), label="After")
        for i in results:
            print(i)
            results_norm = self.normalize(results[i])
            plt.axvline(x=np.argmax(results_norm))
        plt.legend()
        plt.show()
