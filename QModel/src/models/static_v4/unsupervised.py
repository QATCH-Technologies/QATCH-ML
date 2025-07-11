
import matplotlib.pyplot as plt
import ruptures as rpt
from q_model_data_processor import QDataProcessor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import pandas as pd


def load_data_and_labels(load_dir, max_datasets=100):
    content = QDataProcessor.load_content(load_dir, num_datasets=max_datasets)
    X_list, y_list = [], []
    for data_path, poi_path in content:
        df = pd.read_csv(data_path)
        X = df[['Dissipation']].values.astype(np.float32)
        X = MinMaxScaler((0, 1)).fit_transform(X)

        poi_idxs = np.loadtxt(poi_path, int)
        T = X.shape[0]
        y = np.zeros((T,), np.int32)
        poi_idxs = poi_idxs[(poi_idxs >= 0) & (poi_idxs < T)]
        y[poi_idxs] = 1

        X_list.append(X)
        y_list.append(y)
    return X_list, y_list


LOAD_DIRECTORY = os.path.join("content", "dropbox_dump")
X_list, y_list = load_data_and_labels(LOAD_DIRECTORY)


def detect_change_points(dissipation: np.ndarray,
                         pen: float = 6,
                         model: str = "rbf"):
    """
    Returns a list of indices (breakpoints) where the dissipation curve changes.
    You may tune `pen` (penalty) to get roughly the number of POIs you expect.
    """
    algo = rpt.Pelt(model=model).fit(dissipation)
    bkps = algo.predict(pen=pen)
    # `bkps` includes the last index (len), so filter if desired:
    return [idx for idx in bkps if idx < len(dissipation)]


pts = detect_change_points(X_list[0])
plt.figure()
plt.plot(X_list[0])
plt.scatter(pts, X_list[0][pts])
plt.show()
