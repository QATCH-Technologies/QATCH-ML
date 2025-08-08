import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# ← adjust these imports to match your project
from v4_predictor import POIPredictor
from v4_dp import DP


def benchmark_per_poi(predictor, content_dir):
    # Step 1: collect raw actuals & preds per POI
    actuals = defaultdict(list)
    preds = defaultdict(list)

    items = list(DP.load_content(content_dir))
    for data_fp, label_fp in tqdm(items, desc="Benchmarking files", unit="file"):
        df = pd.read_csv(data_fp)
        true_idxs = pd.read_csv(
            label_fp, header=None).values.flatten().astype(int)
        best = predictor.predict_best(df)
        cons = predictor.apply_constraints(best, df)

        for poi_num, pred in cons.items():
            actuals[poi_num].append(true_idxs[poi_num - 1])
            preds[poi_num].append(pred['data_index'])

    # Step 2: compute metrics per POI
    metrics = {
        'MAE': {},
        'MSE': {},
        'RMSE': {},
        'MAPE (%)': {},
        'Median AE': {},
        'Accuracy': {}
    }

    for poi_num in sorted(actuals):
        a = np.array(actuals[poi_num])
        p = np.array(preds[poi_num])
        e = p - a
        abs_e = np.abs(e)

        mae = abs_e.mean()
        mse = (e**2).mean()
        rmse = np.sqrt(mse)
        nonz = a != 0
        mape = (np.abs(e[nonz] / a[nonz])).mean() * 100
        med_ae = np.median(abs_e)

        # “Window-accuracy” at ±window_size
        window = predictor.window_size
        acc = np.mean(abs_e <= window)

        metrics['MAE'][poi_num] = mae
        metrics['MSE'][poi_num] = mse
        metrics['RMSE'][poi_num] = rmse
        metrics['MAPE (%)'][poi_num] = mape
        metrics['Median AE'][poi_num] = med_ae
        metrics['Accuracy'][poi_num] = acc

        # also print to console if you like
        print(f"\n=== POI {poi_num} ===")
        print(f" MAE:       {mae:.3f}")
        print(f" MSE:       {mse:.3f}")
        print(f" RMSE:      {rmse:.3f}")
        print(f" MAPE:      {mape:.2f}%")
        print(f" Median AE: {med_ae:.3f}")
        print(f" Accuracy (@±{window}): {acc:.3f}")

    # Step 3: plot each metric as a bar‐chart
    poi_nums = sorted(actuals.keys())
    for metric_name, vals in metrics.items():
        plt.figure(figsize=(8, 4))
        # bar‐style “histogram”
        plt.bar(poi_nums, [vals[poi] for poi in poi_nums], width=0.6)
        plt.xlabel("POI")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} per POI")
        plt.xticks(poi_nums)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    predictor = POIPredictor(
        model_path="QModel/src/models/static_v4/v4_model.h5",
        scaler_path="QModel/src/models/static_v4/v4_scaler.joblib",
        window_size=128,
        stride=16,
        tolerance=64
    )
    benchmark_per_poi(predictor, "content/static/valid")
