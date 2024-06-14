import pickle
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file_name = "VOYAGER_models/VOYAGER_xgb.pkl"

    # load
    xgb_model = pickle.load(open(file_name, "rb"))
    f_names = xgb_model.feature_names
    in_data = pd.read_csv("VOYAGER_models/W10+QV1862_EL5_L8_3rd.csv").drop(
        columns=["Date", "Time", "Ambient", "Temperature"]
    )

    in_data["Class"] = in_data["Class"] / 30
    actual = in_data["Class"]
    in_data.drop(columns="Class")
    in_data = in_data[f_names]
    in_data = xgb.DMatrix(in_data)
    predictions = xgb_model.predict(in_data)
    plt.plot(actual, color="red", label="Actual")
    plt.plot(predictions, label="Predictions")

    plt.title("Predictions vs. Actual Data for W10+QV1862_EL5_L8_3rd")
    plt.xlabel("Index")
    plt.ylabel("% Confidence")
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()
