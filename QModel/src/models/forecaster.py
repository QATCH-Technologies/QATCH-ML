import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

# Function to load CSV file paths


def load_content(data_dir: str) -> list:
    print(f"[INFO] Loading content from {data_dir}")
    loaded_content = []

    for data_root, _, data_files in os.walk(data_dir):
        for f in data_files:
            if (
                f.endswith(".csv")
                and not f.endswith("_poi.csv")
                and not f.endswith("_lower.csv")
            ):
                loaded_content.append(os.path.join(data_root, f))

    return loaded_content

# Function to load and preprocess data


def load_and_preprocess_data(data_dir):
    all_data = []

    # Load file paths
    file_paths = load_content(data_dir)

    # Load all CSV files
    for file in file_paths:
        df = pd.read_csv(file)
        if not df.empty and 'Relative_time' in df.columns and 'Dissipation' in df.columns:
            df = df[['Relative_time', 'Dissipation']]
            all_data.append(df)

    # Combine all data and scale it
    if all_data:
        combined_data = pd.concat(all_data).sort_values(by='Relative_time')
    else:
        raise ValueError("No valid data found in the specified directory.")

    scaler = MinMaxScaler()
    combined_data['Dissipation'] = scaler.fit_transform(
        combined_data[['Dissipation']])

    return combined_data, scaler

# Function to create sequences for time series modeling


def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    return np.array(sequences), np.array(targets)

# Training pipeline


def train_xgboost_model(data_dir, sequence_length=30):
    # Load and preprocess data
    data, scaler = load_and_preprocess_data(data_dir)

    # Create sequences
    X, y = create_sequences(data['Dissipation'].values, sequence_length)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 6
    }

    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=100,
                      early_stopping_rounds=10, evals=evals)

    # Save model and scaler
    model.save_model("QModel/SavedModels/forecaster_xgb.json")
    joblib.dump(scaler, "QModel/SavedModels/forecaster_scaler.pkl")

    return model, scaler

# Prediction pipeline


def load_saved_model(model_path, scaler_path):
    model = xgb.Booster()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_future(model, scaler, partial_data, sequence_length=30, num_predictions=1000):
    # Scale partial data
    partial_data_scaled = scaler.transform(partial_data[['Dissipation']])

    # Create initial sequence
    sequence = partial_data_scaled[-sequence_length:].flatten()

    predictions = []
    for _ in range(num_predictions):
        dinput = xgb.DMatrix(sequence.reshape(1, -1))
        predicted_value = model.predict(dinput)[0]
        predictions.append(predicted_value)
        sequence = np.append(sequence[1:], predicted_value)

    # Inverse scale predictions
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1))
    return predictions


# Example usage
data_dir = 'content/training_data/full_fill'
model, scaler = train_xgboost_model(data_dir)
partial_data = pd.read_csv(
    r'content/training_data/channel_1/01152/MM231106W10_Y40P_D4_3rd.csv')
full_data = pd.read_csv(
    r'content/training_data/full_fill/01152/MM231106W10_Y40P_D4_3rd.csv')
model, scaler = load_saved_model(
    "QModel/SavedModels/forecaster_xgb.json", "QModel/SavedModels/forecaster_scaler.pkl")
future_predictions = predict_future(model, scaler, partial_data)
print(future_predictions)

# Align the indices of future_predictions to start where partial_data ends
future_predictions_index = range(len(partial_data), len(
    partial_data) + len(future_predictions))

# Plot the data
plt.figure()
plt.plot(full_data["Dissipation"], color='b', label='Full Data')
plt.plot(partial_data["Dissipation"], color='r', label='Partial Data')
plt.plot(future_predictions_index, future_predictions, color='black',
         linestyle='dotted', label='Future Predictions')
plt.legend()
plt.show()
