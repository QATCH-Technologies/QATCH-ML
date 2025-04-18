import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
df = pd.read_csv(
    r"C:\Users\QATCH\dev\QATCH-ML\QModel\src\models\formulations\FormulaTRaining_v2.csv")

# Define features (inputs) and targets (outputs)
X = df[['Protein', 'Sucrose (M)']]
Y = df[['Viscosity@100', 'Viscosity@1000', 'Viscosity@10000',
        'Viscosity@100000', 'Viscosity@15000000']]

# Split data into training and testing sets (80% training, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = Pipeline([
    # ('standard', StandardScaler()),
    ('minmax', MinMaxScaler(feature_range=(0, 1)))
])
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(
    scaler, r"C:\Users\QATCH\dev\QATCH-ML\QModel\src\models\formulations\scaler.pkl")
print("Scaler saved as 'scaler.pkl'")

# ---------------------------
# 1. Linear Regression Model
# ---------------------------
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, Y_train)
linear_pred = linear_model.predict(X_test_scaled)
mse_linear = mean_squared_error(Y_test, linear_pred)
print("Linear Regression MSE:", mse_linear)
joblib.dump(linear_model,
            r"C:\Users\QATCH\dev\QATCH-ML\QModel\src\models\formulations\linear_regression_model.pkl")
print("Linear Regression model saved as 'linear_regression_model.pkl'")

# ---------------------------
# 2. Polynomial Regression Model (degree = 2)
# ---------------------------
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])
poly_model.fit(X_train_scaled, Y_train)
poly_pred = poly_model.predict(X_test_scaled)
mse_poly = mean_squared_error(Y_test, poly_pred)
print("Polynomial Regression MSE:", mse_poly)
joblib.dump(
    poly_model, r"C:\Users\QATCH\dev\QATCH-ML\QModel\src\models\formulations\polynomial_regression_model.pkl")
print("Polynomial Regression model saved as 'polynomial_regression_model.pkl'")

# ---------------------------
# 3. Random Forest Regression Model
# ---------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, Y_train)
rf_pred = rf_model.predict(X_test_scaled)
mse_rf = mean_squared_error(Y_test, rf_pred)
print("Random Forest Regression MSE:", mse_rf)
joblib.dump(
    rf_model, r"C:\Users\QATCH\dev\QATCH-ML\QModel\src\models\formulations\random_forest_model.pkl")
print("Random Forest model saved as 'random_forest_model.pkl'")
# ---------------------------
# 4. Neural Network Regression Model
# ---------------------------
nn_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(Y_train.shape[1])
])

nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
nn_model.fit(X_train_scaled, Y_train, epochs=100, batch_size=8, verbose=0)

nn_pred = nn_model.predict(X_test_scaled)
mse_nn = mean_squared_error(Y_test, nn_pred)
print("Neural Network Regression MSE:", mse_nn)
nn_model.save("QModel/src/formulations/viscosity_model.h5")
print("Neural Network model saved as 'viscosity_model.h5'")
# ---------------------------
# 5. XGBoost Models (One per target)
# ---------------------------
# XGBoost does not natively support multi-output regression in a single booster.
# Instead, we loop over each target column and train a separate model.
xgb_models = {}      # Dictionary to store the models
mse_xgb = {}         # Dictionary to store MSE for each target
params = {
    'objective': 'reg:squarederror',  # Use reg:squarederror for regression
    'eval_metric': 'rmse',
    'seed': 42
}
num_rounds = 100

for col in Y.columns:
    print(f"\nTraining XGBoost model for target: {col}")
    # Create DMatrix for training and testing for the given target
    dtrain = xgb.DMatrix(X_train_scaled, label=Y_train[col])
    dtest = xgb.DMatrix(X_test_scaled, label=Y_test[col])

    # Train the model using the XGBoost native API
    xgb_model = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=[
                          (dtest, 'eval')], verbose_eval=False)

    # Make predictions and calculate MSE for the current target
    xgb_pred = xgb_model.predict(dtest)
    mse_val = mean_squared_error(Y_test[col], xgb_pred)
    print(f"XGBoost MSE for {col}: {mse_val}")

    # Save the model to a file (one file per target)
    model_filename = f"C:/Users/QATCH/dev/QATCH-ML/QModel/src/models/formulations/xgboost_model_{col}.json"
    xgb_model.save_model(model_filename)
    print(f"XGBoost model for {col} saved as '{model_filename}'")

    # Store the model and its MSE in dictionaries
    xgb_models[col] = xgb_model
    mse_xgb[col] = mse_val
# Optionally, save the dictionary of models using joblib
joblib.dump(
    xgb_models, r"C:\Users\QATCH\dev\QATCH-ML\QModel\src\models\formulations\xgboost_models.pkl")
print("\nAll XGBoost models saved as 'xgboost_models.pkl'")
