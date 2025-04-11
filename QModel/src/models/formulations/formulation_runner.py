import numpy as np
import tensorflow as tf
import joblib
import xgboost as xgb

# Load the Keras neural network model.
nn_model = tf.keras.models.load_model(
    r"QModel\src\models\formulations\viscosity_model.h5")
print("Neural Network model loaded successfully.")

# Load scikit-learn models.
linear_model = joblib.load(
    r"QModel\src\models\formulations\linear_regression_model.pkl")
print("Linear Regression model loaded successfully.")

poly_model = joblib.load(
    r"QModel\src\models\formulations\polynomial_regression_model.pkl")
print("Polynomial Regression model loaded successfully.")

rf_model = joblib.load(
    r"QModel\src\models\formulations\random_forest_model.pkl")
print("Random Forest model loaded successfully.")

# Load the feature scaler used during training.
scaler = joblib.load(
    r"QModel\src\models\formulations\scaler.pkl")
print("Feature scaler loaded successfully.")

# Load the dictionary of XGBoost models that were saved during training.
# Each target was modeled individually, so this dictionary keys should be the target names.
xgb_models = joblib.load(
    r"QModel\src\models\formulations\xgboost_models.pkl")
print("XGBoost models loaded successfully.")

# Define the target order (make sure it is the same order as used during training).
target_order = ['Viscosity@100', 'Viscosity@1000', 'Viscosity@10000',
                'Viscosity@100000', 'Viscosity@15000000']

print("\nEnter 'quit' or 'exit' at any time to close the program.")

while True:
    user_input = input(
        "Enter protein concentration (mg/ml) and sucrose concentration (M) separated by a space: ")
    if user_input.strip().lower() in ['quit', 'exit']:
        print("Exiting...")
        break

    try:
        parts = user_input.strip().split()
        if len(parts) != 2:
            print("Please provide exactly two numbers separated by a space.\n")
            continue

        protein_input = float(parts[0])
        sucrose_input = float(parts[1])
    except ValueError:
        print("Invalid input. Make sure you enter numerical values.\n")
        continue

    # Prepare the sample input and scale it.
    new_sample = np.array([[protein_input, sucrose_input]])
    new_sample_scaled = scaler.transform(new_sample)

    # Neural network prediction.
    nn_pred = nn_model.predict(new_sample_scaled)

    # Linear regression prediction.
    linear_pred = linear_model.predict(new_sample_scaled)

    # Polynomial regression prediction.
    poly_pred = poly_model.predict(new_sample_scaled)

    # Random forest prediction.
    rf_pred = rf_model.predict(new_sample_scaled)

    # XGBoost predictions.
    # For XGBoost each individual model is applied to the same scaled input.
    new_sample_dmatrix = xgb.DMatrix(new_sample_scaled)
    xgb_pred = []
    for target in target_order:
        booster = xgb_models[target]
        pred_val = booster.predict(new_sample_dmatrix)
        xgb_pred.append(pred_val[0])

    print("\nPredicted Viscosities (order: Viscosity@100, Viscosity@1000, Viscosity@10000, Viscosity@100000, Viscosity@15000000):")

    print("\nNeural Network Predictions:")
    print("  Viscosity@100:       {:.2f}".format(nn_pred[0][0]))
    print("  Viscosity@1000:      {:.2f}".format(nn_pred[0][1]))
    print("  Viscosity@10000:     {:.2f}".format(nn_pred[0][2]))
    print("  Viscosity@100000:    {:.2f}".format(nn_pred[0][3]))
    print("  Viscosity@15000000:  {:.2f}".format(nn_pred[0][4]))

    print("\nLinear Regression Predictions:")
    print("  Viscosity@100:       {:.2f}".format(linear_pred[0][0]))
    print("  Viscosity@1000:      {:.2f}".format(linear_pred[0][1]))
    print("  Viscosity@10000:     {:.2f}".format(linear_pred[0][2]))
    print("  Viscosity@100000:    {:.2f}".format(linear_pred[0][3]))
    print("  Viscosity@15000000:  {:.2f}".format(linear_pred[0][4]))

    print("\nPolynomial Regression Predictions:")
    print("  Viscosity@100:       {:.2f}".format(poly_pred[0][0]))
    print("  Viscosity@1000:      {:.2f}".format(poly_pred[0][1]))
    print("  Viscosity@10000:     {:.2f}".format(poly_pred[0][2]))
    print("  Viscosity@100000:    {:.2f}".format(poly_pred[0][3]))
    print("  Viscosity@15000000:  {:.2f}".format(poly_pred[0][4]))

    print("\nRandom Forest Predictions:")
    print("  Viscosity@100:       {:.2f}".format(rf_pred[0][0]))
    print("  Viscosity@1000:      {:.2f}".format(rf_pred[0][1]))
    print("  Viscosity@10000:     {:.2f}".format(rf_pred[0][2]))
    print("  Viscosity@100000:    {:.2f}".format(rf_pred[0][3]))
    print("  Viscosity@15000000:  {:.2f}".format(rf_pred[0][4]))

    print("\nXGBoost Predictions:")
    print("  Viscosity@100:       {:.2f}".format(xgb_pred[0]))
    print("  Viscosity@1000:      {:.2f}".format(xgb_pred[1]))
    print("  Viscosity@10000:     {:.2f}".format(xgb_pred[2]))
    print("  Viscosity@100000:    {:.2f}".format(xgb_pred[3]))
    print("  Viscosity@15000000:  {:.2f}".format(xgb_pred[4]))

    print("")
