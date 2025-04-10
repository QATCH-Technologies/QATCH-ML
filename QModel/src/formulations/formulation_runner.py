import numpy as np
import tensorflow as tf
import joblib


model = tf.keras.models.load_model("viscosity_model.h5")
print("Model loaded successfully.")

scaler = joblib.load("scaler.pkl")

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

    new_sample = np.array([[protein_input, sucrose_input]])

    new_sample_scaled = scaler.transform(new_sample)

    predictions = model.predict(new_sample_scaled)

    print("\nPredicted Viscosities:")
    print("Viscosity@100:       {:.2f}".format(predictions[0][0]))
    print("Viscosity@1000:      {:.2f}".format(predictions[0][1]))
    print("Viscosity@10000:     {:.2f}".format(predictions[0][2]))
    print("Viscosity@100000:    {:.2f}".format(predictions[0][3]))
    print("Viscosity@15000000:  {:.2f}".format(predictions[0][4]))
    print("")
