import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import joblib
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


df = pd.read_csv("FormulaTRaining_v2.csv")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
metadata.set_primary_key(column_name='ID')

model = CTGANSynthesizer(metadata)
model.fit(df)
synthetic_df = model.sample(500)

X = synthetic_df[['Protein', 'Sucrose (M)']]
Y = synthetic_df[['Viscosity@100', 'Viscosity@1000', 'Viscosity@10000',
                  'Viscosity@100000', 'Viscosity@15000000']]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(5)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

model.fit(X_train_scaled, Y_train, epochs=100, validation_split=0.2)

loss = model.evaluate(X_test_scaled, Y_test)
print("Test MSE:", loss)
model.save("viscosity_model.h5")
print("Model saved as 'viscosity_model.h5'")
