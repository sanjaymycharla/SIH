import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
data = pd.read_excel('C:/Users/mycha/OneDrive/Documents/sih avalanche project/testing dataset/Book1.xlsx')
X =data[['Temperature  (deg F) ', 'Relative Humidity  (%) ', 'Total Snow Depth  (") ', 'Intermittent/Shot Snow  (") ']]
y =data['label']
scaler = StandardScaler()
X  = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


model = keras.Sequential([
    keras.layers.LSTM(units=64, activation='relu', input_shape=(1, X_train.shape[2]), return_sequences=True),
    keras.layers.LSTM(units=64, activation='relu', return_sequences=True),
    keras.layers.LSTM(units=64, activation='relu'),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64
          , validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
