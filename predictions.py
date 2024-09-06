import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

input_data = pd.read_excel('C:/Users/mycha/OneDrive/Documents/sih avalanche project/testing dataset/testing(avalanche) - Copy.xlsx')

input_data = scaler.transform(input_data)

input_data = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))

predictions = model.predict(input_data)

predictions = model.predict(input_data)

for i, prediction in enumerate(predictions):
    if prediction >= 0.2:
        print(f"Avalanche {i + 1} is predicted to happen.")
    else:
        print(f"Avalanche {i + 1} is predicted not to happen.")

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

y_pred = model.predict(X_test)

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})

correlation_matrix = results_df.corr()
plt.figure(figsize=(8, 6))
plt.title('Correlation Matrix')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
confusion = confusion_matrix(y_test, y_pred_binary)
sensitivity = confusion[1,1]/(confusion[1, 1] + confusion[1, 1])
print(f"Sensitivity (True Positive Rate): {sensitivity:4f}")

y_pred = model.predict(X_test)

# Apply a threshold (e.g., 0.5) to classify predictions
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.4f}")
