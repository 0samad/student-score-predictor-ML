# student_scores.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Dataset
data = {'Hours': [1,2,3,4,5,6,7,8,9],
        'Scores': [35,45,50,55,65,70,75,80,85]}
df = pd.DataFrame(data)

# Prepare data
X = df[['Hours']]
y = df['Scores']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
print(f"Model R² Score: {r2:.2f}")

# Plot
plt.scatter(df['Hours'], df['Scores'], color='blue', label='Actual')
plt.plot(df['Hours'], model.predict(df[['Hours']]), color='red', label='Prediction')
plt.xlabel("Study Hours")
plt.ylabel("Score (%)")
plt.title("Hours vs Scores Prediction")
plt.legend()
plt.show()

# Predict custom input
hours = float(input("Enter study hours: "))
predicted_score = model.predict([[hours]])
print(f"Predicted Score: {predicted_score[0]:.2f}")
