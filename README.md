# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset into a DataFrame and explore its contents to understand the data structure. 
2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.
3.Create a linear regression model and fit it using the training data. 
4.Predict the results for the testing set and plot the training and testing sets with fitted lines.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KRITHI V
RegisterNumber:  212224060128
*/

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/content/exp_2_dataset_student_scores.csv')

print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")

X = df.iloc[:, :-1].values   
Y = df.iloc[:, -1].values

print("X (features):", X.flatten())
print("Y (targets):", Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)

plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

order = np.argsort(X_test.flatten())

X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]

plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)

rmse = np.sqrt(mse)

print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

new_hours = np.array([[2.5], [8.0]])

pred_new = regressor.predict(new_hours)

print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))

```

## Output:
<img width="789" height="449" alt="Screenshot 2026-07-22 210239" src="https://github.com/user-attachments/assets/d4cfb45d-a487-4d29-b8a3-97e8ab3a4e0f" />

<img width="711" height="907" alt="Screenshot 2026-07-22 210257" src="https://github.com/user-attachments/assets/1621f3ce-0d16-41ab-a643-d1ee4e29142f" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
