import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Check scikit-learn version
print("Scikit-learn version:", sklearn.__version__) 

# Load the dataset
iris = pd.read_csv(r"C:\Users\mukun\OneDrive\Documents\iris.csv")

# Basic inspection
print(iris.head())
print(iris.describe())

# Filtering the dataset based on certain conditions
print(iris[iris['Sepal.Width'] > 4])
print(iris[iris['Petal.Width'] > 1])
print(iris[iris['Petal.Width'] > 2])

# Scatter plot of Sepal Length vs Petal Length colored by species
sns.scatterplot(x='Sepal.Length', y='Petal.Length', data=iris, hue='Species')
plt.show()

# Model 1: Linear regression using Sepal.Width to predict Sepal.Length
y = iris[['Sepal.Length']]
x = iris[['Sepal.Width']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Fit the linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Make predictions
y_pred = lr.predict(x_test)
print("Predicted values (first 5):", y_pred[0:5])
print("Actual values (first 5):")
print(y_test.head())

# Calculate mean squared error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (Model 1):", mse)

# Model 2: Linear regression using multiple features (Sepal.Width, Petal.Length, Petal.Width)
y = iris[['Sepal.Length']]
x = iris[['Sepal.Width', 'Petal.Length', 'Petal.Width']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Fit the second linear regression model
lr2 = LinearRegression()
lr2.fit(x_train, y_train)

# Make predictions
y_pred = lr2.predict(x_test)

# Calculate mean squared error for the second model
mse2 = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (Model 2):", mse2)
