# Regression Linear Analysis

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data into a pandas DataFrame
dataframe = pd.read_csv('final_cars_datasets.csv')

# Separate features (X) and targets (y)
X = dataframe[['year', 'mileage', 'engine_capacity']]
y = dataframe['price']

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train_scaled, y_train)

# Price prediction using test data
y_pred = model.predict(X_test_scaled)

# Model evaluation using Mean Squared Error (MSE) and R-squared metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared: {r2}')
print('Koefisien Regresi:', model.coef_)
print('Konstanta Regresi:', model.intercept_)