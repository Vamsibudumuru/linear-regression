# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Step 2: Load Dataset
california = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target  # MedHouseVal is the target variable

# Step 3: Explore and Clean Data
print(df.head())
print(df.describe())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Step 4: Split Data into Training and Testing Sets
# Define features (X) and target (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and Train the Linear Regression Model
# Initialize the model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Step 7: Make Predictions
# Make a prediction for a single example
example = X_test.iloc[0].values.reshape(1, -1)
predicted_value = model.predict(example)
print(f"Predicted value: {predicted_value}")
print(f"Actual value: {y_test.iloc[0]}")

# Optional: Visualize the relationship between actual and predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Home Values")
plt.show()
