# Linear-Health-regression-calculator
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 2: Load the dataset
url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
data = pd.read_csv(url)

# Step 3: Data Exploration
print(data.head())  # Display the first few rows
print(data.info())  # Display information about the dataset

# Check for missing values
print(data.isnull().sum())

# Step 4: Data Preprocessing
# Convert categorical columns to numerical
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data = pd.get_dummies(data, columns=['region'], drop_first=True)  # One-hot encode 'region' column

# Step 5: Split the data into train and test datasets
X = data.drop('charges', axis=1)  # Features
y = data['charges']  # Target variable (healthcare costs)

# 80% training data, 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardize the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Create and Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 8: Evaluate the Model
y_pred = model.predict(X_test_scaled)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Step 9: Plot the Results
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red')  # Line of perfect prediction
plt.title('Actual vs Predicted Healthcare Costs')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.show()

# Check if the model meets the challenge requirement
if mae < 3500:
    print("Challenge passed! MAE is below 3500.")
else:
    print("Challenge failed. MAE is above 3500.")
