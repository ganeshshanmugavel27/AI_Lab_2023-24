# Ex.No: 10 MachineLearning-miniproject
### DATE:24/10/2024                                                                        
### REGISTER NUMBER :212222040042
### Weather Prediction Model Using Random Forest Regression
### AIM: 
To build a machine learning model that predicts average daily temperatures based on historical weather data, using a regression approach.
###  Algorithm:
- Data Preprocessing: Load data, remove missing values, and split it into features (X) and target (y).
- Train-Test Split: Divide data into training and testing sets.
- Model Selection: Use a Random Forest Regressor to predict average temperature.
- Training: Train the model on the training data.
- Evaluation: Assess model accuracy using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and visualize results with a pie chart.
### Program:
```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset (replace with your file path if needed)
file_path = '/content/weather_prediction_dataset.csv'  # adjust path as per Colab
data = pd.read_csv(file_path)

# Select the target variable
target = 'BASEL_temp_mean'

# Drop any columns with missing values for simplicity
data = data.dropna(axis=1)

# Split the data into features and target
X = data.drop(columns=[target, 'DATE', 'MONTH'])
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Define maximum temperature range for comparison (accuracy visualization)
temp_range = y.max() - y.min()
accuracy_proportion = (temp_range - mae) / temp_range * 100
error_proportion = mae / temp_range * 100

# Display accuracy metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Accuracy Proportion: {accuracy_proportion:.2f}%")
print(f"Error Proportion: {error_proportion:.2f}%")

# Plot pie chart of accuracy vs error
labels = ['Accuracy', 'Error']
sizes = [accuracy_proportion, error_proportion]
colors = ['#66b3ff', '#ff6666']
explode = (0.1, 0)  # explode 1st slice for emphasis

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Accuracy vs Error Proportion in Temperature Prediction")
plt.show()

```
### Output:
![image](https://github.com/user-attachments/assets/dfd855de-c7c2-45fb-9648-3c74dff34857)
![image](https://github.com/user-attachments/assets/eafb96b0-f138-4474-b12d-c158e1cbedc9)




### Result:

The model's performance is evaluated with Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to measure prediction accuracy. Additionally, an accuracy vs. error pie chart visually shows the model's effectiveness in predicting temperatures.
