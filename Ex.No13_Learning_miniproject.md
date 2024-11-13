# Ex.No: 10 Learning – Use Supervised Learning  
### DATE:                                                                            
### REGISTER NUMBER :212222040022 
### AIM: 
To write a program to train the classifier for Car Recommentation System.
###  Algorithm:
1.Load and Explore Dataset: Import libraries, load the dataset, and check data structure.
2.Data Preprocessing: Encode categorical variables (like fuel and transmission) into numerical values for analysis.
3.Feature Selection and Scaling: Select relevant features (year, km_driven, etc.) and scale them using StandardScaler to normalize.
4.Clustering: Apply KMeans clustering to group cars into three categories (Budget-Friendly, Mid-Range, Luxury) based on selected features.
5.Visualization: Plot distributions of car prices by category and fuel type across categories to understand patterns.
6.Car Recommendation Function: Define recommend_cars() to filter cars by budget and optional preferences (fuel type, transmission), and test with example inputs.
### Program:
```
import pandas as pd
import numpy as n
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Load dataset
dataset = pd.read_csv('car-details-from-car-dekho.csv')
print(dataset.head())
print(dataset.tail())
# Print columns to confirm structure
print("Columns in dataset:", dataset.columns)
# Encode categorical variables
categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
for column in categorical_columns:
    dataset[column] = pd.factorize(dataset[column])[0]
# Define features and scale them
X = dataset[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply KMeans clustering to segment cars into price categories
kmeans = KMeans(n_clusters=3, random_state=42)
dataset['cluster'] = kmeans.fit_predict(X_scaled)
# Map cluster labels to car categories
dataset['category'] = dataset['cluster'].map({
    0: 'Budget-Friendly',
    1: 'Mid-Range',
    2: 'Luxury'
})
# Plot distribution of car prices by category
plt.figure(figsize=(10, 6))
sns.histplot(data=dataset, x='selling_price', hue='category', kde=True, palette='viridis')
plt.title("Car Price Distribution by Category")
plt.xlabel("Selling Price")
plt.ylabel("Number of Cars")
plt.legend(title='Category')
plt.show()
# Plot fuel type distribution for each category
plt.figure(figsize=(10, 6))
sns.countplot(data=dataset, x='fuel', hue='category', palette='viridis')
plt.title("Fuel Type Distribution by Car Category")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.legend(title='Category')
plt.show()
# Function to recommend cars based on budget and other preferences
def recommend_cars(budget, fuel_type=None, transmission=None):
    recommendations = dataset[(dataset['selling_price'] <= budget)]
    if fuel_type is not None:
        recommendations = recommendations[recommendations['fuel'] == fuel_type]
    if transmission is not None:
        recommendations = recommendations[recommendations['transmission'] == transmission]
    return recommendations[['year', 'km_driven', 'selling_price', 'category']]
# Example recommendations
print(recommend_cars(budget=500000, fuel_type=1, transmission=1).head())
```
### Output:


### Result:
Thus the system was trained successfully and the prediction was carried out.
