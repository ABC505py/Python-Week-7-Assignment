# Python-Week-7-Assignment

# iris_analysis.py

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

try:
    # Load the Iris dataset from sklearn
    iris = load_iris()

    # Convert to a pandas DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Add the target (species) as a new column
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Display the first few rows of the dataset
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check the structure and data types
    print("\nData types and non-null counts:")
    print(df.info())

    # Check for missing values
    print("\nMissing values in dataset:")
    print(df.isnull().sum())

    # Clean the dataset if needed (no missing values in this dataset)
    # For demonstration: drop rows with missing values (if any)
    df.dropna(inplace=True)

except Exception as e:
    print(f"Error loading or processing dataset: {e}")

# Task 2: Basic Data Analysis

# Compute basic statistics
print("\nBasic statistics for numerical columns:")
print(df.describe())

# Group by species and compute mean of features
print("\nAverage measurements by species:")
grouped_means = df.groupby('species').mean()
print(grouped_means)

# Identify patterns
print("\nFindings:")
print("-> Setosa has the smallest petal and sepal sizes.\n"
      "-> Virginica generally has the largest values across most measurements.")

# Task 3: Data Visualization

# Set Seaborn style
sns.set(style="whitegrid")

# Line Chart: Simulated time-series trend for sepal length
df_time = df.copy()
df_time['index'] = df_time.index  # Simulated time
plt.figure(figsize=(10, 5))
sns.lineplot(x='index', y='sepal length (cm)', data=df_time)
plt.title("Line Chart: Sepal Length Over Sample Index")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.tight_layout()
plt.show()

# Bar Chart: Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title("Bar Chart: Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram: Distribution of Sepal Width
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True, color='skyblue')
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter Plot: Sepal Length vs. Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Scatter Plot: Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.tight_layout()
plt.show()
