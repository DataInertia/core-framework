"""
Example: Using DataCleaner for Dataset Cleaning
"""

import pandas as pd
import numpy as np
from data_inertia.cleaning import DataCleaner

# Generate a sample dataset
np.random.seed(42)
data = pd.DataFrame({
    "Numeric1": np.random.randn(100),  # Random numeric data
    "Numeric2": np.random.randint(1, 100, 100),  # Integer data
    "Category": np.random.choice(["cat", "dog", "mouse"], 100),  # Categorical data
    "Sparse": np.where(np.random.rand(100) > 0.8, None, np.random.randint(1, 100, 100)),  # Sparse numeric data
})

# Create duplicate rows
duplicates = data.iloc[:20].copy()
data = pd.concat([data, duplicates], ignore_index=True)

print("Original Dataset:")
print(data.head(10))

# Initialize the DataCleaner
cleaner = DataCleaner(data)

# Step 1: Impute missing values
cleaned_data = cleaner.impute_missing(strategy="mean", columns=["Sparse"])
print("\nAfter Imputing Missing Values (Mean):")
print(cleaned_data.head(10))

# Step 2: Remove outliers using IQR
cleaned_data = cleaner.remove_outliers(method="iqr", columns=["Numeric1"])
print("\nAfter Removing Outliers (IQR Method):")
print(cleaned_data.head(10))

# Step 3: Remove duplicate rows
cleaned_data = cleaner.drop_duplicates(subset=["Numeric1", "Numeric2", "Category", "Sparse"])
print("\nAfter Removing Duplicates:")
print(cleaned_data.head(10))

# Final Dataset Summary
print("\nFinal Dataset Shape:", cleaned_data.shape)
