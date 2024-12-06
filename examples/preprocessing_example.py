"""
Example: Using DataPreprocessor for Dataset Transformation
"""

import pandas as pd
import numpy as np
from data_inertia.preprocessing import DataPreprocessor

# Generate a sample dataset
np.random.seed(42)
data = pd.DataFrame({
    "Numeric1": np.random.randn(100),  # Random numeric data
    "Numeric2": np.random.randint(1, 100, 100),  # Integer data
    "Category": np.random.choice(["cat", "dog", "mouse"], 100),  # Categorical data
    "Sparse": np.where(np.random.rand(100) > 0.8, None, np.random.randint(1, 100, 100))  # Sparse numeric data
})

# Print the raw dataset
print("Original Dataset:")
print(data.head())

# Initialize the preprocessor
preprocessor = DataPreprocessor(data)

# Normalize numeric columns
normalized_data = preprocessor.normalize(columns=["Numeric1", "Numeric2"])
print("\nNormalized Data:")
print(normalized_data.head())

# Standardize numeric columns
standardized_data = preprocessor.standardize(columns=["Numeric1", "Numeric2"])
print("\nStandardized Data:")
print(standardized_data.head())

# One-hot encode categorical columns
encoded_data = preprocessor.encode_categorical(encoding_type="one-hot", columns=["Category"])
print("\nOne-Hot Encoded Data:")
print(encoded_data.head())

# Label encode categorical columns
label_encoded_data = preprocessor.encode_categorical(encoding_type="label", columns=["Category"])
print("\nLabel Encoded Data:")
print(label_encoded_data.head())
