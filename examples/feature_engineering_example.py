"""
Example: Using FeatureEngineer for Feature Engineering
"""

import pandas as pd
import numpy as np
from data_inertia.feature_engineering import FeatureEngineer

# Generate a sample dataset
np.random.seed(42)
data = pd.DataFrame({
    "Numeric1": np.random.randn(100),  # Random numeric data
    "Numeric2": np.random.randint(1, 100, 100),  # Integer data
    "Category": np.random.choice(["cat", "dog", "mouse"], 100),  # Categorical data
})

print("Original Dataset:")
print(data.head())

# Initialize the FeatureEngineer
engineer = FeatureEngineer(data)

# Step 1: Add polynomial features
data_with_poly = engineer.add_polynomial_features(columns=["Numeric1"], degree=2)
print("\nAfter Adding Polynomial Features:")
print(data_with_poly.head())

# Step 2: Add interaction terms
data_with_interactions = engineer.interaction_terms(columns=["Numeric1", "Numeric2"])
print("\nAfter Adding Interaction Terms:")
print(data_with_interactions.head())

# Step 3: One-hot encode categorical columns
data_encoded = engineer.one_hot_encode(columns=["Category"], drop_first=False)
print("\nAfter One-Hot Encoding Categorical Columns:")
print(data_encoded.head())

# Step 4: Scale numeric features (Standard Scaling)
data_scaled = engineer.scale_features(columns=["Numeric1", "Numeric2"], method="standard")
print("\nAfter Standard Scaling Numeric Features:")
print(data_scaled.head())

# Step 5: Scale numeric features (Min-Max Scaling)
data_minmax_scaled = engineer.scale_features(columns=["Numeric1", "Numeric2"], method="minmax")
print("\nAfter Min-Max Scaling Numeric Features:")
print(data_minmax_scaled.head())
