"""
Example: Using DataProfiler for Dataset Diagnostics
"""

import pandas as pd
import numpy as np
from data_inertia.profiling import DataProfiler

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

# Initialize the profiler
profiler = DataProfiler(data)

# Basic summary
summary = profiler.basic_summary()
print("\nBasic Summary:")
print(summary)

# Missing value report
missing_report = profiler.missing_value_report(threshold=0.1)
print("\nMissing Value Report:")
print(missing_report)

# Correlation matrix
try:
    correlation_matrix = profiler.correlation_matrix()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
except ValueError as e:
    print(f"Correlation Analysis Error: {e}")

# Outlier detection
outliers = profiler.outlier_report(method="iqr")
print("\nOutlier Report (IQR Method):")
for column, outlier_list in outliers.items():
    print(f"{column}: {len(outlier_list)} outliers detected")
