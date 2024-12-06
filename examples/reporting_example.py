"""
Example: Using DataReporter for Dataset Reporting
"""

import pandas as pd
import numpy as np
from data_inertia.reporting import DataReporter

# Generate a sample dataset
np.random.seed(42)
data = pd.DataFrame({
    "Numeric1": np.random.randn(100),  # Random numeric data
    "Numeric2": np.random.randint(1, 100, 100),  # Integer data
    "Sparse": np.where(np.random.rand(100) > 0.8, None, np.random.randint(1, 100, 100)),  # Sparse numeric data
})

print("Original Dataset:")
print(data.head())

# Initialize the DataReporter
reporter = DataReporter(data)

# Step 1: Generate a PDF summary report
summary_file = "summary_report.pdf"
reporter.generate_summary_report(filepath=summary_file)
print(f"PDF Summary Report saved to {summary_file}")

# Step 2: Create a heatmap for missing values
heatmap_file = "missing_heatmap.png"
reporter.plot_missing_values(output_path=heatmap_file)
print(f"Missing Values Heatmap saved to {heatmap_file}")

# Step 3: Generate a diagnostics file
diagnostics_file = "diagnostics.txt"
reporter.generate_diagnostics(output_path=diagnostics_file)
print(f"Diagnostics File saved to {diagnostics_file}")

# Print the content of the diagnostics file
with open(diagnostics_file, "r") as file:
    print("\nDiagnostics File Content:")
    print(file.read())
