"""
Data Profiling Module

Provides functionality for profiling datasets to generate insights about
data quality, structure, and statistical properties.
"""

import pandas as pd

class DataProfiler:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataProfiler with a dataset.
        data (pd.DataFrame):    Input dataset to profile.
        """

        self.data = data

    def basic_summary(self):
        """
        Generate a basic summary of the dataset.
        Returns:    dict:   Summary containing column names, data types, missing values, and shape.
        """

        summary = {
            "columns": self.data.columns.tolist(),
            "data_types": self.data.dtypes.astype(str).to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "total_rows": self.data.shape[0],
            "total_columns": self.data.shape[1],
        }
        return summary

    def missing_value_report(self, threshold=0.1):
        """
        Generate a report on missing values.
        threshold (float):  Minimum proportion of missing values to flag a column.
        Returns:    dict:   Report with counts and proportions of missing values per column.
        """

        total_rows = len(self.data)
        missing_counts = self.data.isnull().sum()
        missing_percentage = missing_counts / total_rows

        flagged_columns = missing_percentage[missing_percentage > threshold].index.tolist()

        return {
            "missing_counts": missing_counts.to_dict(),
            "missing_percentage": missing_percentage.round(2).to_dict(),
            "flagged_columns": flagged_columns,
        }

    def correlation_matrix(self):
        """
        Generate a correlation matrix for numeric columns.
        Returns:    pd.DataFrame:   Correlation matrix.
        """

        numeric_data = self.data.select_dtypes(include=["number"])
        if numeric_data.empty:
            raise ValueError("No numeric columns found for correlation analysis.")
        return numeric_data.corr()

    def outlier_report(self, method="iqr"):
        """
        Identify outliers in numeric columns.
        Args:   method (str):   Method to detect outliers ("iqr" or "zscore").
        Returns:    dict:   Outliers identified for each column.
        """
        
        numeric_data = self.data.select_dtypes(include=["number"])
        outliers = {}

        if method == "iqr":
            for col in numeric_data.columns:
                q1 = numeric_data[col].quantile(0.25)
                q3 = numeric_data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers[col] = numeric_data[(numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)][col].tolist()
        elif method == "zscore":
            from scipy.stats import zscore
            z_scores = numeric_data.apply(zscore)
            for col in z_scores.columns:
                outliers[col] = numeric_data[abs(z_scores[col]) > 3][col].tolist()
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")

        return outliers