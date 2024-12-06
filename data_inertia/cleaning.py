"""
Data Cleaning Module

Provides functionality for cleaning datasets, including handling missing values,
removing outliers, and resolving data inconsistencies.
"""

import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        """Initialize with dataset."""
        self.data = data

    def impute_missing(self, strategy="mean", columns=None, k_neighbors=5, constant_value=None):
        """Impute missing values using specified strategy."""
        columns = columns or self.data.columns
        if strategy in ["mean", "median", "most_frequent", "constant"]:
            imputer = SimpleImputer(strategy=strategy, fill_value=constant_value)
        elif strategy == "knn":
            imputer = KNNImputer(n_neighbors=k_neighbors)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        self.data[columns] = imputer.fit_transform(self.data[columns])
        return self.data

    def remove_outliers(self, method="iqr", columns=None):
        """Remove outliers using IQR or Z-score."""
        columns = columns or self.data.select_dtypes(include=["number"]).columns
        if method == "iqr":
            for col in columns:
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
        elif method == "zscore":
            from scipy.stats import zscore
            z_scores = self.data[columns].apply(zscore)
            self.data = self.data[(z_scores.abs() <= 3).all(axis=1)]
        else:
            raise ValueError(f"Unsupported outlier removal method: {method}")
        return self.data

    def drop_duplicates(self, subset=None):
        """Drop duplicate rows based on specified columns."""
        self.data = self.data.drop_duplicates(subset=subset)
        return self.data
