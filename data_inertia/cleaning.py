"""
Data Cleaning Module
Provides functionality for cleaning datasets, including handling missing values,
removing outliers, and resolving data inconsistencies.
"""

import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataCleaner with a dataset.
        Args:   data (pd.DataFrame):    Input dataset to clean.
        """

        self.data = data

    def impute_missing(self, strategy="mean", columns=None, k_neighbors=5, constant_value=None):
        """
        Impute missing values using various strategies.
        Args:
                strategy (str):         Imputation strategy ("mean", "median", "most_frequent", "constant", "knn").
                columns (list):         Columns to impute. If None, all columns are used.
                k_neighbors (int):      Number of neighbors for KNN imputation (if applicable).
                constant_value (any):   Value for "constant" strategy.
        Returns:pd.DataFrame:           Dataset with imputed values.
        """

        columns = columns or self.data.columns
        imputer = None

        if strategy in ["mean", "median", "most_frequent", "constant"]:
            imputer = SimpleImputer(strategy=strategy, fill_value=constant_value)
        elif strategy == "knn":
            imputer = KNNImputer(n_neighbors=k_neighbors)
        else:
            raise ValueError(f"Unsupported imputation strategy: {strategy}")

        self.data[columns] = imputer.fit_transform(self.data[columns])
        return self.data

    def remove_outliers(self, method="iqr", columns=None):
        """
        Remove outliers from numeric columns.
        Args:
                method (str):   Method to detect outliers ("iqr" or "zscore").
                columns (list): Columns to clean. If None, all numeric columns are used.

        Returns:    pd.DataFrame:   Dataset with outliers removed.
        """

        columns = columns or self.data.select_dtypes(include=["number"]).columns
        cleaned_data = self.data.copy()

        if method == "iqr":
            for col in columns:
                q1 = cleaned_data[col].quantile(0.25)
                q3 = cleaned_data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]
        elif method == "zscore":
            from scipy.stats import zscore
            z_scores = cleaned_data[columns].apply(zscore)
            cleaned_data = cleaned_data[(z_scores.abs() <= 3).all(axis=1)]
        else:
            raise ValueError(f"Unsupported outlier removal method: {method}")

        self.data = cleaned_data
        return self.data

    def drop_duplicates(self):
        """
        Drop duplicate rows from the dataset.
        Returns:    pd.DataFrame:   Dataset with duplicates removed.
        """
        
        self.data = self.data.drop_duplicates()
        return self.data
