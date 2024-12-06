"""
Feature Engineering Module
Provides tools for creating and transforming features to enhance dataset
utility for machine learning tasks.
"""

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

class FeatureEngineer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the FeatureEngineer with a dataset.
        Args:   data (pd.DataFrame):    Input dataset for feature engineering.
        """

        self.data = data

    def add_polynomial_features(self, columns=None, degree=2):
        """
        Add polynomial features for specified columns.
        Args:
            columns (list):     Columns to transform. If None, all numeric columns are used.
            degree (int):       Degree of polynomial features.
        Returns: pd.DataFrame:  Dataset with added polynomial features.
        """

        columns = columns or self.data.select_dtypes(include=["number"]).columns
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(self.data[columns])
        poly_feature_names = poly.get_feature_names_out(columns)

        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=self.data.index)
        self.data = pd.concat([self.data, poly_df], axis=1)
        return self.data

    def interaction_terms(self, columns=None):
        """
        Add interaction terms for specified columns.
        Args:   columns (list): Columns to generate interaction terms. If None, all numeric columns are used.
        Returns: pd.DataFrame:  Dataset with interaction terms.
        """

        columns = columns or self.data.select_dtypes(include=["number"]).columns
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        interaction_features = poly.fit_transform(self.data[columns])
        interaction_feature_names = poly.get_feature_names_out(columns)

        interaction_df = pd.DataFrame(interaction_features, columns=interaction_feature_names, index=self.data.index)
        self.data = pd.concat([self.data, interaction_df], axis=1)
        return self.data

    def one_hot_encode(self, columns=None, drop_first=True):
        """
        One-hot encode categorical variables.
        Args:
            columns (list):     Columns to encode. If None, all categorical columns are used.
            drop_first (bool):  Whether to drop the first level of categories.

        Returns: pd.DataFrame:  Dataset with one-hot encoded variables.
        """

        columns = columns or self.data.select_dtypes(include=["object", "category"]).columns
        self.data = pd.get_dummies(self.data, columns=columns, drop_first=drop_first)
        return self.data

    def scale_features(self, columns=None, method="standard"):
        """
        Scale numeric features using specified method.
        Args:
            columns (list):     Columns to scale. If None, all numeric columns are used.
            method (str):       Scaling method ("standard" or "minmax").
        Returns: pd.DataFrame:  Dataset with scaled features.
        """

        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        columns = columns or self.data.select_dtypes(include=["number"]).columns
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self.data
