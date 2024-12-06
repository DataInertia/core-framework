"""
Feature Engineering: Add polynomials, interactions, encodings, and scaling.
"""
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler

class FeatureEngineer:
    def __init__(self, data: pd.DataFrame):
        """Initialize with dataset."""
        self.data = data

    def add_polynomial_features(self, columns=None, degree=2):
        """Add polynomial features for specified columns."""
        columns = columns or self.data.select_dtypes(include=["number"]).columns
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(self.data[columns])
        poly_names = poly.get_feature_names_out(columns)
        self.data = pd.concat(
            [self.data, pd.DataFrame(poly_features[:, len(columns):], columns=poly_names[len(columns):], index=self.data.index)],
            axis=1,
        )
        return self.data

    def interaction_terms(self, columns=None):
        """Add interaction terms between specified columns."""
        columns = columns or self.data.select_dtypes(include=["number"]).columns
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        interaction_features = poly.fit_transform(self.data[columns])
        interaction_names = poly.get_feature_names_out(columns)
        self.data = pd.concat(
            [self.data, pd.DataFrame(interaction_features, columns=interaction_names, index=self.data.index)],
            axis=1,
        )
        return self.data

    def one_hot_encode(self, columns=None, drop_first=True):
        """One-hot encode categorical columns."""
        columns = columns or self.data.select_dtypes(include=["object", "category"]).columns
        self.data = pd.get_dummies(self.data, columns=columns, drop_first=drop_first)
        return self.data

    def scale_features(self, columns=None, method="standard"):
        """Scale numeric features using standard or Min-Max scaling."""
        columns = columns or self.data.select_dtypes(include=["number"]).columns
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self.data
