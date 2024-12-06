"""
Pipelines for Preprocessing and Modeling
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataPipeline:
    @staticmethod
    def create_pipeline(numeric_columns=None, categorical_columns=None, impute="mean", scale="standard", encode="one-hot"):
        # Create a preprocessing pipeline for numeric and categorical data.
        num_transform = Pipeline([
            ("imputer", SimpleImputer(strategy=impute)),
            ("scaler", StandardScaler() if scale == "standard" else MinMaxScaler())
        ])
        cat_transform = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore") if encode == "one-hot" else "passthrough")
        ])
        return ColumnTransformer([("num", num_transform, numeric_columns), ("cat", cat_transform, categorical_columns)])

    @staticmethod
    def full_pipeline(preprocessor, estimator):
        # Combine preprocessor and estimator into a full pipeline.
        return Pipeline([("preprocessor", preprocessor), ("model", estimator)])