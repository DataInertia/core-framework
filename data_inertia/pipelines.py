"""
Pipelines Module

Provides preprocessing and modeling pipelines.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataPipeline:
    @staticmethod
    def create_pipeline(
        numeric_columns=None,
        categorical_columns=None,
        impute_strategy="mean",
        scale_method="standard",
        encode_method="one-hot"
    ):
        """Create a preprocessing pipeline for numeric and categorical data."""
        # Numeric processing: Imputation and scaling
        numeric_transform = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=impute_strategy)),
            ("scaler", StandardScaler() if scale_method == "standard" else MinMaxScaler())
        ])

        # Categorical processing: Imputation and encoding
        categorical_transform = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore") if encode_method == "one-hot" else "passthrough")
        ])

        # Combine preprocessing for numeric and categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transform, numeric_columns),
                ("cat", categorical_transform, categorical_columns)
            ]
        )

        return preprocessor

    @staticmethod
    def full_pipeline(preprocessor, estimator):
        """Combine preprocessor and estimator into a full pipeline."""
        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
