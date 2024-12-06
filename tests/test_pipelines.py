import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from data_inertia.pipelines import DataPipeline

class TestPipelines(unittest.TestCase):

    def setUp(self):
        """Set up a dataset with numeric and categorical data."""
        np.random.seed(42)
        n_rows = 100
        self.data = pd.DataFrame({
            "Numeric1": np.random.randn(n_rows),
            "Numeric2": np.random.randint(1, 100, n_rows),
            "Category": np.random.choice(["cat", "dog", "mouse"], n_rows),
            "Target": np.random.choice([0, 1], n_rows)  # Binary target
        })
        self.numeric_columns = ["Numeric1", "Numeric2"]
        self.categorical_columns = ["Category"]

    def test_create_pipeline(self):
        """Test preprocessing pipeline creation."""
        pipeline = DataPipeline.create_pipeline(
            numeric_columns=self.numeric_columns,
            categorical_columns=self.categorical_columns,
            impute_strategy="mean",
            scale_method="standard",
            encode_method="one-hot"
        )
        self.assertIsInstance(pipeline, ColumnTransformer)
        self.assertIn("num", [name for name, _, _ in pipeline.transformers])
        self.assertIn("cat", [name for name, _, _ in pipeline.transformers])

    def test_full_pipeline_with_model(self):
        """Test full pipeline integration with a machine learning model."""
        preprocessor = DataPipeline.create_pipeline(
            numeric_columns=self.numeric_columns,
            categorical_columns=self.categorical_columns,
            impute_strategy="mean",
            scale_method="minmax",
            encode_method="one-hot"
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        full_pipeline = DataPipeline.full_pipeline(preprocessor, model)

        # Train the pipeline
        X = self.data.drop(columns=["Target"])
        y = self.data["Target"]
        full_pipeline.fit(X, y)

        # Predict
        predictions = full_pipeline.predict(X)
        self.assertEqual(len(predictions), len(y))

    def test_full_pipeline_with_logistic_regression(self):
        """Test full pipeline with Logistic Regression."""
        preprocessor = DataPipeline.create_pipeline(
            numeric_columns=self.numeric_columns,
            categorical_columns=self.categorical_columns,
            impute_strategy="most_frequent",
            scale_method="standard",
            encode_method="one-hot"  # Ensure categorical variables are encoded
        )
        model = LogisticRegression(max_iter=100, random_state=42)
        full_pipeline = DataPipeline.full_pipeline(preprocessor, model)

        # Train the pipeline
        X = self.data.drop(columns=["Target"])
        y = self.data["Target"]
        full_pipeline.fit(X, y)  # Should now work

        # Evaluate pipeline
        score = full_pipeline.score(X, y)
        self.assertGreaterEqual(score, 0.5)  # Model should perform at least as well as random guessing

if __name__ == "__main__":
    unittest.main()
