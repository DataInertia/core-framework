import unittest
import pandas as pd
import numpy as np
from data_inertia.feature_engineering import FeatureEngineer

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        """Set up a dataset with numeric and categorical data."""
        np.random.seed(42)
        n_rows = 1000
        self.data = pd.DataFrame({
            "Numeric1": np.random.randn(n_rows),
            "Numeric2": np.random.randint(1, 100, n_rows),
            "Category": np.random.choice(["cat", "dog", "mouse"], n_rows),
        })
        self.engineer = FeatureEngineer(self.data)

    def test_add_polynomial_features(self):
        """Test polynomial feature addition."""
        transformed = self.engineer.add_polynomial_features(columns=["Numeric1"], degree=2)
        self.assertTrue("Numeric1^2" in transformed.columns)
        self.assertEqual(len(transformed.columns), self.data.shape[1] + 1)

    def test_interaction_terms(self):
        """Test interaction terms between columns."""
        transformed = self.engineer.interaction_terms(columns=["Numeric1", "Numeric2"])
        self.assertGreater(len(transformed.columns), self.data.shape[1])

    def test_one_hot_encode(self):
        """Test one-hot encoding of categorical data."""
        encoded = self.engineer.one_hot_encode(columns=["Category"], drop_first=False)
        self.assertIn("Category_cat", encoded.columns)
        self.assertIn("Category_dog", encoded.columns)
        self.assertIn("Category_mouse", encoded.columns)

    def test_scale_features_standard(self):
        """Test standard scaling of numeric columns."""
        scaled = self.engineer.scale_features(columns=["Numeric1", "Numeric2"], method="standard")
        self.assertAlmostEqual(scaled["Numeric1"].mean(), 0, places=1)
        self.assertAlmostEqual(scaled["Numeric1"].std(), 1, places=1)

    def test_scale_features_minmax(self):
        """Test Min-Max scaling of numeric columns."""
        scaled = self.engineer.scale_features(columns=["Numeric1", "Numeric2"], method="minmax")
        self.assertTrue((scaled["Numeric1"] >= 0).all())
        self.assertTrue((scaled["Numeric1"] <= 1).all())

if __name__ == "__main__":
    unittest.main()
