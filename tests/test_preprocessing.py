import unittest
import pandas as pd
import numpy as np
from data_inertia.preprocessing import DataPreprocessor

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        """Set up a larger dataset with mixed data types."""
        np.random.seed(42)
        n_rows = 1000
        self.data = pd.DataFrame({
            "A": np.random.randn(n_rows),  # Numeric column
            "B": np.random.randint(1, 100, n_rows),  # Integer column
            "C": np.random.choice(["cat", "dog", "mouse"], n_rows),  # Categorical column
            "D": np.where(np.random.rand(n_rows) > 0.9, None, np.random.randint(1, 100, n_rows))  # Sparsely missing column
        })
        self.preprocessor = DataPreprocessor(self.data)

    def test_normalize(self):
        """Test normalization of numeric columns with a large dataset."""
        normalized = self.preprocessor.normalize(columns=["A", "B"])
        self.assertTrue((normalized["A"] >= 0).all() and (normalized["A"] <= 1).all())
        self.assertTrue((normalized["B"] >= 0).all() and (normalized["B"] <= 1).all())

    def test_standardize(self):
        """Test standardization of numeric columns with a large dataset."""
        standardized = self.preprocessor.standardize(columns=["A", "B"])
        self.assertAlmostEqual(standardized["A"].mean(), 0, places=1)
        self.assertAlmostEqual(standardized["A"].std(), 1, places=1)
        self.assertAlmostEqual(standardized["B"].mean(), 0, places=1)
        self.assertAlmostEqual(standardized["B"].std(), 1, places=1)

    def test_encode_categorical_one_hot(self):
        """Test one-hot encoding on larger categorical data."""
        encoded = self.preprocessor.encode_categorical(encoding_type="one-hot", columns=["C"])
        self.assertIn("C_cat", encoded.columns)
        self.assertIn("C_dog", encoded.columns)
        self.assertIn("C_mouse", encoded.columns)

    def test_encode_categorical_label(self):
        """Test label encoding on larger categorical data."""
        encoded = self.preprocessor.encode_categorical(encoding_type="label", columns=["C"])
        self.assertTrue(pd.api.types.is_integer_dtype(encoded["C"]))
        unique_labels = encoded["C"].nunique()
        self.assertEqual(unique_labels, len(self.data["C"].unique()))

if __name__ == "__main__":
    unittest.main()