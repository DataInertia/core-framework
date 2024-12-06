import unittest
import pandas as pd
import numpy as np
from data_inertia.cleaning import DataCleaner

class TestCleaning(unittest.TestCase):

    def setUp(self):
        """Set up a large dataset with mixed data types."""
        np.random.seed(42)
        n_rows = 1000
        self.data = pd.DataFrame({
            "Numeric1": np.random.randn(n_rows),  # Numeric data
            "Numeric2": np.random.randint(1, 100, n_rows),  # Integer data
            "Category": np.random.choice(["cat", "dog", "mouse"], n_rows),  # Categorical data
            "Sparse": np.where(np.random.rand(n_rows) > 0.8, None, np.random.randint(1, 100, n_rows)),  # Sparse data
        })
        # Create duplicate rows by appending part of the dataset
        duplicates = self.data.iloc[:500].copy()
        self.data = pd.concat([self.data, duplicates], ignore_index=True)
        self.cleaner = DataCleaner(self.data)

    def test_impute_missing_mean(self):
        """Test mean imputation for numeric columns."""
        imputed = self.cleaner.impute_missing(strategy="mean", columns=["Sparse"])
        self.assertFalse(imputed["Sparse"].isnull().any())
        self.assertAlmostEqual(imputed["Sparse"].mean(), self.data["Sparse"].dropna().mean(), places=1)

    def test_impute_missing_knn(self):
        """Test KNN imputation for numeric columns."""
        imputed = self.cleaner.impute_missing(strategy="knn", columns=["Sparse"], k_neighbors=5)
        self.assertFalse(imputed["Sparse"].isnull().any())

    def test_remove_outliers_iqr(self):
        """Test outlier removal using IQR."""
        cleaned = self.cleaner.remove_outliers(method="iqr", columns=["Numeric1"])
        original_size = len(self.data)
        cleaned_size = len(cleaned)
        self.assertLess(cleaned_size, original_size)

    def test_remove_outliers_zscore(self):
        """Test outlier removal using Z-score."""
        cleaned = self.cleaner.remove_outliers(method="zscore", columns=["Numeric1"])
        original_size = len(self.data)
        cleaned_size = len(cleaned)
        self.assertLess(cleaned_size, original_size)

    def test_drop_duplicates(self):
        """Test duplicate removal."""
        cleaned = self.cleaner.drop_duplicates(subset=["Numeric1", "Numeric2", "Category", "Sparse"])
        self.assertEqual(len(cleaned), 1000)  # Expect 1000 unique rows

if __name__ == "__main__":
    unittest.main()
