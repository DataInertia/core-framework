import unittest
import pandas as pd
import numpy as np
from data_inertia.profiling import DataProfiler

class TestProfiling(unittest.TestCase):

    def setUp(self):
        """Set up a larger dataset with mixed data types."""
        np.random.seed(42)
        n_rows = 1000
        self.data = pd.DataFrame({
            "Numeric1": np.random.randn(n_rows),  # Numeric column
            "Numeric2": np.random.randint(1, 100, n_rows),  # Integer column
            "Category": np.random.choice(["cat", "dog", "mouse"], n_rows),  # Categorical column
            "Sparse": np.where(np.random.rand(n_rows) > 0.8, None, np.random.randint(1, 100, n_rows))  # Sparsely missing column
        })
        self.profiler = DataProfiler(self.data)

    def test_basic_summary(self):
        """Test basic dataset summary."""
        summary = self.profiler.basic_summary()
        self.assertEqual(summary["total_rows"], 1000)
        self.assertEqual(summary["total_columns"], 4)
        self.assertIn("Numeric1", summary["columns"])
        self.assertEqual(summary["data_types"]["Numeric1"], "float64")
        self.assertTrue(summary["missing_values"]["Sparse"] > 0)

    def test_missing_value_report(self):
        """Test missing value report."""
        report = self.profiler.missing_value_report(threshold=0.1)
        self.assertTrue("Sparse" in report["flagged_columns"])
        self.assertGreaterEqual(report["missing_counts"]["Sparse"], 0)
        self.assertGreaterEqual(report["missing_percentage"]["Sparse"], 0)

    def test_correlation_matrix(self):
        """Test correlation matrix generation."""
        correlation_matrix = self.profiler.correlation_matrix()
        self.assertTrue("Numeric1" in correlation_matrix.columns)
        self.assertTrue("Numeric2" in correlation_matrix.columns)
        self.assertAlmostEqual(correlation_matrix.loc["Numeric1", "Numeric1"], 1.0, places=5)

    def test_outlier_report(self):
        """Test outlier detection using IQR."""
        outliers = self.profiler.outlier_report(method="iqr")
        self.assertIn("Numeric1", outliers)
        self.assertIsInstance(outliers["Numeric1"], list)

if __name__ == "__main__":
    unittest.main()
