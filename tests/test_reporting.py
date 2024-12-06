import unittest
import pandas as pd
import numpy as np
import os
from data_inertia.reporting import DataReporter

class TestReporting(unittest.TestCase):
    def setUp(self):
        """Set up a dataset with numeric and sparse data."""
        np.random.seed(42)
        n_rows = 100
        self.data = pd.DataFrame({
            "Numeric1": np.random.randn(n_rows),
            "Numeric2": np.random.randint(1, 100, n_rows),
            "Sparse": np.where(np.random.rand(n_rows) > 0.8, None, np.random.randint(1, 100, n_rows)),
        })
        self.reporter = DataReporter(self.data)

    def test_generate_summary_report(self):
        """Test PDF summary report generation."""
        filepath = "summary_report.pdf"
        self.reporter.generate_summary_report(filepath=filepath)
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

    def test_plot_missing_values(self):
        """Test missing values heatmap generation."""
        output_path = "missing_heatmap.png"
        self.reporter.plot_missing_values(output_path=output_path)
        self.assertTrue(os.path.exists(output_path))
        os.remove(output_path)

    def test_generate_diagnostics(self):
        """Test diagnostics file generation."""
        output_path = "diagnostics.txt"
        self.reporter.generate_diagnostics(output_path=output_path)
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "r") as f:
            diagnostics = f.read()
        self.assertIn("Total Rows", diagnostics)
        os.remove(output_path)

if __name__ == "__main__":
    unittest.main()
