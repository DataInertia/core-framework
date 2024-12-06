"""
Data Reporting Module
"""

import pandas as pd
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import seaborn as sns

class DataReporter:
    def __init__(self, data: pd.DataFrame):
        """Initialize with dataset."""
        self.data = data

    def generate_summary_report(self, filepath="summary_report.pdf"):
        """Create a PDF summarizing dataset details."""
        c = canvas.Canvas(filepath)
        c.drawString(100, 750, "DataInertia - Summary Report")
        c.drawString(100, 730, f"Total Rows: {len(self.data)}")
        c.drawString(100, 710, f"Total Columns: {len(self.data.columns)}")
        c.drawString(100, 690, f"Missing Values: {self.data.isnull().sum().sum()}")
        c.save()
        print(f"Summary report saved to {filepath}")

    def plot_missing_values(self, output_path=None):
        """Visualize missing value patterns."""
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        if output_path:
            plt.savefig(output_path)
            print(f"Heatmap saved to {output_path}")
        else:
            plt.show()

    def generate_diagnostics(self, output_path="diagnostics.txt"):
        """Save dataset diagnostics to a file."""
        with open(output_path, "w") as f:
            f.write("DataInertia - Diagnostics Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total Rows: {len(self.data)}\n")
            f.write(f"Total Columns: {len(self.data.columns)}\n")
            f.write(f"Missing Values: {self.data.isnull().sum().to_dict()}\n")
            f.write("=" * 40 + "\n")
        print(f"Diagnostics report saved to {output_path}")
