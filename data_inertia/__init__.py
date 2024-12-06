"""
DataInertia - Core Framework for Data Optimization

This package provides modules and utilities for preprocessing, profiling,
cleaning, feature engineering, and reporting, aimed at optimizing datasets
for machine learning accuracy.
"""

# Importing core modules
from .preprocessing import DataPreprocessor
from .profiling import DataProfiler
from .cleaning import DataCleaner
from .feature_engineering import FeatureEngineer
from .reporting import DataReporter
from .pipelines import DataPipeline

__all__ = [
    "DataPreprocessor",
    "DataProfiler",
    "DataCleaner",
    "FeatureEngineer",
    "DataReporter",
    "DataPipeline",
]

__version__ = "0.1.0"