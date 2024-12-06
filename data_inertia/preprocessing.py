"""
Data Preprocessing Module
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataPreprocessor with a dataset.
        data (pd.DataFrame):
        """

        self.data = data

    def normalize(self, columns: list = None):
        """
        Normalize specified columns using Min-Max scaling.
        columns (list):
        pd.DataFrame:
        """

        columns = columns or self.data.select_dtypes(include=['number']).columns.tolist()
        scaler = MinMaxScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self.data

    def standardize(self, columns: list = None):
        """
        Standardize columns using z-score scaling.
        columns (list) List of columns to standardize. If None, all numeric columns are used.
        pd.DataFrame: Dataset with standardized columns.
        """

        columns = columns or self.data.select_dtypes(include=['number']).columns.tolist()
        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self.data

    def encode_categorical(self, encoding_type: str = "one-hot", columns: list = None):
        """
        Encode categorical columns using one-hot or label encoding.
        encoding_type (str): Encoding method ("one-hot" or "label").
        columns (list): List of columns to encode. If None, all categorical columns are used.

        Returns:
            pd.DataFrame: Dataset with encoded columns.
        """

        columns = columns or self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        if encoding_type == "one-hot":
            return pd.get_dummies(self.data, columns=columns, drop_first=True)
        elif encoding_type == "label":
            for col in columns:
                self.data[col] = self.data[col].astype('category').cat.codes
            return self.data
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")