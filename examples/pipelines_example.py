"""
Example: Using DataPipeline for Preprocessing and Model Training
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data_inertia.pipelines import DataPipeline

# Generate a sample dataset
np.random.seed(42)
data = pd.DataFrame({
    "Numeric1": np.random.randn(100),
    "Numeric2": np.random.randint(1, 100, 100),
    "Category": np.random.choice(["cat", "dog", "mouse"], 100),
    "Target": np.random.choice([0, 1], 100)  # Binary target
})

print("Original Dataset:")
print(data.head())

# Split dataset into features (X) and target (y)
X = data.drop(columns=["Target"])
y = data["Target"]

# Step 1: Create a preprocessing pipeline
preprocessor = DataPipeline.create_pipeline(
    numeric_columns=["Numeric1", "Numeric2"],
    categorical_columns=["Category"],
    impute_strategy="mean",
    scale_method="standard",
    encode_method="one-hot"
)

# Step 2: Combine preprocessing with a RandomForest model
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_pipeline = DataPipeline.full_pipeline(preprocessor, rf_model)

# Train and predict using RandomForest pipeline
rf_pipeline.fit(X, y)
rf_predictions = rf_pipeline.predict(X)
print("\nRandomForest Predictions:")
print(rf_predictions)

# Step 3: Combine preprocessing with a LogisticRegression model
lr_model = LogisticRegression(max_iter=100, random_state=42)
lr_pipeline = DataPipeline.full_pipeline(preprocessor, lr_model)

# Train and evaluate using LogisticRegression pipeline
lr_pipeline.fit(X, y)
lr_score = lr_pipeline.score(X, y)
print("\nLogistic Regression Model Accuracy:")
print(lr_score)
