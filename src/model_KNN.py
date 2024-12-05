import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, \
    precision_score, recall_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import os


# Suppress warnings
warnings.filterwarnings("ignore")

# import load_split_preprocess helper function to load raw data, split it, then preprocess it
from load_split_preprocess import load_and_preprocess_data
X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("../data/processed/heart_cleaned_data.csv")

# Create a KNN classifier
knn = KNeighborsClassifier()

# Create a pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", knn)])

# Define parameter grid for KNN classifier
param_grid = {
    'classifier__n_neighbors': np.arange(2, 50, 2),  # Varying n_neighbors
    'classifier__weights': ['uniform', 'distance'],  # Varying the weights
    'classifier__metric': ['euclidean', 'manhattan'],  # Different distance metrics
}

# Grid search for best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', return_train_score=True)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best Parameters Found:")
print(grid_search.best_params_)

# Best model based on grid search
best_knn_model = grid_search.best_estimator_

# Test predictions
y_pred = best_knn_model.predict(X_test)

# Test accuracy on the best model
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Calculate AUC for the best model
y_pred_prob = best_knn_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Score: {auc_score:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

# import the helper function
from plot_model_evaluation import plot_model_evaluation 

plot_model_evaluation(
    model_name="KNN",
    y_test=y_test,
    y_pred=y_pred,
    y_pred_prob=y_pred_prob,
    test_accuracy=test_accuracy,
    precision=precision,
    recall=recall
)

# Save the best model using pickle
with open('../models/best_knn_model.pkl', 'wb') as file:
    joblib.dump(best_knn_model, file)
    print("Best model saved as 'best_knn_model.pkl'")