import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score,\
    recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import seaborn as sns    
    
    
    
# Suppress warnings
warnings.filterwarnings("ignore")

# import load_split_preprocess helper function to load raw data, split it, then preprocess it
from load_split_preprocess import load_and_preprocess_data
X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("../data/processed/heart_cleaned_data.csv")
    
# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=67)

# Create a pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf_classifier)])

# Define parameter grid for Random Forest
param_grid = {
    'classifier__n_estimators': [50, 100, 200],  # Number of trees
    'classifier__max_depth': [3],  # Maximum depth of trees
    'classifier__min_samples_split': [20],  # Minimum samples required to split a node
    'classifier__min_samples_leaf': [20],    # Minimum samples required at leaf node
}

# Grid search for best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', return_train_score=True)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best Parameters Found:")
print(grid_search.best_params_)

# Best model based on grid search
best_rf_model = grid_search.best_estimator_


y_pred = best_rf_model.predict(X_test)

# Test accuracy on the best model
train_accuracy = accuracy_score(y_train, best_rf_model.predict(X_train))
test_accuracy = accuracy_score(y_test, best_rf_model.predict(X_test))
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Perform cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='accuracy')

print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")
print(f"Training Accuracy: {train_accuracy:.4f}")

# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


# Calculate AUC for the best model
y_pred_prob = best_rf_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Score_test: {auc_score:.4f}")

y_pred_prob_train = best_rf_model.predict_proba(X_train)[:, 1]
auc_score_train = roc_auc_score(y_train, y_pred_prob_train)
print(f"AUC Score_train: {auc_score:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

# import the plot_model_evaluation helper function
from plot_model_evaluation import plot_model_evaluation 
# Plot confusion matrix, AUC_ROC, (accuracy, recall, and precision) for the best model
plot_model_evaluation(
    model_name="Random Forest",
    y_test=y_test,
    y_pred=y_pred,
    y_pred_prob=y_pred_prob,
    test_accuracy=test_accuracy,
    precision=precision,
    recall=recall
)

# Save the best model using pickle
with open('../models/best_rf_model.pkl', 'wb') as file:
    joblib.dump(best_rf_model, file)
    print("Best model saved as 'best_rf_model.pkl'")
    
  