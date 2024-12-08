## **This folder contains the following :**
# **Data Processing Helper Functions**

This folder contains reusable helper functions to streamline various data preprocessing and visualization tasks. These functions help improve efficiency, reduce code duplication, and maintain consistency across data projects.

---

## **Table of Contents**

1. [Installation](#installation)
2. [Models codes](#Models-codes) 
3. [Helper Functions](#helper-functions)
   - [Preprocess Data](#1-preprocess-data)
   - [Save DataFrame to SQL Database](#2-save-dataframe-to-sql-database)
   - [Save DataFrame to CSV](#3-save-dataframe-to-csv)
   - [Histogram Plot for Numerical Features](#4-histogram-plot-for-numerical-features)
   - [Count Plot for Categorical Features](#5-count-plot-for-categorical-features)
   - [Stacked Histogram Plot for Numerical Features](#6-stacked-histogram-plot-for-numerical-features)
4. [Contributing](#contributing)

---

## **Installation**

To use the helper functions, Install  the dependencies:

```bash
pip install -r requirements.txt
```

---
```
## **Models codes**
 
1. model_KNN.py
2. model_logistic_regression.py
3. model_neural_network.py
4. model_Random_forest.py
5. model_comparison.py

The first four scripts are used to train and evaluate specific machine learning models (KNN, Random Forest, Logistic Regression, and Neural Network), respectively.

```
This script , `model_comparison.py`, evaluates multiple machine learning models (Neural Network, KNN, Logistic Regression, and Random Forest) on the heart disease prediction dataset. It selects the best-performing model based on metrics such as accuracy, precision, recall, and AUC score, and visualizes the model's performance through ROC curves, confusion matrices, and other evaluation metrics.

### Script

```python
import os
import joblib
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, roc_auc_score, roc_curve, confusion_matrix

# Run the script of the models (in src folder)
os.system("python model_KNN.py")
os.system("python model_neural_network.py")
os.system("python model_logistic_regression.py")
os.system("python model_Random_forest.py")

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING messages (shows only ERRORs)
tf.get_logger().setLevel(logging.ERROR)  # Suppress TensorFlow logging at lower levels

# Function to load models from .pkl files
def load_model(filepath):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the models
models = {
    "Neural Network": joblib.load("../models/best_nn_model.pkl"),
    "KNN": joblib.load("../models/best_knn_model.pkl"),
    "Logistic Regression": joblib.load("../models/best_logreg_model.pkl"),
    "Random Forest": joblib.load("../models/best_rf_model.pkl")
}

# Load the data
heart_data = pd.read_csv("../data/processed/heart_cleaned_data.csv")
numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
target_variable = 'HeartDisease'

X = heart_data[numerical_features + categorical_features]
y = heart_data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

# Evaluate all the models
model_metrics = {}
for model_name, model in models.items():
    # Predict and evaluate metrics
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None

    model_metrics[model_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc_score": auc_score,
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

# Determine the best model based on accuracy
best_model_name = max(model_metrics, key=lambda name: model_metrics[name]['accuracy'])
best_model = models[best_model_name]
best_metrics = model_metrics[best_model_name]
print(f"Best Model: {best_model_name}")
print(f"Metrics: {best_metrics}")

output_dir="../reports/final_visualizations"
output_dir_1="../reports"

# Save the evaluation results to a text file
results_file = os.path.join(output_dir_1, "results.txt")

with open(results_file, "w") as file:
    file.write("Model Evaluation Results\n")
    file.write("="*30 + "\n\n")
    
    for model_name, metrics in model_metrics.items():
        file.write(f"Model: {model_name}\n")
        file.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        file.write(f"  Precision: {metrics['precision']:.4f}\n")
        file.write(f"  Recall: {metrics['recall']:.4f}\n")
        if metrics['auc_score'] is not None:
            file.write(f"  AUC-ROC: {metrics['auc_score']:.4f}\n")
        file.write(f"  Confusion Matrix:\n{metrics['confusion_matrix']}\n\n")
    
    file.write("Best Model\n")
    file.write("="*30 + "\n")
    file.write(f"Model: {best_model_name}\n")
    file.write(f"Metrics:\n")
    file.write(f"  Accuracy: {best_metrics['accuracy']:.4f}\n")
    file.write(f"  Precision: {best_metrics['precision']:.4f}\n")
    file.write(f"  Recall: {best_metrics['recall']:.4f}\n")
    if best_metrics['auc_score'] is not None:
        file.write(f"  AUC-ROC: {best_metrics['auc_score']:.4f}\n")
    file.write(f"  Confusion Matrix:\n{best_metrics['confusion_matrix']}\n")    
print(f"Results saved to {results_file}")


# Plot AUC_ROC for the best model
if best_metrics['auc_score'] is not None:
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {best_metrics['auc_score']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    plt.title(f"ROC Curve of the best model : {best_model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"{output_dir}/ROC Curve of the best model_{best_model_name}.png")
    plt.show()

# Plot confusion matrix for the best model
cm = best_metrics['confusion_matrix']
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Heart Disease', 'Heart Disease']\
                                                 , yticklabels=['No Heart Disease', 'Heart Disease'])
plt.title(f"Confusion matrix of the best model: {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{output_dir}/Confusion matrix of the best model_{best_model_name}.png")
plt.show()

# Plot metrics for all models
metric_names = ['accuracy', 'precision', 'recall']
metrics_values = {metric: [model_metrics[model][metric] for model in models] for metric in metric_names}

plt.figure(figsize=(10, 6))
x = np.arange(len(models))
bar_width = 0.2

for i, metric in enumerate(metric_names):
    plt.bar(x + i * bar_width, metrics_values[metric], bar_width, label=metric.capitalize())

plt.xticks(x + bar_width, models.keys())
plt.ylabel("Score")
plt.title("Models Comparison: Accuracy, Precision, and Recall")
plt.legend()
plt.savefig(f"{output_dir}/Models Comparison.png")
plt.show()
```

---
## **Helper Functions**

### **1. Preprocess Data**

This function loads a dataset, removes missing values, remove duplicates and saves the cleaned data.

```python
def preprocess_data(input_path, output_path):
    """
    Preprocess the dataset by dropping missing values and renaming columns.
    """
```

- **Parameters**:
  - `input_path`: Path to the raw dataset.
  - `output_path`: Path to save the processed dataset.

---

### **2. Save DataFrame to SQL Database**

This function saves a DataFrame to a SQLite database.

```python
def save_to_sqlite(db_file, table_name, dataframe):
    """
    Saves a DataFrame to an SQLite database.
    """
```

- **Parameters**:
  - `db_file`: Path to the SQLite database.
  - `table_name`: Name of the SQL table.
  - `dataframe`: The DataFrame to save.

---

### **3. Save DataFrame to CSV**

This function saves a DataFrame to a CSV file.

```python
def save_to_csv(dataframe, file_path):
    """
    Save a DataFrame to a CSV file.
    """
```

- **Parameters**:
  - `dataframe`: The DataFrame to save.
  - `file_path`: Path to the CSV file.

---

### **4. Histogram Plot for Numerical Features**

This function creates histograms for multiple numerical features.

```python
def plot_histograms(df, numeric_features, output_path):
    """
    Plot histograms for numerical features in a dataset.
    """
```

- **Parameters**:
  - `df`: Input DataFrame.
  - `numeric_features`: List of numerical feature names.
  - `output_path`: Path to save the plot.

---

### **5. Count Plot for Categorical Features**

This function creates count plots for categorical features.

```python
def plot_categorical_count(df, categorical_features, hue, output_path):
    """
    Plot count plots for categorical features against a target.
    """
```

- **Parameters**:
  - `df`: Input DataFrame.
  - `categorical_features`: List of categorical feature names.
  - `hue`: Target feature for hue.
  - `output_path`: Path to save the plot.

---

### **6. Stacked Histogram Plot for Numerical Features**

This function creates stacked histograms with KDE for numerical features.

```python
def plot_stacked_histograms(df, numeric_features, hue, output_path):
    """
    Plot stacked histograms with KDE for numerical features.
    """
```

- **Parameters**:
  - `df`: Input DataFrame.
  - `numeric_features`: List of numerical feature names.
  - `hue`: Target feature for hue.
  - `output_path`: Path to save the plot.

---

## **Contributing**

Feel free to contribute additional helper functions or suggest improvements! Submit a pull request with a clear description of your changes.

--- 