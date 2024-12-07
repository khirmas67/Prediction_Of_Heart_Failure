## **This folder contains the following :**
# **Data Processing Helper Functions**

This repository contains reusable helper functions to streamline various data preprocessing and visualization tasks. These functions help improve efficiency, reduce code duplication, and maintain consistency across data projects.

---

## **Table of Contents**

1. [Installation](#installation)
2. [Helper Functions](#helper-functions)
   - [Preprocess Data](#1-preprocess-data)
   - [Save DataFrame to SQL Database](#2-save-dataframe-to-sql-database)
   - [Save DataFrame to CSV](#3-save-dataframe-to-csv)
   - [Histogram Plot for Numerical Features](#4-histogram-plot-for-numerical-features)
   - [Count Plot for Categorical Features](#5-count-plot-for-categorical-features)
   - [Stacked Histogram Plot for Numerical Features](#6-stacked-histogram-plot-for-numerical-features)
3. [Models codes](#Models-codes) 
4. [Folder Structure](#folder-structure)
5. [Contributing](#contributing)

---

## **Installation**

To use the helper functions, Install  the dependencies:

```bash
pip install -r requirements.txt
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
## **Models codes**
 
1. model_KNN.py
2. model_logistic_regression.py
3. model_neural_network.py
4. model_Random_forest.py
5. model_comparison.py

he first four codes are used to train and evaluate specific machine learning models (KNN, Random Forest, Logistic Regression, and Neural Network), respectively. The fifth code compares the performance of these models and selects the best one based on defined evaluation metrics.

## **Folder Structure**

The project follows this directory structure:

```
├── data/
│   ├── processed/            # Processed datasets
│   ├── raw/                  # Raw datasets
│   └── SQL/                  # SQLite databases
├── src/                      # Source code for helper functions
│   ├── data_preprocessing.py # Preprocessing functions
│   ├── df_to_sql.py          # SQLite-related functions
│   └── visualization_helpers.py  # Plotting functions
├── reports/                  # Reports and visualizations
│   └── visualizations/       # Saved plots
├── README.md                 # Project overview and instructions
├── requirements.txt          # Dependencies
└── experiments/              # Jupyter notebooks
```

---

## **Contributing**

Feel free to contribute additional helper functions or suggest improvements! Submit a pull request with a clear description of your changes.

--- 