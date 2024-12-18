# **Prediction OF Heart Failure Using Machine Learning**

![heart](https://github.com/user-attachments/assets/3669e712-1efa-42c7-bf2f-7775dc0acf41)

## **Table of Contents**  
1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Data preprocessing](#data-preprocessing)
4. [Assessing the Performance of Machine Learning Algorithms](#assessing-the-performance-of-machine-learning-algorithms)
5. [Model Evaluation and Selection](#model-evaluation-and-selection)  
6. [Conclusion](#conclusion)  
7. [How to Run the Project](#how-to-run-the-project) 
8. [Project Structure](#project-structure)  
9. [References](#references)  
10. [Contact](#contact)

--- 

## **Overview**
This project aims to develop machine learning models for predicting heart failure risk using patient clinical data, then choose the one with the best performance metrics. The candidate algorithms to develop these models are:  K-Nearest Neighbors , Random Forest ,  Logistic Regression and  Neural Network.  Such predictions can assist pharmaceutical companies in optimizing clinical trial participant selection and improving treatment effectiveness.

---

## **Business Case: Pharmaceutical Industry**
In the pharmaceutical industry, **identifying target patients** is very important for the success of clinical trials and drug treatments. The machine learning model in this project provides several benefits:  
1. Improves candidate selection for clinical trials.  
2. Reduces trial time and costs by selecting the right participants.  
3. Enhances treatment effectiveness.

---

## **Dataset**
The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). It contains **918 samples** with **11 clinical features**, including:
- Age  
- Sex
- Chest pain type
- Resting blood pressure  
- Serum cholesterol  
- Fasting blood sugar
- Resting electrocardiogram
- Maximum heart rate achieved
- Exercise Angina
- ST depression(OldPeak)
- Slope of the ST segment  

![The Dataset](https://github.com/user-attachments/assets/243b0c6f-2009-4561-a69c-f7846f4e6d66)


The distribution of Numerical Features in Relation to Heart Disease is shown in the figure below:
<div align="center">
  <img src="https://github.com/user-attachments/assets/caf41bff-79a2-4421-8c5b-e34229c38462" alt="Analysis of Numeric Features Related to Heart Disease" width="90%">
</div>


The distribution of Categorical Features vs Heart Disease Status is shown in the figure below:

<div align="center">
  <img src="https://github.com/user-attachments/assets/7796aa0f-40e2-43d8-88b8-8f6923c9b265" alt="Analysis of Categorical Features Related to Heart Disease" width="60%">
</div>

The target variable (HeartDisease) indicates whether heart failure was detected, with binary values: No = 0 and Yes = 1. The pie chart below visualizes the distribution of heart disease cases in the dataset. Approximately 55.3% of the cases have heart disease, while 44.7% do not. This distribution highlights the imbalance in the dataset.



<div align="center">
  <img src="https://github.com/user-attachments/assets/46490ede-1d9c-4711-82ec-7e6875acda73" alt="Heart Disease Distribution" width="50%">
</div>

---

### Data preprocessing
**The data were processeed as follows:**

**1 - Reading, Preprocessing, and Visualizing Data:**
- the Dataset :The raw dataset is loaded into a pandas DataFrame for further processing.
- Checking for Duplicates:  Duplicate entries are identified and handled accordingly.
- Displaying Dataset Information:Metadata such as data types, non-null counts, and summary statistics are explored
- Statistical Summary: A statistical overview of the dataset is calculated (e.g., mean, median, standard deviation).
- Conducted statistical tests (e.g., Chi-Square and T-tests) to identify feature significance.

**2 - Data Cleaning and Feature Engineering:**
- Necessary preprocessing steps are applied, such as:
  -  Removing or handling duplicate rows.
  -  Separating features into numerical and categorical groups. 
  -  Encoding categorical variables using one-hot encoding. 
  -  Standardizing numerical variables for model training.    

**3 - Data Splittting :**
- The dataset is divided into training and testing subsets to evaluate model performance effectively. A typical 80:20 split is applied.

**4 - Data Visualization**
-  plots are used to understand data distributions and relationships and identifying trends.

---

### Assessing the Performance of Machine Learning Algorithms
In this project, the aimed was to identify the most effective machine learning model for solving a classification problem- The prediction of Heart Failure. The selection process involved training and evaluating four popular algorithms: **Random Forest**, **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and a **Neural Network**. These models were chosen because they represent a diverse range of machine learning paradigms, offering different strengths and trade-offs in terms of accuracy, interpretability, and computational complexity.

To ensure a fair comparison, all models were evaluated using consistent preprocessing and performance metrics, including **accuracy**, **precision**, **recall**, and the **AUC-ROC**. These metrics were selected to capture the overall performance, balance between false positives and false negatives, and the suitability of each model for our specific application needs.

The goal of this evaluation was to:
1. Compare the models’ performances across key metrics.
2. Identify the model best suited for deployment based on specific requirements, such as high recall (sensitivity) or overall balance in performance.
3. Save the trained models - in pickle files - for future use, ensuring reproducibility and efficient implementation.

---

### Model Evaluation and Selection
This script , `src/model_comparison.py`, evaluates multiple machine learning models (Neural Network, KNN, Logistic Regression, and Random Forest) on the heart disease prediction dataset,then selects the best-performing model based on metrics such as accuracy, precision, recall, and AUC score, and visualizes the model's performance through ROC curves, confusion matrices, and other evaluation metrics.

### The Script
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
## **The results:** 
`Evaluation Metrics:`
Each model's performance is summarized by four metrics: 
- **Accuracy**
- **Precision**
- **Recall**
- **AUC score**

| **Model**                | **Accuracy** | **Precision** | **Recall** | **AUC score** |
|--------------------------|--------------|---------------|------------|---------------|
| **Random Forest**        | 0.9239       | 0.9048        | 0.9596     | 0.9755        |
| **Logistic Regression**  | 0.9076       | 0.9271        | 0.8990     | 0.9697        |
| **K-Nearest Neighbors**  | 0.9076       | 0.9362        | 0.8889     | 0.9700        |
| **Neural Network**       | 0.9185       | 0.9375        | 0.9091     | 0.9471        |

`Key Observations:`
1. **Random Forest**: Achieves the highest accuracy (0.9239 )and recall (0.9596) AUC score (0.9755), making it an excellent choice for applications prioritizing sensitivity and overall performance.
2. **Logistic Regression**: Offers the highest precision (0.9271) and balances metrics well, but recall (0.8990) is lower than Random Forest.
3. **KNN**: Precision is highest (0.9362), but recall (0.8889) lags, making it less balanced.
4. **Neural Network**: Performs competitively across all metrics, with the highest precision (0.9375) among models but slightly lower AUC score (0.9471) than Random Forest.

`Saved Models`
- **Random Forest**: `models/best_rf_model.pkl` 
- **Logistic Regression**: `models/best_logreg_model.pkl`
- **K-Nearest Neighbors**: `models/best_knn_model.pkl`
- **Neural Network**: `models/best_nn_model.pkl`

![Models Comparison](https://github.com/user-attachments/assets/63bd6b64-03e5-4933-806b-55a9d2cb8c2c)


![ROC Curve and Confusion matrix of the best model_Random Forest](https://github.com/user-attachments/assets/fdb6ef9a-0c18-420d-ba81-58b56ee4c56e)


---

### **Conclusion**
In this project, we developed and evaluated multiple machine learning models to predict heart failure risk using clinical data. Among the four models—Random Forest, K-Nearest Neighbors (KNN), Logistic Regression, and Neural Network—the **Random Forest** model emerged as the most accurate and reliable, achieving a **92.39% test accuracy** and an **AUC-ROC score of 0.9755**. These results demonstrate the model's strong ability to differentiate between patients with and without heart failure.

The successful identification of heart failure risk using machine learning can significantly aid pharmaceutical companies by enhancing **clinical trial participant selection**, **reducing trial costs**, and **improving treatment effectiveness**. By implementing this model, stakeholders in the healthcare and pharmaceutical industries can make more informed decisions that ultimately lead to better patient outcomes.

This project underscores the potential of machine learning in clinical applications, paving the way for more advanced and precise healthcare solutions.

---

### **How to Run the Project**
1. **Clone the repository:**
   ```bash
   git clone (https://github.com/khirmas67/Prediction_Of_Heart_Failure.git)
   cd Prediction_Of_Heart_Failure
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt 
   ```
---

### **Project Structure**
The project Structure is as follows:
```bash
Prediction_Of_Heart_Failure/
├── data/
│   ├── processed/            # Folder for processed data
│   ├── raw/                  # Folder for raw data
│   └── SQL/                  # Folder for tables
├── experiments/              # Jupyter notebooks for experiments and EDA
├── models/                   # Trained models 
├── reports/                  # Visualizations, evaluation reports from EDA and model performance
├── README.md                 # Project overview and instructions (this file)
├── requirements.txt          # List of dependencies
└── src/                      # Python scripts for preprocessing, training, and evaluation

```
--- 

### **References**
1. **Basic writing and formatting syntax** [🔗](https://www.markdownguide.org/basic-syntax/)  

2. **G. James et al.,** *An Introduction to Statistical Learning with Applications in Python* (Springer: 2023)  
   
3. **Heart Failure Prediction Dataset on Kaggle** [🔗](https://www.kaggle.com/datasets)  
  
4. **Choosing color palettes** [🔗](https://www.colorhexa.com/)  

5. **Scikit-learn Documentation** [🔗](https://scikit-learn.org/)  
   
6. **TensorFlow Documentation** [🔗](https://www.tensorflow.org/)  
   
7. **Kaggle - Machine Learning Tutorials** [🔗](https://www.kaggle.com/learn)  
   A collection of tutorials and datasets from Kaggle to help learn and apply machine learning algorithms to real-world problems.

8. **Python Plotly Documentation** [🔗](https://plotly.com/python/)  
   
---

### **Contact**
Feel free to reach out if you have any questions or feedback:  
**Khaled Hirmas** : https://github.com/khirmas67


The Project Video: Prediction of Heart Failure: https://drive.google.com/file/d/1zAGJU20T4gPiQutDFkGph8t0_Z0UzCZF/view?usp=drive_link



   
