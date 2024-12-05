import pandas as pd
import numpy as np
import warnings
import joblib
import pickle
import seaborn as sns
from keras.layers import Dense
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from keras.models import Sequential
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scikeras.wrappers import KerasClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import  roc_auc_score, roc_curve, accuracy_score,\
    recall_score, precision_score, confusion_matrix
    
def plot_model_evaluation(model_name, y_test, y_pred, y_pred_prob, test_accuracy, \
    precision, recall, output_dir="../reports"):
    """
    Generate evaluation plots for each model.

    Parameters:
    - model_name (str): Name of the model (e.g., "KNN", "Neural Network").
    - y_test (array-like): True labels of the test set.
    - y_pred (array-like): Predicted classes of the model.
    - y_pred_prob (array-like): Predicted probabilities of the positive class.
    - test_accuracy (float): Accuracy score of the model.
    - precision (float): Precision score of the model.
    - recall (float): Recall score of the model.
    - output_dir (str): Directory to save the plots (default: "../reports").
    """

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f}, Accuracy = {test_accuracy:.4f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.title(f"{model_name}: ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"{output_dir}/{model_name}_ROC_Curve.png")
    plt.show()

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Heart Disease', 'Heart Disease'],
                yticklabels=['No Heart Disease', 'Heart Disease'])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"{output_dir}/{model_name}_Confusion_Matrix.png")
    plt.show()

    # Bar Plot for Accuracy, Precision, and Recall
    metrics = [test_accuracy, precision, recall]
    metric_names = ['Accuracy', 'Precision', 'Recall']

    plt.figure(figsize=(8, 6))
    plt.barh(metric_names, metrics, color=['green', 'orange', 'skyblue'])
    plt.title(f'{model_name}: Accuracy, Recall, and Precision')
    plt.xlabel('Scores')
    plt.savefig(f"{output_dir}/{model_name}_Scores.png")
    plt.show()

