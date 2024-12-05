import pandas as pd
import numpy as np
import warnings
import joblib
import pickle
import os
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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score,\
    recall_score, precision_score, confusion_matrix
    
    
import sys
sys.path.append('..') 
from src.load_split_preprocess import load_and_preprocess_data
X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("../data/processed/heart_cleaned_data.csv")


os.system("python model_KNN.py")
os.system("python model_neural_network.py")
os.system("python model_logistic_regression.py")
os.system("python model_Random_forest.py")


'''os.system("python code model_comparison.py") '''