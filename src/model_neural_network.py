import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib 




# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# import load_split_preprocess helper function to load raw data, split it, then preprocess it
from load_split_preprocess import load_and_preprocess_data
X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("../data/processed/heart_cleaned_data.csv")

    
    
    
    
    
    
    
    
    
    
  
  
  
  
  
  
  
import pandas as pd
import numpy as np
import warnings
import joblib
import sys
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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score,\
    recall_score, precision_score, confusion_matrix  
    
import warnings
import tensorflow as tf
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


warnings.filterwarnings('ignore')

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages (shows only ERRORs)

# Additional logging configuration
import logging
tf.get_logger().setLevel(logging.ERROR)


# To get the correct number of input features after one-hot encoding
X_train_transformed = preprocessor.fit_transform(X_train)
input_dim = X_train_transformed.shape[1]  # Number of features after preprocessing

# Function to create the Sequential model
def create_nn_model(hidden_layer_sizes=(64, 128)):
    model = Sequential()
    # Add input layer (with number of features as input_dim)
    model.add(Dense(64, input_dim=input_dim, activation='relu'))  # First hidden layer
    for units in hidden_layer_sizes:
        model.add(Dense(units, activation='relu'))  # Additional hidden layers
    model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=5,  # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore the model weights from the epoch with the best value
)

# ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    filepath='../best_nn_model_checkpoint',  # File path to save the model
    monitor='val_loss',  # Monitor validation loss
    save_best_only=True,  # Only save the best model based on validation loss
    verbose=0,  #  verbose = 1 Print messages when the model is saved
    save_weights_only=False  # Save the entire model (including architecture, optimizer, and weights)
)

# Create a KerasClassifier wrapper for the neural network with EarlyStopping and ModelCheckpoint
nn_classifier = KerasClassifier(
    build_fn=create_nn_model, 
    epochs=200, 
    batch_size=32, 
    verbose=0, 
    validation_split=0.2,  # Use 20% of training data for validation
    callbacks=[early_stopping, checkpoint]  # Add both callbacks
)

# Create a pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", nn_classifier)])

# Define parameter grid for Neural Network
param_grid = {
    'classifier__epochs': [50, 100, 200],  # Number of epochs for training
    'classifier__batch_size': [32],  # Batch sizes
}

# Grid search for best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', return_train_score=True)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best Parameters Found:")
print(grid_search.best_params_)

# Best model based on grid search
best_nn_model = grid_search.best_estimator_

# Extract the Keras model from the pipeline
keras_model = best_nn_model.named_steps["classifier"].model_


if keras_model is None:
    raise ValueError("The Keras model could not be extracted. Ensure the classifier is properly fitted.")

# Create a TensorFlow Checkpoint
checkpoint = tf.train.Checkpoint(model=keras_model)

# Define the path to the checkpoint directory
checkpoint_path = '../models/best_nn_model_checkpoint'

# Restore the checkpoint
status = checkpoint.restore(checkpoint_path)

# Check if all variables are restored
try:
    status.assert_consumed()
    print("All variables restored successfully.")
except AssertionError:
    print("Some variables were not restored.")



# Predictions
y_pred = best_nn_model.predict(X_test)

# Perform cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(best_nn_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# Test accuracy on the best model
train_accuracy = accuracy_score(y_train, best_nn_model.predict(X_train))
test_accuracy = accuracy_score(y_test, best_nn_model.predict(X_test))
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Calculate AUC for the best model
y_pred_prob = best_nn_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Score_test: {auc_score:.4f}")

y_pred_prob_train = best_nn_model.predict_proba(X_train)[:, 1]
auc_score_train = roc_auc_score(y_train, y_pred_prob_train)
print(f"AUC Score_train: {auc_score:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

# Plot confusion matrix, AUC_ROC, (accuracy, recall, and precision) for the best model

from plot_model_evaluation import plot_model_evaluation 

plot_model_evaluation(
    model_name="NEURAL NETWORK",
    y_test=y_test,
    y_pred=y_pred,
    y_pred_prob=y_pred_prob,
    test_accuracy=test_accuracy,
    precision=precision,
    recall=recall
)

# Save the best model using pickle
with open('../models/best_nn_model.pkl', 'wb') as file:
    joblib.dump(best_nn_model, file)
    print("Best model saved as 'best_nn_model.pkl'")
    