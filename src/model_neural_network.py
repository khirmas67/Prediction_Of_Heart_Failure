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

# Preprocess and get input dimensions
X_train_transformed = preprocessor.fit_transform(X_train)
input_dim = X_train_transformed.shape[1]  # Number of features after preprocessing

# Function to create the Sequential model
def create_nn_model(hidden_layer_sizes=(128,)):
    model = Sequential()
    # Input layer
    model.add(Dense(64, input_dim=input_dim, activation='relu'))  # First hidden layer
    for units in hidden_layer_sizes:
        model.add(Dense(units, activation='relu'))  # Additional hidden layers
    model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Directory for model checkpoints
checkpoint_dir = "../models/checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_filepath = os.path.join(checkpoint_dir, "best_nn_model.h5")

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# Create and train the model
nn_model = create_nn_model()
history = nn_model.fit(
    preprocessor.transform(X_train),
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint_callback],
    verbose=1
)

# Load the best model from the checkpoint
best_nn_model = load_model(checkpoint_filepath)

# Evaluate the model
y_pred = (best_nn_model.predict(preprocessor.transform(X_test)) > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Calculate AUC
y_pred_prob = best_nn_model.predict(preprocessor.transform(X_test))
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Score: {auc_score:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
'''plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f}, Accuracy = {test_accuracy:.4f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
plt.title("Neural Network: ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig("../reports/visualizations_models/Neural_Network_ROC_Curve.png")
plt.show()

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Heart Disease', 'Heart Disease'], yticklabels=['No Heart Disease', 'Heart Disease'])
plt.title("Neural Network : Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("../reports/visualizations_models/Neural_Network_Confusion_Matrix.png")
plt.show()

# Bar plot for Accuracy, Precision, and Recall
metrics = [test_accuracy, precision, recall]
metric_names = ['Accuracy', 'Precision', 'Recall']
plt.figure(figsize=(8, 6))
plt.barh(metric_names, metrics, color=['green', 'orange', 'skyblue'])
plt.title('Accuracy, Recall, and Precision')
plt.xlabel('Neural Network : Scores')
plt.savefig("../reports/visualizations_models/Neural_Network_Scores.png")
plt.show()'''

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


with open('../models/best_nn_model.pkl', 'wb') as file:
    joblib.dump(best_nn_model, file)
    print("Best model saved as 'best_nn_model.pkl'")