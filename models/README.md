This folder contains all files related to the machine learning models used in this project. It includes trained models, checkpoints, configurations, evaluation results. Below is a description of each subdirectory and its contents.

---

## Contents

### 1. **Trained Models**
   - it ontains fully trained and saved models in a form of pickle files ready for deployment.
   

### 2. **Checkpoints**
   - **Location**: `best_nn_model_checkpoint/`
   - **Description**: Contains intermediate states of models saved during training for resuming or debugging.
   



### 4. **Model Evaluation Results**
   -  Includes evaluation metrics and assessing models performances.
---

## Usage

1. **Loading a Trained Model**:
   - Example (Keras model):
     ```
     
     import joblib
     joblib.load('../models/best_model.pkl')

     ```

2. **Resuming Training from a Checkpoint**:
   - Example:
     ```
     import tensorflow as tf

     checkpoint = tf.train.Checkpoint(model=model)
     checkpoint.restore('models/best_nn_model_checkpoint/checkpoint_epoch_10.ckpt')
     ```
