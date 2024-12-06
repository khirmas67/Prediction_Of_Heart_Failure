This file contains description of the following:
---

## I - Data Exploration and Preprocessing ( notebook : data_processing ):

This notebook explores and preprocesses a dataset related to heart disease prediction. It includes:

* Importing required libraries
* Loading and inspecting the heart disease dataset
* Handling missing values (if any)
* Identifying target variable, numerical and categorical features
* Visualizing the data distribution
* Performing Chi-Square tests for categorical features

### Dependencies

This notebook requires the following Python libraries:

* pandas
* numpy
* seaborn
* sqlite3
* matplotlib.pyplot
* matplotlib.colors
* sklearn.preprocessing
* sklearn.neighbors
* sklearn.metrics
* sklearn.model_selection
* sklearn.pipeline
* sklearn.compose

### Data Exploration and Preprocessing

1. **Importing Libraries:** The notebook starts by importing the necessary libraries for data manipulation, visualization, and model building.

2. **Loading Data:** The heart disease dataset is loaded using pandas and stored in a DataFrame named `heart_data`.

3. **Data Inspection:**
   - Checks for duplicate rows in the dataset.
   - Displays information about the DataFrame, including data types and missing values.
   - Generates a summary of the numerical features using descriptive statistics.
   - Converts the summary statistics into an SQL table and saves it to a database.

4. **Data Cleaning:** 
   - Verifies there are no missing values in the dataset.

5. **Feature Engineering:**
   - Identifies the target variable (HeartDisease) and separates it from the other features.
   - Categorizes the features into numerical and categorical.

6. **Data Visualization:**
   - Visualizes the distribution of the target variable (heart disease vs. no heart disease).
   - Creates histograms to explore the distribution of numerical features.
   - Creates KDE plots to analyze the distribution of numerical features with respect to the target variable.
   - Generates bar charts to visualize the distribution of categorical features and their relationship to the target variable.

7. **Statistical Analysis:**
   - Performs T-tests to compare the means of numerical features between groups with and without heart disease.
   - Conducts Chi-Square tests to assess the association between categorical features and the target variable.

8. **Data Storage:**
   - saved the following data to SQL database:
      - Heart Disease dataset and its statistical summary
      - Numerical features
      - Categorical features
**Note:** The notebook includes comments explaining each step of the process for better understanding.




5. **Figures**     

- **Heart disease distribution**

   ![heart_disease_distribution](https://github.com/user-attachments/assets/6e59b24a-cb39-4b66-89ce-0f92bde434c9)
   - **Distribution of numerical features in the dataset**
![distribution of numerical features in the dataset](https://github.com/user-attachments/assets/096f2b33-3531-4198-ac68-e5af5112f6a9)
   - **Heart disease numeric features analysis**

![heart_disease_numeric_features_analysis](https://github.com/user-attachments/assets/9044c20c-f476-4bb6-9c0c-74322adde5ee)
   - **Heart disease categorical features analysis**

![heart_disease__categorical_features_analysis](https://github.com/user-attachments/assets/2f958617-f291-4a99-846f-f81ceafa4939)


### II - The models ( notebookes : all_models_and_comaparison, all_models_and_comaparison_1 ):

In the `all_models_and_comparison` notebook, I experimented with various models without explicitly addressing overfitting. I added cross-validation (CV) scores, compared training accuracy with test accuracy, and included the AUC score for the training set to assess performance. In contrast, in the `all_models_and_comparison_1` notebook, I further optimized the models to mitigate overfitting. Both notebooks explore the performance of various machine learning algorithms for predicting heart disease using a heart disease dataset. The evaluated algorithms include:

    Random Forest
    Logistic Regression
    K-Nearest Neighbors (KNN)
    Neural Network

The notebooks perform the following steps:

    Import Libraries: Imports necessary libraries for data manipulation, model building, evaluation, and visualization.
    Load, Split, and Preprocess Data: Loads the heart disease data, splits it into training and testing sets, and preprocesses the data using standard scaling and one-hot encoding for categorical features.
    Random Forest:
        Creates a Random Forest classifier with hyperparameter tuning using GridSearchCV.
        Evaluates the best model on the test set and reports performance metrics including accuracy, precision, recall, and AUC-ROC score.
        Saves the best model using pickle.
    Logistic Regression:
        Creates a Logistic Regression classifier with hyperparameter tuning using GridSearchCV.
        Evaluates the best model on the test set and reports performance metrics.
        Saves the best model using pickle.
    KNN:
        Creates a KNN classifier with hyperparameter tuning using GridSearchCV.
        Evaluates the best model on the test set and reports performance metrics.
        Saves the best model using pickle.
    Neural Network:
        Defines a function to create a sequential neural network model with variable hidden layer sizes.
        Creates a KerasClassifier wrapper for the neural network with hyperparameter tuning using GridSearchCV.
        Evaluates the best model on the test set and reports performance metrics.
        Saves the best model using pickle.


After finding the best model for each classifier a comaprison is made between them to find the best from them     

Note: The notebook includes warnings related to TensorFlow restoring checkpoint values. These warnings can be ignored for the purpose of this analysis.
