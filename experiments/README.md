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




5. **Examples of the Figures**     

- **Heart disease distribution**

   ![heart_disease_distribution](https://github.com/user-attachments/assets/6e59b24a-cb39-4b66-89ce-0f92bde434c9)
   - **Distribution of numerical features in the dataset**
![distribution of numerical features in the dataset](https://github.com/user-attachments/assets/096f2b33-3531-4198-ac68-e5af5112f6a9)
   - **Heart disease numeric features analysis**

![heart_disease_numeric_features_analysis](https://github.com/user-attachments/assets/9044c20c-f476-4bb6-9c0c-74322adde5ee)
   - **Heart disease categorical features analysis**

![heart_disease__categorical_features_analysis](https://github.com/user-attachments/assets/2f958617-f291-4a99-846f-f81ceafa4939)


### II - The models ( notebookes : all_models_and_comaparison, all_models_and_comaparison_1 , experimental_model_neural_network, experimental_model_Logistic, experimental_models_KNN_and_RandomForest):

In the `experimental_model_neural_network`, `experimental_model_Logistic` and  `experimental_models_KNN_and_RandomForest` notebooks, I experimented with various baseline models and tested the processors in `experimental_models_KNN_and_RandomForest` to select the best processor. which I used in `all_models_and_comparison` and `all_models_and_comparison_1`:

<img src="https://github.com/user-attachments/assets/8463dc45-7ba2-4910-9a63-1095d369f7f5" width="400" />
```
In the `all_models_and_comparison` notebook, I experimented with various models without explicitly addressing overfitting. I added cross-validation (CV) scores, compared training accuracy with test accuracy, and included the AUC score for the training set to assess performance. In contrast, in the `all_models_and_comparison_1` notebook, I further optimized the models to mitigate overfitting. Both notebooks explore the performance of various machine learning algorithms for predicting heart disease using a heart disease dataset. The evaluated algorithms include:

    Random Forest
    Logistic Regression
    K-Nearest Neighbors (KNN)
    Neural Network


    The notebooks perform the following steps:
   Import Libraries: Imports necessary libraries for data manipulation, and visualization, then Load the heart disease data, splits it into training and testing sets, and preprocesses the data using standard scaling and one-hot encoding for categorical features, then model build and evaluate the modles on the test set and reports performance metrics and select the best model, then save the best model for each algorithm as pickel file. The summary is as follows:

    Random Forest:

        Best Parameters Found:
            {'classifier__max_depth': 3, 'classifier__min_samples_leaf': 20, 'classifier__min_samples_split': 20, 'classifier__n_estimators': 100}
            Train Accuracy: 0.8638
            Test Accuracy: 0.9239
            Mean Cross-Validation Accuracy: 0.8515
            Precision: 0.9048
            Recall: 0.9596
            AUC Score_test: 0.9755
            AUC Score_train: 0.9755

         conclusion:
         a difference of just 0.01 between training and validation accuracy is typically a good sign of balanced performance, not overfitting.

   Logistic Regression:

         Best Parameters Found:
            {'classifier__C': 0.19144819761699575, 'classifier__penalty': 'l2', 'classifier__solver': 'liblinear'}
            Mean Cross-Validation Accuracy: 0.8570
            Train Accuracy: 0.8638
            Test Accuracy: 0.9076
            Precision: 0.9271
            Recall: 0.8990
            AUC Score_test: 0.9697
            AUC Score_train: 0.9697

         conclusion:
         a difference of just 0.0068 between training and validation accuracy is typically a good sign of balanced performance, not overfitting.

    KNN:
        
         Best Parameters Found:
            {'classifier__n_neighbors': 20}
            Mean Cross-Validation Accuracy: 0.8638
            Train Accuracy: 0.8638
            Test Accuracy: 0.9076
            Precision: 0.9362
            Recall: 0.8889
            AUC Score_test: 0.9700
            AUC Score_train: 0.9700

         conclusion:
         a difference of just 0 between training and validation accuracy is typically a good sign of balanced performance, not overfitting.


    Neural Network:
        Defines a function to create a sequential neural network model with variable hidden layer sizes.
        Creates a KerasClassifier wrapper for the neural network with hyperparameter tuning using GridSearchCV.
        Evaluates the best model on the test set and reports performance metrics.
        Saves the best model using pickle.

        Best Parameters Found:
            {'classifier__batch_size': 32, 'classifier__epochs': 100}
            Some variables were not restored.
            Mean Cross-Validation Accuracy: 0.8447
            Train Accuracy: 0.8488
            Test Accuracy: 0.8859
            Precision: 0.8750
            Recall: 0.9192
            AUC Score_test: 0.9626
            AUC Score_train: 0.9626 

         conclusion:
         a difference of just 0.0041 between training and validation accuracy is typically a good sign of balanced performance, not overfitting.

After finding the best model for each classifier a comaprison is made between them to find the best from them  

### **The best model was: Random Forest**
```