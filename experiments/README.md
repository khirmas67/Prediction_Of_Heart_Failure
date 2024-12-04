This file contains description of the following:
---
# (I) Data Processing and Analysis

## Overview
This part of the project focuses on data processing and analysis for predicting heart failure outcomes. The notebook includes steps for loading, cleaning, and analyzing the dataset, as well as visualizations to interpret the results.

## Steps in the Notebook

1. **Loading the Data**  
   - Imported the dataset using `pandas`.  
   - Verified the structure and checked for missing values or anomalies. 

2. **Data Cleaning**  
   - Removed duplicates using `pandas.DataFrame.duplicated()`.  
   - Handled missing values by imputing or dropping rows/columns.  


3. **Exploratory Data Analysis (EDA)**
   - Visualized the distribution of the categories (0 = No heart disease, 1 = Heart disease) of target variable
   - Visualized the distribution of numerical features
   - Visualized Distribution of numerical features with respect to the target variable (HeartDisease) categories
   

4. **Statistical Analysis**  
   - Applied a **t-test** to analyze the impact of numerical features on Heart Disease Risk.       
   - Applied a **chi-test** to analyze the impact of categorical features on Heart Disease Risk . 

5. **Figures**     

- **Heart disease distribution**

   ![heart_disease_distribution](https://github.com/user-attachments/assets/6e59b24a-cb39-4b66-89ce-0f92bde434c9)
   - **Distribution of numerical features in the dataset**
![distribution of numerical features in the dataset](https://github.com/user-attachments/assets/096f2b33-3531-4198-ac68-e5af5112f6a9)
   - **Heart disease numeric features analysis**

![heart_disease_numeric_features_analysis](https://github.com/user-attachments/assets/9044c20c-f476-4bb6-9c0c-74322adde5ee)
   - **Heart disease categorical features analysis**

![heart_disease__categorical_features_analysis](https://github.com/user-attachments/assets/2f958617-f291-4a99-846f-f81ceafa4939)
6. **Processed Data**    
   - saved the following data to SQL database:
      - Heart Disease dataset and its statistical summary
      - Numerical features
      - Categorical features

# (II) - How to use helper finctions

# (III) KNN Classification Notebook


## Dependencies
- Python 3.8+
- Libraries:  
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`



