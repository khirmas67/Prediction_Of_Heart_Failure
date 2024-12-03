# Data Processing and Analysis

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
   - Key visualizations include:  
     - A histogram of feature distributions.
     - A correlation heatmap.
     

   Example:  
   ![heart_disease_distribution](reports/visualizations/heart_disease_distribution.png)  
   ![Heatmap](path/to/heatmap.png)

6. **Processed Data**    
   - saved the following data to SQL database:
      - Heart Disease dataset and its statistical summary
      - Numerical features
      - Categorical features

## Dependencies
- Python 3.8+
- Libraries:  
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`



## Results



