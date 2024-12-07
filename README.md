# **Prediction OF Heart Failure Using Machine Learning**

## **Table of Contents**  
1. [Overview](#overview)  
3. [Project Structure](#project-structure)  
4. [Dataset](#dataset)  
5. [How to Run the Project](#how-to-run_the-Project)  
7. [Conclusion](#conclusion) 
6. [References](#refernces) 
7. [Contact](#contact) 

## **Overview**
This project aims to develop machine learning models for predicting heart failure risk using patient clinical data, then choose the one with the best performance metrics. The candidate algorithms to develop these models are:  K-Nearest Neighbors , Random Forest ,  Logistic Regression and  Neural Network.  Such predictions can assist pharmaceutical companies in optimizing clinical trial participant selection and improving treatment effectiveness.


 
## **Business Case: Pharmaceutical Industry**
In the pharmaceutical industry, **identifying target patients** is very important for the success of clinical trials and drug treatments. The machine learning model in this project provides several benefits:  
1. Improves candidate selection for clinical trials.  
2. Reduces trial time and costs by selecting the right participants.  
3. Enhances treatment effectiveness.

## **Project Structure**
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

The distribution of numerical features with respect to the target variable (HeartDisease) is shown in the figure below:
![heart_disease_numeric_features_analysis](https://github.com/user-attachments/assets/caf41bff-79a2-4421-8c5b-e34229c38462)

The distribution of numerical features with respect to the target variable (HeartDisease) is shown in the figure below:![heart_disease__categorical_features_analysis](https://github.com/user-attachments/assets/7796aa0f-40e2-43d8-88b8-8f6923c9b265)

The **target variable (HeartDisease** indicates whether heart failure was detected (binary: No = 0, Yes = 1).
The pie chart below represents the distribution of heart disease and no heart disease cases in the dataset. It shows shows that approximately 55.3% of the cases in the dataset have heart disease, while 44.7% do not

![heart_disease_distribution](https://github.com/user-attachments/assets/46490ede-1d9c-4711-82ec-7e6875acda73)

## **How to Run the Project**
1. **Clone the repository:**
   ```bash
   git clone (https://github.com/khirmas67/Prediction_Of_Heart_Failure.git)
   cd Prediction_Of_Heart_Failure
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt 
   ```


## **The results:** 
 ```
1.Random Forest 
   i.Best Parameters: max_depth=3, min_samples_leaf=20, n_estimators=100
   ii.Performance: Train Accuracy: 86.38%, Test Accuracy: 92.39%, AUC-ROC: 0.9755
2.Logistic Regression 
   i.Best Parameters: C=0.19, penalty='l2', solver='liblinear'
   ii.Performance: Train Accuracy: 86.38%, Test Accuracy: 90.76%, AUC-ROC: 0.9697
3.K-Nearest Neighbors (KNN) 
   i.Best Parameters: n_neighbors=20
   ii.Performance: Train Accuracy: 86.38%, Test Accuracy: 90.76%, AUC-ROC: 0.9700
4.Neural Network 
   i.Best Parameters: batch_size=32, epochs=100
   ii.Performance: Train Accuracy: 84.88%, Test Accuracy: 88.59%, AUC-ROC: 0.9626

 ```
## **Conclusion**
The Random Forest classifier was identified as the best-performing model, achieving a 92.39% test accuracy and an AUC-ROC of 0.9755. This model provides a reliable tool for predicting heart failure risk, which can benefit pharmaceutical companies in improving clinical trial participant selection and optimizing treatment strategies.

   

## **References**
1. Basic writing and formatting syntax [🔗](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) 

2. G. James et al., An Introduction to Statistical Learning with applications in Python(pringer: 2023)

3. Heart Failure Prediction Dataset on Kaggle [🔗](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

4. Choosing color palettes [🔗](https://seaborn.pydata.org/tutorial/color_palettes.html)

5. Helper Function for Plotting [🔗](https://matplotlib.org/stable/gallery/color/named_colors.html#)

## **Contact**
Feel free to reach out if you have any questions or feedback:  
**Khaled Hirmas** : https://github.com/khirmas67

https://drive.google.com/file/d/1zAGJU20T4gPiQutDFkGph8t0_Z0UzCZF/view?usp=drive_link

 [Watch the Project Video: Prediction of Heart Failure](https://drive.google.com/file/d/your_file_id/view?usp=sharing)


   
