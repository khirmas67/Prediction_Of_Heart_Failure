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
The purpose of this project is to develop a KNN classification model to predict heart failure risk using patient data. It helps **pharmaceutical companies** identify patients who are likely to benefit from specific treatments or be part of the  clinical trials. Identifying the right patients can make trials run smoothly, improve drug development, and increase the effectiveness of new treatments.



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

The **target variable (HeartDisease** indicates whether heart failure was detected (binary: No = 0, Yes = 1).

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
## **Conclusion**


## **References**
1. Basic writing and formatting syntax [🔗](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) 

2. Heart Failure Prediction Dataset on Kaggle [🔗](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

3. 

## **Contact**
Feel free to reach out if you have any questions or feedback:  
**Khaled Hirmas** : https://github.com/khirmas67

   
