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
This project aims to develop a machine learning model, KNN Classification, that predicts the risk of heart failure using patient data. The project focuses on helping **pharmaceutical companies** identify patients who are most likely to benefit from specific treatments or participate in clinical trials. If the right individuals are identified, then pharmaceutical companies can smooth clinical trials, optimize drug development, and enhance the effectiveness of the new treatments.



## **Business Case: Pharmaceutical Industry**
In the pharmaceutical industry, **identifying target patients** plays an important role in ensuring the success of clinical trials and drug treatments. The machine learning model  used in this project offers several benefits for the industry: 
1. Improve the selection of candidates for clinical trials.
2. Selection of right candidates reduces time and cost of the trials.
3. Increases treatment efficiency.

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
The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). It contains **299 samples** with **13 clinical features**, including:
- Age  
- Resting blood pressure  
- Serum cholesterol  
- Maximum heart rate achieved  
- ST depression induced by exercise  
- Slope of the ST segment  
- Presence of chest pain  
- Whether the patient has diabetes or hypertension

The **target variable** indicates whether heart failure was detected (binary: No = 0, Yes = 1).

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


## **Refernces**
1. Basic writing and formatting syntax [🔗](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) 

2. Heart Failure Prediction Dataset on Kaggle [🔗](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

3. 

## **Contact**
Feel free to reach out if you have any questions or feedback:  
**Khaled Hirmas** : https://github.com/khirmas67

   
