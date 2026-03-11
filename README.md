# Machine Learning Pipeline: Loan Interest Rate Prediction & Stroke Prediction

## Project Overview

This project demonstrates a complete machine learning workflow including exploratory data analysis (EDA), preprocessing pipelines, and model comparison across multiple algorithms.

Two predictive modeling tasks were developed:

1. Loan Interest Rate Prediction (Regression) – Predict the interest rate assigned to a loan application.
2. Stroke Prediction (Classification) – Predict whether a patient is at risk of experiencing a stroke.

More than 13 machine learning models were evaluated using structured preprocessing pipelines and cross-validation to ensure fair model comparison.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Datasets

### Credit Risk Dataset – Loan Interest Rate Prediction

Source: Kaggle  
Observations: 32,416  
Variables: 12 features

Target variable:
loan_int_rate

The dataset includes borrower demographic characteristics, financial information, loan attributes, and credit history indicators.

Initial data inspection showed:

- Minimal missing values (~0.1%)
- No duplicate records
- Limited multicollinearity

---

### Healthcare Dataset – Stroke Prediction

Observations: 5,110  
Variables: 12 features

Target variable:
stroke

Key features include:

- Age
- Hypertension
- Heart disease
- BMI
- Smoking status
- Average glucose level

The dataset is highly imbalanced:

- 95% non-stroke
- 5% stroke

Class imbalance handling techniques such as class weighting and SMOTE were used.

---

## Exploratory Data Analysis (EDA)

EDA was conducted to understand dataset structure and relationships.

Key steps included:

- Missing value analysis
- Outlier detection using IQR
- Correlation analysis
- Statistical testing

Statistical tests performed:

- Chi-square test
- Independent t-test
- ANOVA

Key insights:

- Mortgage holders tended to have lower interest rates
- Defaulted loans had higher average interest rates
- Stroke risk was strongly associated with age, hypertension, heart disease, and smoking status

---

## Machine Learning Pipeline

Model-specific preprocessing pipelines were designed to accommodate different algorithm requirements.

Pipeline steps included:

- Missing value handling
- Categorical encoding
- Feature scaling
- Outlier treatment

Preprocessing varied by model type.

Tree-based models (Random Forest, Decision Tree) required encoding and missing value handling but did not require scaling.

Scale-sensitive models (Linear Regression, KNN, SVM) required encoding, scaling, and outlier management.

Gradient boosting models required minimal preprocessing due to their robustness.

---

## Model Training

The dataset was split into:

- 80% training
- 20% testing

Cross-validation was used to ensure robust model evaluation.

---

## Regression Models (Loan Interest Rate)

Models evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- K-Nearest Neighbors
- Support Vector Regression
- Decision Tree
- Random Forest
- AdaBoost
- XGBoost
- LightGBM
- CatBoost

### Results

Linear models performed strongly with test R² ≈ 0.905 and stable generalization.

Gradient boosting models achieved the highest performance, with tuned XGBoost and CatBoost achieving R² values above 0.906.

Top performing models:

1. LightGBM
2. CatBoost
3. Random Forest

These models captured nonlinear relationships and feature interactions effectively.

---

## Classification Models (Stroke Prediction)

Models evaluated:

- Logistic Regression
- LDA
- KNN
- SVM
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- Gradient Boosting

Due to severe class imbalance, evaluation focused on:

- F1 Score
- ROC-AUC
- Recall

Accuracy alone was not reliable because models could predict the majority class.

### Base Model Results

Most models achieved high accuracy (~95%) but failed to detect stroke cases due to imbalance.

### After Tuning and SMOTE

Performance improved with better minority class detection.

Top classification models:

- Logistic Regression
- Linear Discriminant Analysis (LDA)

These models achieved the best balance between recall and precision with F1 ≈ 0.28 and ROC-AUC ≈ 0.81.

---

## Feature Importance

Important predictors for loan interest rate:

- Person income
- Loan amount
- Employment length
- Credit history length
- Loan grade

Important predictors for stroke prediction:

- Age
- Hypertension
- Heart disease
- Smoking status

These predictors align with known financial and medical risk factors.

---

## Project Structure

```
ml-pipeline-model-comparison
│
├── notebooks
│   ├── 1_eda.ipynb
│   ├── 2_pipeline.ipynb
│   └── 3_modeling.ipynb
│
├── data
│   ├── loan_interest_rate.csv
│   └── stroke_prediction.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Key Takeaways

- Ensemble models significantly improved regression performance.
- Severe class imbalance limited stroke prediction performance.
- Logistic Regression and LDA provided the most stable classification results.
- Pipelines ensured consistent preprocessing across multiple models.

---

## Author

Zemenawit Kahsay  
Advanced Diploma in Data Science  
Toronto, Canada

