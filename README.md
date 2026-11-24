# Heart Disease Prediction using Machine Learning

> **Disclaimer:** this is an educational project. It is not a medical device and must not be used for diagnosis or treatment decisions.

## Overview
A Streamlit web app that predicts the likelihood of heart disease from patient clinical data and compares several machine learning models.

## Features
- Heart disease prediction from 13 clinical inputs, with clinically labelled options
- Four models: SVM, KNN, Decision Tree and XGBoost
- Probability chart for each prediction
- Model comparison computed live: accuracy, precision, recall, F1 on a held-out test set, plus 5-fold cross-validated accuracy

## Dataset
`G3_heart_disease.csv` holds 270 patients (150 without and 120 with heart disease), the Statlog heart dataset.

| Column | Meaning |
|---|---|
| Age | Age in years |
| Sex | 1 = male, 0 = female |
| Chest pain type | 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic |
| BP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FBS over 120 | Fasting blood sugar > 120 mg/dl (1 = yes) |
| EKG results | 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy |
| Max HR | Maximum heart rate achieved |
| Exercise angina | Exercise-induced angina (1 = yes) |
| ST depression | ST depression induced by exercise relative to rest |
| Slope of ST | 1 = upsloping, 2 = flat, 3 = downsloping |
| Number of vessels fluoro | Major vessels (0–3) coloured by fluoroscopy |
| Thallium | 3 = normal, 6 = fixed defect, 7 = reversible defect |
| Heart Disease | Presence / Absence (target) |

## Models
SVM and KNN are wrapped in a pipeline with `StandardScaler`, since both are sensitive to feature scale. Models are trained once per server start and cached.

## Installation

```bash
pip install -r requirements.txt
streamlit run app_mlpheart.py
```
