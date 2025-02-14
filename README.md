# Predicting Hazardous NEOs (Nearest Earth Objects)

## Overview

This project involves predicting whether a Nearest Earth Object (NEO) is hazardous or not using machine learning techniques. The dataset provided by NASA includes information about NEOs, and the goal is to build and evaluate models to classify these objects accurately.

## Project Structure

- **data/**: Directory containing the dataset.
- **models/**: Directory where the trained models are saved.
- **reports/**: Directory where evaluation reports are saved.
- **Predicting_Hazardous_NEOs_EDA.ipynb**: Jupyter notebooks used for exploratory data analysis (EDA) and visualizations.
- **model.py**: Source code for data processing, model training, and evaluation.

## Requirements

- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `imblearn`
  - `joblib`

You can install the required libraries using:

```bash
pip install -r requirements.txt
```

## Data

The dataset used in this project is stored in `data/nearest-earth-objects(1910-2024).csv`. It contains various features related to NEOs and a target variable indicating whether the NEO is hazardous.

## Steps

1. **Data Importing and Cleaning**

   - Load the dataset.
   - Handle missing values by dropping rows with any missing values.

2. **Data Preprocessing**

   - Select relevant features and target variable.
   - Split the data into training and test sets.
   - Apply SMOTE to handle class imbalance.
   - Scale features using `StandardScaler`.

3. **Model Training and Evaluation**

   - Train and evaluate three models:
     - **Random Forest Classifier**
     - **Gradient Boosting Classifier**
     - **AdaBoost Classifier**
   - Evaluate models using cross-validation, ROC-AUC score, and confusion matrix.
   - Save the trained models to the `models/` directory.

4. **Model Saving and Reporting**

   - Save evaluation reports for each model to the `reports/` directory.

## Results

### Model Evaluation Summary

**1. AdaBoost**:

- **Classification Report:**
  - **Not Hazardous:** Precision: 0.99, Recall: 0.71, F1-Score: 0.82
  - **Hazardous:** Precision: 0.32, Recall: 0.96, F1-Score: 0.48
  - **Overall Accuracy:** 0.74
  - **Macro Average:** Precision: 0.66, Recall: 0.83, F1-Score: 0.65
  - **Weighted Average:** Precision: 0.91, Recall: 0.74, F1-Score: 0.78
- **ROC-AUC Score:** 0.878
- **Confusion Matrix:** [[62478, 26025], [550, 12399]]

**2. Gradient Boosting**:

- **Classification Report:**
  - **Not Hazardous:** Precision: 0.99, Recall: 0.72, F1-Score: 0.83
  - **Hazardous:** Precision: 0.33, Recall: 0.95, F1-Score: 0.49
  - **Overall Accuracy:** 0.75
  - **Macro Average:** Precision: 0.66, Recall: 0.84, F1-Score: 0.66
  - **Weighted Average:** Precision: 0.91, Recall: 0.75, F1-Score: 0.79
- **ROC-AUC Score:** 0.885
- **Confusion Matrix:** [[63699, 24804], [585, 12364]]

**3. Random Forest**:

- **Classification Report:**
  - **Not Hazardous:** Precision: 0.95, Recall: 0.95, F1-Score: 0.95
  - **Hazardous:** Precision: 0.66, Recall: 0.65, F1-Score: 0.65
  - **Overall Accuracy:** 0.91
  - **Macro Average:** Precision: 0.80, Recall: 0.80, F1-Score: 0.80
  - **Weighted Average:** Precision: 0.91, Recall: 0.91, F1-Score: 0.91
- **ROC-AUC Score:** 0.944
- **Confusion Matrix:** [[84158, 4345], [4593, 8356]]

#### **Summary**

- Random Forest achieved the highest ROC-AUC score and overall accuracy, indicating the best overall performance among the models.
- Gradient Boosting and AdaBoost showed lower ROC-AUC scores and overall accuracy compared to Random Forest, with Gradient Boosting performing slightly better than AdaBoost in terms of accuracy and ROC-AUC.
- All models demonstrated a strong ability to identify "Not Hazardous" NEOs, but the identification of "Hazardous" NEOs varied, with higher recall but lower precision in some models.
