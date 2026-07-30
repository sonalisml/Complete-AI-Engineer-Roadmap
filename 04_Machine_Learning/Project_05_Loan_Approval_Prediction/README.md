# Loan Approval Prediction

## Project Overview

This project predicts whether a loan application will be approved based on applicant information such as income, education, employment status, credit history, and property area.

The objective is to automate loan approval decisions using supervised machine learning classification algorithms.

---

## Dataset

- Source: Kaggle - Loan Prediction Dataset
- Records: 614
- Features: 13
- Target:
  - Loan_Status
    - Y → Approved
    - N → Rejected

---

## Machine Learning Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Missing Value Handling
5. Feature Encoding
6. Feature Scaling
7. Pipeline Construction
8. Model Comparison
9. Hyperparameter Tuning using GridSearchCV
10. Final Model Selection
11. Model Saving using Joblib

---

## Models Compared

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## Best Model

Model:
(Logistic Regression / Random Forest)

Accuracy:
(Your Accuracy)

ROC-AUC:
(Your ROC-AUC)

Best Hyperparameters:

```python
{
    ...
}
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib

---

## Project Structure

Project_05_Loan_Approval_Prediction/

├── train_model.py

├── app.py

├── loan.csv

├── Loan_Prediction_Model.pkl

├── requirements.txt

└── README.md

---

## How to Run

Clone the repository

```bash
git clone <repository_link>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python train_model.py
```

---

## Results

| Model | Accuracy | ROC-AUC |
|--------|----------|----------|
| Logistic Regression | 0.8618 | 0.8523 |
| Decision Tree | 0.7561 | 0.7144 |
| Random Forest | 0.8211 | 0.7808 |

---

## Future Improvements

- Deploy using Streamlit
- Add XGBoost and LightGBM
- Perform Feature Engineering
- Handle Class Imbalance
- Use SHAP for Explainability

---

## Author

Dr. Sonali Samal

AI Engineer Roadmap Project 05