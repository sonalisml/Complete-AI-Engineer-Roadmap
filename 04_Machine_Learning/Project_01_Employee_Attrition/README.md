# 👨‍💼 Employee Attrition Prediction using Machine Learning

A Machine Learning project that predicts whether an employee is likely to leave an organization based on demographic, job-related, and workplace attributes. This project demonstrates the complete machine learning pipeline, from data preprocessing and exploratory data analysis (EDA) to model training, hyperparameter tuning, evaluation, and model persistence.

---

## 📌 Project Overview

Employee attrition is a major challenge for organizations because replacing skilled employees is expensive and time-consuming. This project aims to build a predictive model that helps HR departments identify employees who are at risk of leaving the organization.

The project includes:

- Data Loading
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Encoding
- Train-Test Split
- Multiple Machine Learning Models
- Hyperparameter Tuning using GridSearchCV
- Cross Validation
- Model Evaluation
- Feature Importance Analysis
- Model Saving using Joblib

---

## 🎯 Problem Statement

Predict whether an employee will leave the company (**Attrition = Yes**) or stay (**Attrition = No**) based on various employee-related features.

This is a **Binary Classification** problem.

---

## 📂 Dataset

**Dataset Name**

IBM HR Analytics Employee Attrition & Performance

**Source**

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

**Dataset File**

```
WA_Fn-UseC_-HR-Employee-Attrition.csv
```

---

## 📊 Dataset Information

- Total Records: **1470**
- Features: **35**
- Target Variable: **Attrition**

The dataset contains employee information such as:

- Age
- Business Travel
- Daily Rate
- Department
- Distance From Home
- Education
- Environment Satisfaction
- Gender
- Job Involvement
- Job Level
- Job Role
- Monthly Income
- OverTime
- Performance Rating
- Stock Option Level
- Total Working Years
- Work-Life Balance
- Years at Company
- Years Since Last Promotion
- Years With Current Manager

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📈 Exploratory Data Analysis

The following analyses are performed:

- Dataset Overview
- Missing Value Analysis
- Statistical Summary
- Employee Attrition Distribution
- Age Distribution
- Correlation Heatmap

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Missing Value Check
- Label Encoding of Categorical Variables
- Feature Selection
- Train-Test Split

---

## 🤖 Machine Learning Models

The following classification algorithms are implemented:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

The best-performing model is further optimized using **GridSearchCV**.

---

## 🔍 Hyperparameter Tuning

GridSearchCV is applied to the Random Forest Classifier.

Optimized Parameters include:

- Number of Trees (`n_estimators`)
- Maximum Tree Depth (`max_depth`)
- Minimum Samples Split (`min_samples_split`)

---

## 📏 Model Evaluation

The models are evaluated using:

- Accuracy Score
- Cross Validation Score
- ROC-AUC Score
- Classification Report
- Confusion Matrix

---

## 📉 Visualizations

The project generates the following visualizations:

- Employee Attrition Count Plot
- Age Distribution Histogram
- Correlation Heatmap
- Confusion Matrix
- ROC Curve
- Top 15 Feature Importance Plot

---

## 💾 Save and Load Model

The trained model is saved using Joblib.

```python
import joblib

joblib.dump(best_model, "best_model.pkl")

model = joblib.load("best_model.pkl")
```

---

## 📁 Project Structure

```
Employee-Attrition-Prediction/
│
├── dataset/
│     └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── employee_attrition.py
├── employee_attrition.ipynb
├── best_model.pkl
├── requirements.txt
├── README.md
├── images/
│     ├── attrition_distribution.png
│     ├── age_distribution.png
│     ├── correlation_heatmap.png
│     ├── confusion_matrix.png
│     ├── roc_curve.png
│     └── feature_importance.png
│
└── .gitignore
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Employee-Attrition-Prediction.git
```

Move to the project folder

```bash
cd Employee-Attrition-Prediction
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Execute

```bash
python employee_attrition.py
```

or open

```
employee_attrition.ipynb
```

using Jupyter Notebook.

---

## 📊 Machine Learning Workflow

```
Load Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Encoding
      │
      ▼
Train-Test Split
      │
      ▼
Model Training
      │
      ▼
Cross Validation
      │
      ▼
GridSearchCV
      │
      ▼
Model Evaluation
      │
      ▼
Feature Importance
      │
      ▼
Save Best Model
```

---

## 📦 Required Python Libraries

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

---

## 🔮 Future Improvements

- Handle class imbalance using SMOTE
- Use One-Hot Encoding with Pipelines
- Compare additional models (XGBoost, LightGBM, CatBoost)
- Perform SHAP-based feature importance analysis
- Deploy the model using Flask or Streamlit
- Create an interactive dashboard for HR analytics

---

## 👨‍💻 Author

**Dr. Sonali Samal**

## ⭐ Support

If you found this project useful, please consider giving the repository a **⭐ Star** on GitHub.