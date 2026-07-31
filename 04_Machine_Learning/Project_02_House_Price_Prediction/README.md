# 🏠 Housing Price Prediction using Random Forest Regression

A Machine Learning project that predicts house prices based on various housing features using the **Random Forest Regression** algorithm. This project demonstrates the complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, evaluation, and model persistence.

---

## 📌 Project Overview

The objective of this project is to develop a regression model capable of estimating house prices from multiple housing attributes such as area, number of bedrooms, bathrooms, parking availability, furnishing status, and other amenities.

The project includes:

- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Correlation Analysis
- Model Training using Random Forest Regressor
- Hyperparameter Optimization using GridSearchCV
- Cross Validation
- Performance Evaluation
- Saving and Loading the trained model using Joblib

---

## 📂 Dataset

Dataset Source:

https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction

Dataset File:

```
Housing.csv
```

---

## 📊 Features

The dataset contains the following features:

| Feature | Description |
|----------|-------------|
| price | House Price (Target Variable) |
| area | Area of the house |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| stories | Number of floors |
| mainroad | Connected to main road |
| guestroom | Guest room availability |
| basement | Basement availability |
| hotwaterheating | Hot water heating |
| airconditioning | Air conditioning |
| parking | Number of parking spaces |
| prefarea | Preferred residential area |
| furnishingstatus | Furnishing status |

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📈 Exploratory Data Analysis

The following visualizations are performed:

- Distribution of House Prices
- Area vs Price Scatter Plot
- Correlation Heatmap

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Missing value inspection
- One-Hot Encoding for categorical variables
- Feature selection
- Train-Test Split

---

## 🤖 Machine Learning Model

Algorithm:

```
Random Forest Regressor
```

Hyperparameter tuning:

- GridSearchCV
- 5-Fold Cross Validation

Hyperparameters optimized:

- Number of Trees
- Maximum Tree Depth

---

## 📏 Performance Metrics

The model is evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Cross Validation Score

---

## 📁 Project Structure

```
Housing-Price-Prediction/
│
├── Housing.csv
├── housing_price_prediction.py
├── housing_price_prediction.ipynb
├── best_model.pkl
├── requirements.txt
├── README.md
└── images/
      ├── histogram.png
      ├── scatter_plot.png
      └── correlation_heatmap.png
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Housing-Price-Prediction.git
```

Move into the project folder

```bash
cd Housing-Price-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Execute

```bash
python housing_price_prediction.py
```

or open

```
housing_price_prediction.ipynb
```

using Jupyter Notebook.

---

## 💾 Save the Model

The trained model is automatically saved as

```
best_model.pkl
```

using Joblib.

Load the model:

```python
import joblib

model = joblib.load("best_model.pkl")
```

---

## 📊 Workflow

```
Load Dataset
      │
      ▼
Data Cleaning
      │
      ▼
EDA
      │
      ▼
Feature Engineering
      │
      ▼
Train-Test Split
      │
      ▼
GridSearchCV
      │
      ▼
Random Forest Regression
      │
      ▼
Model Evaluation
      │
      ▼
Save Model
```

---

## 📚 Python Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

## 📌 Future Improvements

- Feature Scaling Comparison
- XGBoost Regression
- LightGBM Regression
- CatBoost Regression
- Flask Web Application
- Streamlit Deployment
- SHAP Explainability
- Feature Importance Visualization

---

## 👨‍💻 Author

**Dr. Sonali Samal**

---

## ⭐ If you found this project useful

Give the repository a ⭐ on GitHub.