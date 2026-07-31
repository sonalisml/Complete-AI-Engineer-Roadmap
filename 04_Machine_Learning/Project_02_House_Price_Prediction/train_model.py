
# ==========================================================
# Import Libraries
# ==========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)
# ==========================================================
# Load Dataset
# ==========================================================
df = pd.read_csv("Housing.csv")
# ==========================================================
# EDA
# ==========================================================
print("\nFirst Five Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nShape of Dataset")
print(df.shape)

print("\nStatistical Summary")
print(df.describe())

print("\nMissing values")
print(df.isnull().sum())
# ==========================================================
# Numerical column and categorical column
# ==========================================================
print("\nCategorical columns are:")
print(df.select_dtypes(include="object").columns)
print("\nNumerical coumns are:")
print(df.select_dtypes(include =["int64", "float64"]).columns)

#Fill the null values if any and encode the values 
df = pd.get_dummies(df, columns=["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"])
df.columns
#ASSIGN FEATURES
X=  df.drop("price", axis = 1)
y = df["price"]

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# ==========================================================
# Grid Search
# ==========================================================

model = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 15]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# ==========================================================
# Performance Metrics
# ==========================================================

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("Mean Squared Error :", mse)
print("Root Mean Squared Error :", rmse)
print("R² Score :", r2)

scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("Cross Validation Scores")
print(scores)

print("Average R² Score :", scores.mean())

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(best_model, "best_model.pkl")

print("Model Saved Successfully")
loaded_model = joblib.load("best_model.pkl")
loaded_predictions = loaded_model.predict(X_test)

results = pd.DataFrame({
    "Model":["RandomForestRgerssor"],
    "mse":[mse],
    "rmse":[rmse]
   }
)

print(results)