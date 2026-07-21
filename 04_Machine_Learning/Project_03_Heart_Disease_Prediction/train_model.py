# ============================================================
# STEP 1 : IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#PIPELINE(Import)#################
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
##################################
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import joblib
# ============================================================
# STEP 2 : LOAD DATASET
# ============================================================
df = pd.read_csv("heart_disease_uci.csv")
# ============================================================
# STEP 3 : EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print(df.shape)
print(df.head())
print(df.isnull().sum())
print(df.columns)
# ============================================================
# STEP 4 : CREATE TARGET
# ============================================================
df["target"] = df["num"].apply(lambda x:0 if x<=0 else 1)
df.drop(columns=["id","num","dataset"], inplace=True)
X =df.drop(columns=["target"])
y = df["target"]

from sklearn.preprocessing import OneHotEncoder, StandardScaler
#STEP 5- AUTOMATIC NUM AND CAT FEATURES
numerical_features = X.select_dtypes(include = ["int64","float64"]).columns
categorical_features = X.select_dtypes(include = "object").columns

##STEP 6- Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size =0.2,
    random_state =42,
    stratify =y
)
# =========================
# 7. PREPROCESSING PIPELINE
# =========================
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
category_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy ="most_frequent")),
    ("OneHotEncoder", OneHotEncoder(handle_unknown ="ignore"))
                              
])

#Combine Processing
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", category_pipeline, categorical_features)
])

# =========================
# 8. COMPLETE ML PIPELINE
# =========================
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# =========
# 9. HYPERPARAMETER TUNING
# ===============
param_grid = {
    "classifier__C" : [0.1,0.01,1,10]
}
grid_search = GridSearchCV(
    estimator = pipeline,
    param_grid = param_grid,
    cv = 5,
    scoring = "accuracy"
)
grid_search.fit(X_train, y_train)
grid_search.best_params_
best_model = grid_search.best_estimator_

# =========
# 10. Model Evaluation
# ===============
print("Predictions:")
predictions = best_model.predict(X_test)
print("Accuracy score is:")
print(accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# =========
# 11. Cross validation
# ===============
scores = cross_val_score(
    best_model,
    X,
    y,
    cv = 5,
    scoring = "accuracy"
)
print("Avg accuracy after applying crossvalidation is:", scores.mean())
# =========================
# 12. ROC CURVE & AUC
# =========================
#ROC curve
y_probability = best_model.predict_proba(X_test)[:,1]

fpr,tpr,thresholds = roc_curve(
    y_test,
    y_probability
)
#auc roc curve
auc_score = roc_auc_score(
    y_test,
    y_probability
)
# =========================
# 13. Feature Importance
# =========================
print("\nAUC Score :", auc_score)

# =========================
# 14. Save the model
# =========================
joblib.dump(best_model, "Heart_disease_model.pkl")
print("Model saved successfully")
loaded_model = joblib.load("Heart_disease_model.pkl")
print("model loaded succseesfully")
# =========================
print(df["sex"].unique())
print(df["cp"].unique())
print(df["fbs"].unique())
print(df["restecg"].unique())
print(df["exang"].unique())
print(df["slope"].unique())
print(df["thal"].unique())
print(df["target"].value_counts())
print(classification_report(y_test, predictions))
print(pd.Series(predictions).value_counts())