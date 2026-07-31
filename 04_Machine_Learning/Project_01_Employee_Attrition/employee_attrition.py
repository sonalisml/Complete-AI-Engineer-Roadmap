
# ==========================================================
# Employee Attrition Prediction using Machine Learning
# Author: Dr. Sonali Samal
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ==========================================================
# Load Dataset
# ==========================================================
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("\nFirst Five Rows")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe(include="all"))

# ==========================================================
# EDA
# ==========================================================
plt.figure(figsize=(6,4))
sns.countplot(x="Attrition", data=df)
plt.title("Employee Attrition Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
corr = df.select_dtypes(include=["int64","float64"]).corr()
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ==========================================================
# Encoding
# ==========================================================
label_cols = df.select_dtypes(include="object").columns

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# ==========================================================
# Features & Target
# ==========================================================
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# ==========================================================
# Train-Test Split
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ==========================================================
# Models
# ==========================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42)
}

results = []

print("\n================ MODEL PERFORMANCE ================\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    cv = cross_val_score(model, X, y, cv=5).mean()

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    results.append([name, acc, cv, auc])

    print("="*60)
    print(name)
    print("="*60)
    print("Accuracy :", round(acc,4))
    print("CV Score :", round(cv,4))
    print("ROC AUC  :", round(auc,4))
    print(classification_report(y_test,pred))

best_df = pd.DataFrame(results,
                       columns=["Model","Accuracy","CV Score","ROC AUC"])

print("\nOverall Performance")
print(best_df.sort_values("Accuracy",ascending=False))

# ==========================================================
# Grid Search on Random Forest
# ==========================================================
param_grid = {
    "n_estimators":[100,200],
    "max_depth":[5,10,None],
    "min_samples_split":[2,5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train,y_train)

best_model = grid.best_estimator_

print("\nBest Parameters")
print(grid.best_params_)

pred = best_model.predict(X_test)

print("\nFinal Accuracy:",accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

# ==========================================================
# Confusion Matrix
# ==========================================================
cm = confusion_matrix(y_test,pred)

ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ==========================================================
# ROC Curve
# ==========================================================
probs = best_model.predict_proba(X_test)[:,1]

fpr,tpr,_ = roc_curve(y_test,probs)

plt.figure(figsize=(6,5))
plt.plot(fpr,tpr,label="Random Forest")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================================
# Feature Importance
# ==========================================================
importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8,8))
importance.head(15).plot(kind="barh")
plt.title("Top 15 Important Features")
plt.tight_layout()
plt.show()

# ==========================================================
# Save Model
# ==========================================================
joblib.dump(best_model,"best_model.pkl")
print("\nModel saved as best_model.pkl")

loaded = joblib.load("best_model.pkl")
print("Saved model loaded successfully.")
