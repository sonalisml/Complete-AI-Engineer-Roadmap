#=============================
#Step 1:- Importing Libraraies
import numpy as np
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
#=============================
#Step 2:- Load the data
#=============================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
#=============================
#Step 3:- EDA
#=============================
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.columns)
print("Categorical columns are:")
print(df.select_dtypes(include="object").columns)
print("Numerical columns are:")
print(df.select_dtypes(include=["int64", "float64"]).columns)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors ="coerce")
#=============================
#Step 4:- Create Features and Target
#=============================
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])
X = df.drop(columns = ["customerID","Churn"])
y = df["Churn"]

Categorical_features= X.select_dtypes(include="object").columns
Numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
#=============================
##STEP 5- Train test split
#=============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size =0.2,
    random_state =42,
    stratify =y
)
#=============================
#Step 6:- Pipelining
#=============================
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("Scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("OneHotEncoder", OneHotEncoder(handle_unknown = "ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, Numerical_features),
    ("cat", categorical_pipeline, Categorical_features)
])

experiments = {
    "LogisticRegression":{

        "model": LogisticRegression(max_iter=1000, random_state =42),

       "param_grid": {"classifier__C":[0.1,0.2,1]}

    },
    "DecisionTreeClassifier":{
        "model": DecisionTreeClassifier(random_state =42),

         "param_grid": { "classifier__max_depth": [3, 5, 10, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4]}
            
    },
    "RandomForestClassifier":{
        "model":RandomForestClassifier(random_state=42),
         "param_grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        }
    },
       
    }
    
results=[]
for name, experiment in experiments.items():
    model = experiment["model"]
    param_grid = experiment["param_grid"]

    pipeline = Pipeline([("preprocessor", preprocessor),("classifier", model)])
#=============================

    pipeline.fit(X_train, y_train)
#=============================
#Step 7:- Model Evaluation
#=============================
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test,predictions)
    #=============================
#Step 09:- ROC AND AUC Curve
#=============================
    y_probability = pipeline.predict_proba(X_test)[:,1]
    fpr, tpr, threshold = roc_curve(
          y_test,
          y_probability)
    auc_score = roc_auc_score(
      y_test,
      y_probability
      )
#=============================
#Step 8:- Cross Validation
#=============================
    scores = cross_val_score(
               pipeline,
               X,
               y,
               cv=5,
               scoring = "accuracy"
             )
    Acc_cross_validation= scores.mean()
    results.append({
       "Model":name,
       "Accuracy":accuracy,
       "ROC-AUC": auc_score,
       "Cross_val": Acc_cross_validation
       }
    )
#=============================
#Storing Results
#=============================
results = pd.DataFrame(results)
print("\n")
print("="*60)
print("MODEL COMPARISON")
print("="*60)
print(results)
#=============================
#Storing Best model
#=============================
best = results.sort_values(
       by="ROC-AUC",
       ascending = False
)
print("Best Model so far is", best.iloc[0])
Best_model_name = best.iloc[0]["Model"]#Takes only the model name
print(Best_model_name)
#=====================
#Extracting from experiments
#=====================
Best_experiment = experiments[Best_model_name]
Best_model = Best_experiment["model"]
Best_model_params= Best_experiment["param_grid"]

pipeline = Pipeline([("preprocessor", preprocessor),("classifier", Best_model)])
grid_search = GridSearchCV(
    estimator = pipeline,
    param_grid = Best_model_params,
    cv = 5,
    scoring ="accuracy"
)
grid_search.fit(X_train, y_train)
final_params= grid_search.best_params_
final_model = grid_search.best_estimator_
print("\n")
print("="*60)
print("Final model is:", final_model)
