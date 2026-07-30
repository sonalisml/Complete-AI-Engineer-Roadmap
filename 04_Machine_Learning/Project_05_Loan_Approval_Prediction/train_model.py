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
#=============================
#Step 2:- Load dataset
df = pd.read_csv("loan.csv")
#=============================
#Step 3:- EDA
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.columns)
print("Before processing category and numeical columns are:")
print(df.select_dtypes(include= "object").columns)
print(df.select_dtypes(include = ["int64", "float64"]).columns)
#=============================
#Step 4:- Preprocessing(convert if any, encode will be done in pipeline)
#=============================
#df["Loan_ID"]= pd.to_numeric(df["Loan_ID"], errors = "coerece")
le = LabelEncoder()
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])
#=============================
#Step 5:- Set Target and features
#=============================
X = df.drop(columns = ["Loan_Status", "Loan_ID"])
y= df["Loan_Status"]
#=============================
#Step 6:- train test split
#=============================
X_train,X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)
#=============================
#Step 7:-  Pipelining(filling null values, encoding part.)
#=============================
Categorical_features = X.select_dtypes(include = "object").columns 
Numerical_features = X.select_dtypes(include = ["int64","float64"]).columns

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("Scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
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

         "param_grid": { 
            "classifier__max_depth": [3, 5, 10, None],
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
    
#=============================
#Step 6:- set up and call the experiments
#=============================
result = []
for name, experiment in experiments.items():
    model = experiment["model"]
    param_grid = experiment["param_grid"]
    pipeline = Pipeline([("preprocessor", preprocessor),("classifier", model)])
    pipeline.fit(X_train, y_train)
#=============================
#Step 7:- Model Evaluation
#=============================
    predictions = pipeline.predict(X_test)
    print("Accuracy score is :")
    accuracy = accuracy_score(y_test, predictions)
    y_probability = pipeline.predict_proba(X_test)[:,1]
    fpr,tpr,threshold = roc_curve(
        y_test,
        y_probability
    )
    auc_score = roc_auc_score(
       y_test,
       y_probability
    )
    result.append(
       {
         "Model": name,
         "accuracy": accuracy,
         "ROC-AUC": auc_score,
       }
    )
   #=============================
#Storing Results
#=============================
results = pd.DataFrame(result)
print("\n")
print("="*60)
print("Model Comparision")
print("="*60)
print(results)
#=============================
#Storing Best model
#=============================
results = results.sort_values(by = "ROC-AUC", ascending = False)
best_model_name = results.iloc[0]["Model"]
   #=================
   #Gridserachcv
   #=================
best_exp = experiments[best_model_name]
best_modelbg= best_exp["model"]
best_modelparams = best_exp["param_grid"]

pipeline = Pipeline([("preprocessor", preprocessor),("classifier", best_modelbg)])
grid_search = GridSearchCV(
        estimator = pipeline,
        param_grid = best_modelparams,
        cv = 5,
        scoring = "accuracy"
   )
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
final_predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, final_predictions)
print("\n")
print("="*60)
print("Final model is:")
print(best_model)
   #=================
joblib.dump(
     best_model,
    "Loan_Prediction_Model.pkl"
)
   #=================