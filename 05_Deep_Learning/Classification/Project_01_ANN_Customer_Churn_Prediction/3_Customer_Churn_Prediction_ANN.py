#Ignore warning message
#=======================
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Data Manipulation
# -----------------------------
import numpy as np
import pandas as pd
# -----------------------------
# Data Visualization
# -----------------------------
import matplotlib.pyplot as plt
import seaborn as sns
# -----------------------------
# Machine Learning Utilities
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# -----------------------------
# Deep Learning
# -----------------------------
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
# -----------------------------
# Model Evaluation
# -----------------------------
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ==========================================================
# Load Dataset
# ==========================================================
df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully")
# ==========================================================
# Display Dataset- EDA
# ==========================================================
print("="*60)
print("First Five Rows")
print("="*60)

print(df.head())

print("="*60)
print("Dataset information")
print("="*60)

print(df.info())

print("="*60)
print("Dataset shape")
print("="*60)
print(df.shape)

print("="*60)
print("Statistical summary")
print("="*60)
print(df.describe(include ="all"))

print("="*60)
print("Missing values")
print("="*60)
print(df.isnull().sum())

print("="*60)
print("Duplicated Records")
print("="*60)
print(df.duplicated().sum())

print("="*60)
print("Display columns")
print("="*60)
print(df.columns)

print("Categorical columns are")
print(df.select_dtypes(include = ["int64", "float64"]).columns)
print("Numerical columns are")
print(df.select_dtypes(include = "object").columns)

# ==========================================================
# Data Types
# ==========================================================

print("="*60)
print("Data Types")
print("="*60)

print(df.dtypes)
# ==========================================================
# Convert TotalCharges into Numeric
# ==========================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nData type after conversion:")
print(df["TotalCharges"].dtype)

# ==========================================================
# Convert TotalCharges into Numeric
# ==========================================================

# ==========================================================
# Missing Values
# ==========================================================
print("\nMissing Values")
print(df.isnull().sum())
# ==========================================================
# Remove Missing Values
# ==========================================================
df.dropna(inplace=True)
print("\nShape after removing missing values:")
print(df.shape)
# ==========================================================
# Remove customerID
# ==========================================================

df.drop("customerID", axis=1, inplace=True)

print("\nColumns after removing customerID")
print(df.columns)
# ==========================================================
# Final Dataset Check
# ==========================================================

print("="*60)
print("Dataset Ready for Preprocessing")
print("="*60)

print(df.head())

print("\n")

print(df.info())

print("\n")

print(df.shape)
# ==========================================================
# DATA PREPROCESSING
# ==========================================================

print("="*60)
print("Data Preprocessing")
print("="*60)

# ----------------------------------------------------------
# Encode Target Variable (Churn)
# ----------------------------------------------------------

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df["Churn"] = label_encoder.fit_transform(df["Churn"])

print("\nTarget Variable Encoding")
print(df["Churn"].value_counts())

# ----------------------------------------------------------
# One-Hot Encoding for Categorical Variables
# ----------------------------------------------------------

categorical_columns = df.select_dtypes(include="object").columns

print("\nCategorical Columns")
print(categorical_columns)

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

print("\nDataset Shape After Encoding")
print(df.shape)

# ----------------------------------------------------------
# Separate Features and Target
# ----------------------------------------------------------

X = df.drop("Churn", axis=1)

y = df["Churn"]

print("\nFeature Matrix Shape :", X.shape)
print("Target Shape :", y.shape)

# ----------------------------------------------------------
# Train-Test Split
# ----------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Set :", X_train.shape)
print("Testing Set  :", X_test.shape)

# ----------------------------------------------------------
# Feature Scaling
# ----------------------------------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed")

print("\nScaled Training Data Shape :", X_train.shape)
print("Scaled Testing Data Shape  :", X_test.shape)


#====================
#DeepLearning-Build An Model
#===================
model = Sequential()
#=============
#Hidden layer 1
#=============
model.add(
   Dense(
      units =16, 
      activation = "relu",
      input_shape = (X_train.shape[1],),
    )
 )
 model.add(BatchNormalization())
#=============
#Hidden layer 2
#=============
model.add(
    Dense(
        units = 8,
        activation = "relu",

    )
)
model.add(Dropout(0.2))
#=============
#Output layer
#=============
model.add(
    Dense(
        units = 1,
        activation = "sigmoid"
    )
)
print("ANN Model Created Successfully")

#==========
#Compiling model
#==========
model.compile(
    optimizer= "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)
print("model compiled successfully")
model.summary()

#===============
#Fit model
#============
early_stop = EarlyStopping(
    monitor = "val_loss",
    patience =10,
    restore_best_weights= True
)
checkpoint = ModelCheckpoint(
    "best_model.keras",
     monitor = "val_loss",
     save_best_only = True,
     verbose = 1

)
tensorboard = TensorBoard(
    log_dir ="logs",
    histogram_freq=1
)
history = model.fit(
    X_train,
    y_train,
    validation_split= 0.20,
    epochs = 100,
    batch_size= 32,
    callbacks = [early_stop, checkpoint, tensorboard],
    verbose= 1
)
# ==========================================================
# MODEL EVALUATION
# ==========================================================
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)
print("="*60)
print("Test result")
print(loss, accuracy)
#=============
#Predictions, cm, cr, auc
#=============
y_pred_prob = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_pred_prob
)

y_pred = (y_pred_prob > 0.5).astype(int)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)

cr= classification_report(y_test, y_pred)
print(cr)

auc = roc_auc_score(y_test, y_pred_prob)
print(auc)


plt.figure(figsize=(7,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f}"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()
print(history.history.keys())

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.show()

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.show()

# ==========================================================
# SAVE THE TRAINED MODEL
# ==========================================================
# ==========================================================
# LOAD THE MODEL
# ==========================================================
from tensorflow.keras.models import load_model
loaded_model = load_model("best_customer_churn_model.keras")
print("Model Loaded Successfully")
# ==========================================================
# PREDICT USING LOADED MODEL
# ==========================================================
predictions = loaded_model.predict(X_test)
print(predictions[:10])
predicted_class = (predictions > 0.5).astype(int)
print(predicted_class[:10])
# ==========================================================
# COMPARE RESULTS
# ==========================================================
comparison = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": predicted_class.flatten()

})
print(comparison.head(20))
comparison.to_csv(
    "prediction_results.csv",
    index=False
)