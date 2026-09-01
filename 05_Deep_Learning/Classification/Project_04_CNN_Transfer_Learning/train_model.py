import warnings
warnings.filterwarnings("ignore")
#============================
#  Data Manipulation
#============================
import numpy as np
import matplotlib.pyplot as plt
#============================
#  Tensorflow
#============================
import tensorflow as tf
print("tensorflow version", tf._version_)
#===========================
#  DatasetLoading
#===========================
from tensorflow.keras.utils import image_dataset_from_directory
#===========================
#  Pretrained model
#===========================
from tensorflow.keras.applications import VGG16
#===========================
#  DeepLearning layers
#===========================
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Flatten)
from tensorflow.keras.callbacks import callbacks 
#===========================
#  Load dataset
#===========================
train_dataset = 'dataset/train'
validation_dataset = 'dataset/validation'
test_dataset = 'dataset/test'
train_dataset = image_dataset_from_directory(
    train_dataset,
    input_size = (224,224),
    batch_size = 32,
    shuffle = True
)
validation_dataset = image_dataset_from_directory(
    validation_dataset,
    image_size = (224,224),
    batch_size = 32,
    shuffle = True
)
test_dataset = image_dataset_from_directory(
    test_dataset,
    input_size = (224,224),
    batch_size = 32,
    shuffle = False
)
print("="*60)
print("Datasets Loaded Successfully")
print("="*60)

#======================
# Class names
#=====================
class_names = train_dataset.class_names
print(class_names)
print(len(class_names))
#======================
# Load pretrained model
#=====================
base_model = VGG16(
    weights = 'imagenet',
    include_top =False,
    input_shape = (224,224,3)
)
print("="*60)
print("VGG16 model loaded successfully")
#======================
# Base model freeze
#=====================
base_model.trainable = False
print("\nbase model frozen")
# ==========================================================
# Display Base Model Summary
# ==========================================================
print("="*60)
print("VGG16 Summary")
print("="*60)
base_model.summary()
# ==========================================================
# Model build
# ==========================================================
model = Sequential([
    base_model,
    Flatten(),
    Dense(units = 128, activation = "relu"),
    Dropout(rate = 0.5),
    Dense(units = 64, activation = "relu"),
    Dropout(rate = 0.2),
    Dense(units = 1, activation = "sigmoid")
])
# ==========================================================
# Display Model Summary
# ==========================================================
print("="*60)
print("Transfer Learning Model Summary")
print("="*60)
model.summary()
# ==========================================================
# Compile Model
# ==========================================================
model.compile(
    optimizer ="adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)
print("Model Compiled Successfully")
early_stop = EarlyStopping(
    monitor="val_loss",
    patience = 10,
    restore_best_weights = True
)

history = model.fit(
    train_datatset,
    validation_dataset,
    epochs = 5,
    call_backs = [early_stop],
    verbose =1

)
#=================================
#Model Evaluation
#================================
loss, accuracy = model.evaluate(
    test_dataset,
    verbose =0
)
print("Test Result")
predictions = model.predict(test_dataset)
y_pred = (predictions>0.5).astype(int)
print(predicted_class[:10])
#======================
#Confusion matrix
#======================
  # ==========================================================
  # Get Actual Labels
  # ==========================================================
y_true = []
for images, labels in test_dataset:
    
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
print("Actual labels")
print(y_true[:10])
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="blue")
cr = classification_report(y_true, y_pred)
print(cr)
#======================
#Accuracy Graphs
#======================
history.history.keys()
plt.figure(fig_size = (8,5))
plt.plot(
    history.history['accuracy'],
    label = "training_accuracy"
)
plt.plot(
    history.history['val_accuracy'],
    label = "val_accuracy"
)
plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.grid(True)

plt.show()
# ==========================================================
# Loss Plot
# ==========================================================

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

plt.grid(True)

plt.show()
#======================
#Model saving
#======================
model.save("vgg16_cat_dog_classifier.kera")
# ==========================================================
# Load Model
# ==========================================================

from tensorflow.keras.models import load_model
loaded_model = load_model("vgg16_cat_dog_classifier.keras")
print("Model Loaded Successfully")