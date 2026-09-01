# ==========================================================
# Ignore Warning Messages
# ==========================================================
import warnings
warnings.filterwarnings("ignore")
# ==========================================================
# Data Manipulation
# ==========================================================
import numpy as np
import matplotlib.pyplot as plt
# ==========================================================
# TensorFlow
# ==========================================================
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    Rescaling,
    RandomFlip,
    RandomRotation,
    RandomZoom
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
print("TensorFlow Version :", tf.__version__)
#====================================================
#Dataset downloading=================================
#====================================================
train_path = "dataset/train"
test_path = "dataset/test"

train_dataset = image_dataset_from_directory(
    train_path,
    image_size=(224,224),
    batch_size=32,
    shuffle=True
)

test_dataset = image_dataset_from_directory(
    test_path,
    image_size=(224,224),
    batch_size=32,
    shuffle=False
)

num_classes = train_dataset.class_names
print(num_classes)
# ==========================================================
# step 5- Image Normalization
# ==========================================================
normalization_layer = Rescaling(scale = 1./255)
print("Normalizationlayer created successfully")
# ==========================================================
# Step 6-Data Augmentataion
# ==========================================================
data_augmentataion = Sequential([
    RandomFlip("Horizontal"),
    RandomRotation(0.10),
    RandomZoom(0.10)
])
print("Data augmentataion layer created successfully")
# ==========================================================
# Step 7-Optimize Dataset Pipeline
# ==========================================================
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(
    buffer_size = AUTOTUNE
)
test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)
print("="*60)
print("Dataset Pipeline Optimized Successfully")
print("="*60)
# ==========================================================
# Step 8-Building CNN model
# ==========================================================
model = Sequential([
    normalization_layer,
    data_augmentataion,
    Conv2D(filters= 32, kernel_size =(3,3), activation = "relu", padding="same"),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters= 32, kernel_size =(3,3), activation = "relu", padding="same"),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters=128,kernel_size=(3,3), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(units =128, activation = "relu"),
    Dense(units =1, activation = "sigmoid")
])
print("\nCNN model created successfully")
# ==========================================================
# Step 9-Compiling the CNN model
# ==========================================================
model.compile(
    loss = "binary_crossentropy",
    optimizer="adam",
    metrics =["accuracy"]
)
print("\nModel Compiled succesfully")
early_stop = EarlyStopping(
    monitor = "val_loss",
    patience = 10,
    restore_best_weights = True
)
history = model.fit(
    train_dataset,
    validation_data = test_dataset,
    epochs = 10,
    callbacks = [early_stop],
    verbose = 1
)
history.history.keys()
# ==========================================================
# Evaluate Model
# ==========================================================
predictions = model.predict(test_dataset)
print(predictions[:10])
predicted_class = (predictions > 0.5).astype("int")
# ==========================================================
# Get Actual Labels
# ==========================================================
y_true = []
for images, labels in test_dataset:
    
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
print("Actual labels")
print(y_true[:10])

cm = confusion_matrix(y_true, predicted_class)
ConfusionMatrixDisplay(Confusion_matrix=cm).plot(cmap="Blues")
plt.title("Confusion matrix")
cr = classification_report(y_true, predicted_class, target_names = class_names)

# ==========================================================
# Accuracy Graph
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
# ==========================================================
# Loss Graph
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
# ==========================================================
# Save Model
# ==========================================================

model.save(
    "cat_dog_classifier.keras"
)
print("="*60)
print("Model Saved Successfully")
print("="*60)
# ==========================================================
# Load Saved Model
# ==========================================================
from tensorflow.keras.models import load_model
loaded_model = load_model(
    "cat_dog_classifier.keras"
)
print("Model Loaded Successfully")
# ==========================================================
# Predict New Image
# ==========================================================

from tensorflow.keras.preprocessing import image
img_path = "predict_images/my_cat.jpg"
img = image.load_img(
    img_path,
    target_size=(224,224)
)
img_array = image.img_to_array(img)
img_array = np.expand_dims(
    img_array,
    axis=0
)
prediction = loaded_model.predict(
    img_array
)
print(prediction)

# ==========================================================
# Display Prediction
# ==========================================================

if prediction[0][0] > 0.5:

    print("Prediction : Dog 🐶")

else:

    print("Prediction : Cat 🐱")