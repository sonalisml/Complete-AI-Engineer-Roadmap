import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense)  
from tensorflow.keras.models import Sequential

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# ==========================================================
# Display First Image
# ==========================================================
plt.imshow(X_train[0], cmap="gray")
plt.title(y_train[0])
plt.axis("off")
plt.show()
# ==========================================================
# Normalizing images
# ==========================================================
X_train = X_train.astype("float32")/255.0
y_train = y_train.astype("float32")/255.0
print("normalizing completed")
print("min pixel value", X_train.min())
print("max pixel value", X_train.max())
# ==========================================================
# Reshaping
# ==========================================================
X_train = X_train.reshape(
    -1,
    28,
    28,
    1
)
X_test = X_test.reshape(
    -1,
    28,
    28,
    1
)
print(X_train.shape)
print(X_test.shape)
#=============
#=Categorical
#=============
y_train = to_categorical(
    y_train,
    num_classes = 10
)
y_test = to_categorical(
    y_test,
    num_classes = 10
)
print(y_train.shape)
print(y_test.shape)
#==============
#model creation
#==============
model = Sequential()

model.add(
    Conv2D(
        filters =32,
        kernel_size = (3,3),
        activation = "relu" ,
        input_shape= (28,28,1),

    )
)
model.add(
    MaxPooling2D(pool_size=(2,2)
    )
)

model.add(
    Conv2D(
        filters =32,
        kernel_size =(3,3),
        activation = "relu",
    )
)
model.add(
    MaxPooling2D(pool_size=(2,2))
)

model.add(Flatten())
#cnn-ann
model.add(
    Dense(
        128,
        activation = "relu"
    )
)
model.add(
    Dense(
        10,
        activation = "softmax"
    )
)
model.summary()
#=========
#Compile
#=========
model.compile(
    optimizer ="adam",
    loss ="binary_crossentropy",
    metrics = "accuracy"
)

#=========
#Train
#=========
history = model.fit(
    X_train,

    y_train,

    validation_split=0.20,

    epochs=10,

    batch_size=64,

    verbose=1
)
#============
#Predictions 
#============
y_prob_pred = model.predict(X_test)
print(y_prob_pred.shape)

y_pred = np.argmax(
    y_prob_pred,
    axis=1
)
y_true = np.argmax(
    y_test,
    axis = 1
)

accuracy = accuracy_score(
    y_pred,
    y_true
)
cm = confusion_matrix(
    y_pred,
    y_true
)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="blues")
plt.title("Confusion Matrix")
plt.show()

cr = classification_report(y_pred,y_true)

plt.figure(fig_size=(8,8))
plt.plot(
    history.history("accuracy"),
    label ="Training_accuracy"
)
plt.plot(
    history.history("accuracy"),
    label = "validation_accuracy"
)

plt.xlabel("epoch")
plt.ylabel("accuracy")

plt.plot(
    history.history("loss"),
    label = "Training_loss"
)
plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("CNN Loss")

plt.legend()

plt.show()
# ==========================================================
# Predictions
# ==========================================================
for i in range(10):
    print(f"pred: {y_pred[i]}\nActual:{y_true[i]}",)
# ==========================================================
# Save CNN Model
# ==========================================================

model.save(
    "mnist_cnn_model.keras"
)

print("CNN Model Saved Successfully")

loaded_model = load_model("mnist_cnn_model.keras")
predictions = loaded_model.predict(X_test[0].reshape(1,28,28,1))
digit = np.argmax(predictions)
print("predicted digit", digit)