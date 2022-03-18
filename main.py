import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers,models, losses

train = pd.read_csv("sign_mnist_train.csv") # Train Set
test = pd.read_csv("sign_mnist_test.csv") # Test set

X_train = train.loc[:, "pixel1":].to_numpy().astype(np.float32) / 255.0 # Get Features for Training
y_train = train.loc[:, "label"].to_numpy().astype(np.int32)  # Get Labels for Training

X_test = test.loc[:, "pixel1":].to_numpy().astype(np.float32) / 255.0 # Features for Testing
y_test = test.loc[:, "label"].to_numpy().astype(np.int32) # Labels for Testing

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)

height = 28 # MNIST requirements
width = 28 # MNIST requirements
channel = 1

convLayerFMaps1 = 32 # Build first convolutional layer
convLayerKSize1 = 3
convLayerStride1 = 1
convLayerPad1 = 'same'

convLayerFMaps2 = 64 # Build second convolutional layer
convLayerKSize2 = 3
convLayerStride2 = 2
convLayerPad2 = 'same'

model = models.Sequential()
model.add(layers.Conv2D(filters=convLayerFMaps1, kernel_size=convLayerKSize1, strides= convLayerStride1,# First Conv Layer with
                        padding=convLayerPad1, activation='relu',input_shape=(height,width,channel)))   # RELU activation
model.add(layers.Conv2D(filters=convLayerFMaps2, kernel_size=convLayerKSize2, strides= convLayerStride2,# Second Conv Layer with
                        padding=convLayerPad2, activation='relu'))                                      # RELU activation
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding= 'valid')) # Use valid padding so no zero padding outsize edges
model.add(layers.Flatten()) # Flatten the input
model.add(layers.Dense(units=64, activation='relu')) # Input units
model.add(layers.Dense(units=25)) # Output units
model.add(layers.Softmax()) # Softmax activation

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(X_test, y_test)