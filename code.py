# exp 1 build a simple artificial neural network with 1 layer, with 1 neuron and the input shape equal to 1. feed some data use the equation y=5x-3 so where x = -2 , y = -4 and train the network.
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
# Define the training data
x_train = np.array([-2, 0, 1, 3, 5, -1, 2, 4,6, -3, -5, 7, -4, 8, -6, 9, -7, -8, -9, 10]) #Input x
y_train = np.array([-4, -3, 2, 12, 22, -8, 7,17, 27, -13, -23, 32, -18, 37, -33, 42, -38, -43, -48, 47]) # Expected output y
# Create the neural network model
model = Sequential()
model.add(Dense(units=1, input_shape=(1,))) #One layer with one neuron
# Compile the model
model.compile(optimizer='sgd',loss='mean_squared_error')
# Train the model
model.fit(x_train, y_train, epochs=2000)
# Define the x values for testing

x_test = np.array([-2, 0, 1, 3, 15])
# Predict the corresponding y values using the trained model
y_test = model.predict(x_test)
# Calculate the expected y values based on the equation y = 5x - 3
y_expected = 5 * x_test - 3
# Print the predicted and expected outputs for comparison
for x, y_pred, y_exp in zip(x_test, y_test,y_expected):
  print("Input x =", x)
  print("Predicted output y =", y_pred[0])
  print("Expected output y =", y_exp)
  print()


# exp 2 #using tensorflow build a network with a single hidden layer and at least 300,000 trainable parameters
import tensorflow as tf

# Input size for grayscale images (28x28 pixels)
input_size = 28 * 28

# Output size for 10-class classification
output_size = 10

# Define the number of units in the hidden layer
hidden_units = 500

# Create a sequential model
model = tf.keras.models.Sequential()

# Add the hidden layer with the specified number of units
model.add(tf.keras.layers.Dense(hidden_units, activation='relu', input_shape=(input_size,)))

# Add the output layer
model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

# Print the model summary
model.summary()

''' EXP 3 using tensorflow build 3 networks each with at least 10 hidden layers such that:
    the first model has fewer than 10 nodes per layer
    the second model has between 10-50 nodes per layer
     the third model has between 50-100 nodes per layer '''

import tensorflow as tf

# Input size for grayscale images (28x28 pixels)
input_size = 28 * 28

# Output size for 10-class classification
output_size = 10

# Model 1: Fewer than 10 nodes per layer
model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.InputLayer(input_shape=(input_size,)))

# Add at least 10 hidden layers with fewer than 10 nodes per layer
for i in range(10):
    model1.add(tf.keras.layers.Dense(units=8, activation='relu'))

model1.add(tf.keras.layers.Dense(output_size, activation='softmax'))

# Print the model summary
model1.summary()

# Model 2: Between 10-50 nodes per layer
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.InputLayer(input_shape=(input_size,)))

# Add at least 10 hidden layers with 10-50 nodes per layer
for i in range(10):
    model2.add(tf.keras.layers.Dense(units=30, activation='relu'))

model2.add(tf.keras.layers.Dense(output_size, activation='softmax'))

# Print the model summary
model2.summary()

# Model 3: Between 50-100 nodes per layer
model3 = tf.keras.Sequential()
model3.add(tf.keras.layers.InputLayer(input_shape=(input_size,)))

# Add at least 10 hidden layers with 50-100 nodes per layer
for i in range(10):
    model3.add(tf.keras.layers.Dense(units=80, activation='relu'))

model3.add(tf.keras.layers.Dense(output_size, activation='softmax'))

# Print the model summary
model3.summary()





# exp 4 build  network with at least 3 hidden layers that achieves better than 92% accuracy on validation and test data. you may need to train for more than 10 epochs to achieve this result
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.2, verbose=2)

# Evaluate the model on validation data
validation_loss, validation_accuracy = model.evaluate(x_test, y_test)

print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")





#exp 5 build a network for classification using the built in mnist dataset
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be in the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into a 1D array
    Dense(128, activation='relu'),  # Hidden layer with ReLU activation
    Dense(10, activation='softmax')  # Output layer with softmax activation (10 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Loss function for classification
              metrics=['accuracy'])  # Evaluation metric

# Train the model on the training data
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")




# exp 6 build a network for classification using the built in mnist dataset and use the sigmoid activation function
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the neural network
model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")




#exp 7 build a network for classification using the built in mnist dataset and use the sigmoid activation function and use the categorical cross entropy loss function

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the neural network
model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model with categorical cross-entropy loss
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")




#exp 9 to conduct experiment on obj detection using CNN
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build a convolutional neural network (CNN) model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")



