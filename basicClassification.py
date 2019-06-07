#6/6/2019
#Using a tutorial from tensorflow.org
#Running through Anaconda, and VS Code

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#importing the fashion_minst dataset
#fashion_minst dataset contains 70,000 grayscale images in 10 catagories
#these clothing images are 28 x 28 pixels
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

"""
The images are 28x28 NumPy arrays, with pixel values ranging between 0
and 255. The labels are an array f integers, ranging from 0 to 9. These
correspond to the class of clothing the image represents:
Label   Class
0       T-Shirt/top
1       Trouser
2       Pullover
3       Dress
4       Coat
5       Sandal
6       Shirt 
7       Sneaker
8       Bag
9       Ankle boot
"""
#class names are not included, storing classes here to use later when plotting
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Explore the data
#This will print the size of the data set, and the size of the images
print(train_images.shape)

#This will print the size of the data set
print(len(train_labels))

#Each label is an integer between 0 and 9
print(train_labels)

#The testing set contains 10,000 images. 28x28 pixels
print(test_images.shape)

print(len(test_labels))

#Preprocessing the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#We scale these values to a range of 0 to 1 before feeding to the neural 
#network model. For this, we divide the values by 255. It's important that the
#training set and the testing set are preprocessed in the same way
train_images = train_images / 255.0
test_images = test_images / 255.0

#display the first 25 images from the training set and display the class name
#below each image. Verify that the data is in the correct format and we're
#ready to build and train the network

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#Setup the layers
#The basic building block of a neural network is the layer. Layers extract
#representations from the data fed into them. And, opefully these
#representations are more meaningful for the problem at hand.

#Most of deep learning consists of chaining together simple layers. Most
#layers, like tf.keras.layers.Dense, have parameters that are learning
#during training

#The first layer in this network, tf.keras.layers.Flatten, transforms the
#format of the images from a 2d-array(of 28 by 28 pixells), to a 1d-array of 28
# * 28 = 784 pixels. Think of this layer as unstacking rows of pixels in the
#image and lining them up. This layer has no parameters to learn; it only reformats the data

#After the pixels are flattened, the network consists of a sequence of two
#tf.keras.layers.Dense layers. These are densely-connected, or fully-
#connected, neural layers. The first Dense layer has 128 nodes(or neurons).
#The second (and last) layer is a 10-node softmax layer - this  returns an array
#of 10 probability scores that sum to 1. Each node contains a score that
#indicates the probability that the current image belongs to one of the 10 classes

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Compile the model

#Before the model is ready for training, it needs a few more settings. These
#are added during the model's compile step

#Loss function - Measures how accurate the model is during training.
#We want to minimaix this function to "steer" the model in the right direction

#Optimizer - This is how the model updates based on the data it sees and its loss function

#Metrics - Used to monitor the training and testing steps. The following 
#example uses accuracy, the fraction of the images that are correctly classified

model.compile(optimizer='adam',
                loss = 'sparse_categorical_crossentropy',
                metrics=['accuracy'])

#Train the model
#Steps to train the neural network model
#1 Feed the training data to the model- in this example, the 
#   train_images and train_labels arrays
#
#2 The model learns to associate images and labels
#
#3 We ask the model to make predictions about a test set - in this
#   example, the test_images array. We verify that the predictgions match
#   the labels fromt he test_labels array

#model.fit is used to start training
model.fit(train_images, train_labels, epochs=5)

#Evaluate accuracy
#Now we compare the model to the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#Make predictions
#With the model trained, we can use it to make predictions about some images
#A prediction is an arry of 10 numbers. These describe the "confidence" of
#the model that the image corresponds to each of the 10 different articles
#of clothing. We can see which label has the highest confidence value
predictions = model.predict(test_images)
print(predictions[0])

#highest confidence value label
print(np.argmax(predictions[0]))

print(test_labels[0])

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#plot the first x test images, their predicted label, and the true label
#Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()


#finally use the trained model to make a prediction about a single image

#Grab an image from the test dataset
img = test_images[0]
print(img.shape)

#tf.keras models are optimized to make predictions on a batch, or collection,
#of examples at once. So even though we're using a single image,
#we need to add it to a list
img = (np.expand_dims(img,0))
print(img.shape)

predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

model.predict
plt.show()

print(np.argmax(predictions_single[0]))

#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.