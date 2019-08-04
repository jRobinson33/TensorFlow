# Tutorial from tensorflow.org 
# https://www.tensorflow.org/tutorials/keras/basic_text_classification

# keras.datasets.imdb is brokin in 1.13 and 1.14, by np 1.16.3
# pip install -q tf_nightly

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf_nightly

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

imdb = keras.datasets.imdb

""" The argument num_words=10000 keeps the top 10,000 most frequently
 occurring words in the training data. The rare words are discarded to
  keep the size of the data manageable. """
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# The text reviews have been conveted to integers, each integer reprsents a specific
# word in a dictionary
#print(train_data[0])

#-------------------------------------------------------------------
# Converting the integers back to words
#-------------------------------------------------------------------

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# the first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))

#-------------------------------------------------------------------
#Prepare the data
#-------------------------------------------------------------------

#The following block comment was copy pasted from the tutorial
"""
    The reviews—the arrays of integers—must be converted to tensors before fed
    into the neural network. This conversion can be done a couple of ways:

        Convert the arrays into vectors of 0s and 1s indicating word occurrence, 
        similar to a one-hot encoding. For example, the sequence [3, 5] would 
        become a 10,000-dimensional vector that is all zeros except for indices 
        3 and 5, which are ones. Then, make this the first layer in our network—a 
        Dense layer—that can handle floating point vector data. This approach is 
        memory intensive, though, requiring a num_words * num_reviews size matrix.

        Alternatively, we can pad the arrays so they all have the same length, then 
        create an integer tensor of shape max_length * num_reviews. We can use an 
        embedding layer capable of handling this shape as the first layer in our network.

    In this tutorial, we will use the second approach.

    Since the movie reviews must be the same length, we will use the pad_sequences function to standardize the lengths:
"""
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

print(len(train_data[0]), len(train_data[1]))

print(train_data[0])

#-------------------------------------------------------------------
# Build the model
#-------------------------------------------------------------------

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

#-------------------------------------------------------------------
# Loss function and optimizer
#-------------------------------------------------------------------
# The tutorial explains that there are a few loss functions
# mean_squared_error is an option, however it says that
# binary_crossentropy is better generally for dealing with 
# probabilities, and that it measures the "distance" between
# probability distributions

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

#-------------------------------------------------------------------
# Create a validation set
#-------------------------------------------------------------------
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val=train_labels[:10000]
partial_y_train = train_labels[10000:]



#-------------------------------------------------------------------
# Train the model
#-------------------------------------------------------------------
"""
    Train the model for 40 epochs in mini-batches of 512 samples. This is 
    40 iterations over all samples in the x_train and y_train tensors. 
    While training, monitor the model's loss and accuracy on the 10,000 
    samples from the validation set:
"""
history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=40,
                  batch_size=512,
                  validation_data=(x_val, y_val),
                  verbose=1)


#-------------------------------------------------------------------
# Evaluate the model
#-------------------------------------------------------------------
results = model.evaluate(test_data, test_labels)

print(results)

#-------------------------------------------------------------------
# create a graph of accuracy and loss over time
#-------------------------------------------------------------------
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt 

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf() # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()