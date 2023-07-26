import pickle
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from random import random
from tqdm import tqdm
from numpy import loadtxt
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from keras.utils import np_utils
import keras
from sklearn.model_selection import train_test_split
from plot_keras_history import show_history, plot_history
from keras.layers import Conv2D, Flatten, Dropout
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
from skimage.io import imread_collection
from tensorflow.keras.layers import Layer


### (2)

# Load the MNIST Dataset and Normalize
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

print (train_images.shape)
print (test_images.shape)

# Add Noise
noise_factor = 0.2
train_images_noisy = train_images + noise_factor * tf.random.normal(shape=train_images.shape)
test_images_noisy = test_images + noise_factor * tf.random.normal(shape=test_images.shape)
# Make sure values still in (0,1)
train_images_noisy = tf.clip_by_value(train_images_noisy, clip_value_min=0., clip_value_max=1.)
test_images_noisy = tf.clip_by_value(test_images_noisy, clip_value_min=0., clip_value_max=1.)

### (3)

# CNN Classifier Model 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10))
print(model.summary())

# Compile the model
model.compile(optimizer='adam',loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model using clean images
model.fit(train_images[..., tf.newaxis], train_labels, epochs=10)

### (4)

# Evaluate the model on the clean test images
test_loss, test_acc = model.evaluate(test_images[..., tf.newaxis], test_labels, verbose=2)
print('Clean test accuracy:', test_acc)

### (5)

# Evaluate the model on the noisy test images
test_loss, test_acc = model.evaluate(test_images_noisy[..., tf.newaxis], test_labels, verbose=2)
print('Noisy test accuracy:', test_acc)

### (6)

# Define the encoder layers
encoder = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])


# Define the decoder layers
decoder = tf.keras.Sequential([
      tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

# Define the input layer
inputs = tf.keras.layers.Input(shape=(28, 28, 1))

# Chain the encoder and decoder layers together
x = encoder(inputs)
x = decoder(x)

# Define the output layer
outputs = x

# Define the autoencoder model
autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

# epochs 5-10 should be enough
autoencoder.fit(train_images_noisy, train_images, epochs=5, shuffle=True, validation_data=(test_images_noisy, test_images))
autoencoder.summary()

encoder.summary()
decoder.summary()

encoded_imgs = encoder(test_images).numpy()
decoded_imgs = decoder(encoded_imgs).numpy()

# Denoise some examples and plot them
# Select some noisy test images to denoise
num_examples = 10
noisy_examples = test_images[:num_examples] + noise_factor * tf.random.normal(shape=(num_examples, 28, 28))
noisy_examples = tf.clip_by_value(noisy_examples, clip_value_min=0., clip_value_max=1.)

# Denoise the noisy test images using the trained autoencoder
denoised_examples = autoencoder.predict(noisy_examples[..., tf.newaxis])

# Plot the noisy and denoised test images side-by-side
fig, axs = plt.subplots(2, num_examples, figsize=(num_examples*2, 4))
for i in range(num_examples):
    axs[0][i].imshow(noisy_examples[i], cmap='gray')
    axs[0][i].set_axis_off()
    axs[1][i].imshow(denoised_examples[i,...,0], cmap='gray')
    axs[1][i].set_axis_off()
plt.tight_layout()
plt.show()

### (7)

# Evaluate the model on the denoised test images
test_loss = autoencoder.evaluate(denoised_examples, noisy_examples)
print('Test loss:', test_loss)

### (8)

# Train the model using noisy images
model.fit(train_images_noisy[..., tf.newaxis], train_labels, epochs=10)

# Evaluate the model on the noisy test images
test_loss, test_acc = model.evaluate(test_images_noisy[..., tf.newaxis], test_labels, verbose=2)
print('Noisy test accuracy:', test_acc)


# Extra Credit
# End to end denoising-classifier network for noisy mnist

# Define the input shape
input_shape = (28, 28, 1)

# Define the encoder
inputs = tf.keras.Input(shape=input_shape)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
encoder_output = tf.keras.layers.Flatten()(x)

# Define the classifier branch
classifier_x = tf.keras.layers.Dense(64, activation='relu')(encoder_output)
classifier_output = tf.keras.layers.Dense(10, activation='softmax')(classifier_x)

# Define the decoder branch
decoder_x = tf.keras.layers.Dense(7 * 7 * 128, activation='relu')(encoder_output)
decoder_x = tf.keras.layers.Reshape((7, 7, 128))(decoder_x)
decoder_x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(decoder_x)
decoder_x = tf.keras.layers.UpSampling2D((2, 2))(decoder_x)
decoder_x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(decoder_x)
decoder_x = tf.keras.layers.UpSampling2D((2, 2))(decoder_x)
decoder_x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(decoder_x)
decoder_output = tf.keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(decoder_x)

# Define the outputs dictionary
outputs = {"classifier_output": classifier_output, "decoder_output": decoder_output} 

# Define the full model
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="denoising_classifier")

# Compile the model
model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy', 'mse'], loss_weights=[0.8, 0.2], metrics=['accuracy'])

# Train the model
model.fit(x=train_images_noisy, y={"classifier_output": train_labels, "decoder_output": train_images_noisy}, validation_data=(test_images_noisy, {"classifier_output": test_labels, "decoder_output": test_images_noisy}), epochs=20, batch_size=200)

# Evaluate the model on the test data
loss_and_accuracy = model.evaluate(test_images_noisy, {"classifier_output": test_labels, "decoder_output": test_images_noisy})
loss = loss_and_accuracy[0]
accuracy = loss_and_accuracy[1]
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

