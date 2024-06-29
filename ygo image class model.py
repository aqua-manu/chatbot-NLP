import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator

dataset_path = r'C:\Users\Aquamanu\Documents\AI IMAGES\archive\images'


# image dimensions
img_height, img_width = 187, 128  
num_channels = 3  #  3 for RGB

num_classes = 8

# ImageDataGenerator for preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#training data
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\Aquamanu\Documents\AI IMAGES\archive\images\train',
    target_size=(img_height, img_width),
    batch_size=16,
    class_mode='categorical'
)

#test data
test_generator = test_datagen.flow_from_directory(
    r'C:\Users\Aquamanu\Documents\AI IMAGES\archive\images\test',
    target_size=(img_height, img_width),
    batch_size=16,
    class_mode='categorical'
)

#model
model = keras.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(2, 2), activation='relu', input_shape=(img_height, img_width, num_channels)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(filters=32, kernel_size=(2, 2), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
epochs = 10  
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# Save HDF5
model.save('yugioh_model.h5')