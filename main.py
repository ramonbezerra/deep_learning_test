import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

img_height = 64
img_widht = 64

classes = 4
names_of_classes = ["upright", "rotated_left", "rotated_right", "upside_down"]

batch_size = 32
epochs = 20
train_data_dir = os.path.join(os.getcwd(), 'data') + "\\train"

print(train_data_dir)

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'model.h5'

# The data, split between train and test sets:
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_gen = image_generator.flow_from_directory(directory=str(train_data_dir),
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     target_size=(img_height, img_widht),
                                                     classes = names_of_classes)
