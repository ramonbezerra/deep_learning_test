import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

# data_train = pd.read_csv('data/train.truth.csv')

# print(data_train.count())

# for i in data_train.itertuples():
#     os.link("data/train/{}".format(i.fn), "train/{0}/{1}".format(i.label, i.fn))
#     os.remove("data/train/{}".format(i.fn))

img_height = 64
img_widht = 64

num_classes = 4
names_of_classes = ["upright", "rotated_left", "rotated_right", "upside_down"]

batch_size = 32
epochs = 20
train_data_dir = os.path.join(os.getcwd(), "data\\train")

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'model.h5'

# The data, split between train and test sets:
train_img = ImageDataGenerator(rescale=1./255)
train_data_gen = train_img.flow_from_directory(directory=train_data_dir,
                                                     batch_size=batch_size,
                                                     classes=names_of_classes,
                                                     target_size=(img_height, img_widht))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Load model
model = load_model(model_name)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print(model.summary())

# model.fit(train_data_gen, epochs=5, verbose=1, shuffle=True)
# model.save(model_name)

output = open('test.preds.csv', 'a')
output.write("fn, label\n")

def predict_image(fn):
    test_img_1 = load_img(fn, target_size=(64,64))
    test_img_1 = img_to_array(test_img_1)
    test_img_1 = np.expand_dims(test_img_1, axis=0)
    result = model.predict(test_img_1)
    if result[0][0] == 1:
        output.write("{},upright\n".format(fn).replace("data/test\\", ""))
    elif result[0][1] == 1:
        output.write("{},rotated_left\n".format(fn).replace("data/test\\", ""))
    elif result[0][2] == 1:
        output.write("{},rotated_right\n".format(fn).replace("data/test\\", ""))
    elif result[0][3] == 1:
        output.write("{},upside_down\n".format(fn).replace("data/test\\", ""))

count = 1
# load all images
for img in os.listdir("data/test"):
    img = os.path.join("data/test", img)
    predict_image(img)
    print(count)
    count += 1

output.close()

# stack up images list to pass for prediction
# images = np.vstack(images)
# classes = model.predict_classes(images, batch_size=10)
# print(classes)

# upside_down_image = 'data/test/90-890_1981-06-07_2009.jpg'
# predict_image(upside_down_image)

# rotated_left_image = 'data/test/90-15890_1922-03-12_1943.jpg'
# predict_image(rotated_left_image)

# rotated_right_image = 'data/test/90-17190_1940-01-17_2014.jpg'
# predict_image(rotated_right_image)

# upright_image = 'data/test/90-55490_1916-09-13_1954.jpg'
# predict_image(upright_image)