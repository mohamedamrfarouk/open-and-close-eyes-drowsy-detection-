import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir
import PIL
import PIL.Image
from keras.models import load_model
import h5py
import pathlib
import cv2

import os
import zipfile

base_dir = '/data'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'Open')
train_dogs_dir = os.path.join(train_dir, 'Closed')

# Directory with our validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'Open')
validation_dogs_dir = os.path.join(validation_dir, 'Closed')

train_cat_fnames = os.listdir('F:\\senior year CS department\\graduation project\\first demo'+train_cats_dir)
train_dog_fnames = os.listdir('F:\\senior year CS department\\graduation project\\first demo'+train_dogs_dir)


import matplotlib.image as mpimg
import matplotlib.pyplot as plt

DESIRED_ACCURACY = 0.85


class myCallback(tf.keras.callbacks.Callback):
    # Your Code
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get("acc") > DESIRED_ACCURACY):
            self.stop_traning = True


callbacks = myCallback()

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.15),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.15)  ,

    tf.keras.layers.Dense(16, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. ,
                                    horizontal_flip=True,
                                    zoom_range=0.3,
                                    rotation_range=0.3,
                                    shear_range=0.2
                                    )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )




# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory('F:\\senior year CS department\\graduation project\\first demo'+train_dir,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(150, 150))


# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory('\\senior year CS department\\graduation project\\first demo'+validation_dir,
                                                         batch_size=5,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

from tensorflow.keras.callbacks import ModelCheckpoint

chechpoint_path = 'F:\\senior year CS department\\graduation project\\first demo\\checkpoints\\'
chechpoint = ModelCheckpoint(filepath=chechpoint_path,
                             frequency = 'epoch',
                             save_weights_only=True,
                             verbose=1
                             )

history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=90,
    epochs=20,
    verbose=1,
    callbacks=[chechpoint]
)

