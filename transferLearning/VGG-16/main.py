import numpy as np
import os
import pathlib
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras

# === Import Dataset Images ===

# Data directories
train_img_dir = pathlib.Path('dataset/training')
val_img_dir = pathlib.Path('dataset/validation')

# Count images
val_img_count = len(list(val_img_dir.glob('*/*/*.png')))
train_img_count = len(list(train_img_dir.glob('*/*/*.png')))

# Parameters
batch_size = 32
img_height = 224
img_width = 224
img_channels = 3


# === Create Pretrained Model ===

# Full model input tensor
in_tensor = keras.Input(shape=(img_height, img_width, img_channels))

# Instantiate a VGG-16 network trained on Imagenet with no "top" layers and a predefined Inputs tensor
vgg16_model = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_tensor=in_tensor,
    input_shape=(img_height, img_width, img_channels),
    pooling=None,
    classes=None,
    classifier_activation="softmax",
)

# Print VGG-16 model summary
vgg16_model.summary()

# Freeze VGG-16
vgg16_model.trainable = False

# Add dense layers
d1 = tf.keras.layers.Flatten
