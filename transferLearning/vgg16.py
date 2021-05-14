import numpy as np
import os
import pathlib
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# === Create Pretrained Model ===
def vgg16(img_height, img_width, img_channels, std_input):

    # Full model input tensor
    in_layer = keras.Input(shape=(img_height, img_width, img_channels))

    # Instantiate a VGG-16 network trained on Imagenet with no "top" layers and a predefined Inputs tensor
    vgg16_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(img_height, img_width, img_channels),
        pooling=None,
        classes=None,
        classifier_activation="softmax",
    )

    # Print VGG-16 model summary
    # vgg16_model.summary()

    # Freeze VGG-16
    vgg16_model.trainable = False

    # Flatten, dense and softmax layers
    flat_layer = tf.keras.layers.Flatten()
    dense1_layer = tf.keras.layers.Dense(100, activation='relu')
    dropout1_layer = tf.keras.layers.Dropout(0.25)
    dense2_layer = tf.keras.layers.Dense(100, activation='relu')
    dropout2_layer = tf.keras.layers.Dropout(0.25)
    dense3_layer = tf.keras.layers.Dense(50, activation='relu')
    dropout3_layer = tf.keras.layers.Dropout(0.25)
    softmax_layer = tf.keras.layers.Dense(8, activation='softmax')

    # Model Layers
    if std_input:
        # With Standardization Layer ([0,255] -> [0,1])
        std_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        model_layers = [in_layer, std_layer, vgg16_model, flat_layer, dense1_layer, dense2_layer, dense3_layer, softmax_layer]
    else:
        # Without Standardization Layer
        model_layers = [in_layer, vgg16_model, flat_layer, dense1_layer, dense2_layer, dense3_layer, softmax_layer]


    # Build full model and print its summary
    full_model = tf.keras.Sequential(model_layers)
    # full_model.summary()

    return full_model