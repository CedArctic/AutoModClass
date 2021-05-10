import numpy as np
import os
import pathlib
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# === Import Dataset Images ===

# Data directories
train_img_dir = pathlib.Path('dataset/training')
val_img_dir = pathlib.Path('dataset/validation')

# Count images
val_img_count = len(list(val_img_dir.glob('*/*/*.png')))
train_img_count = len(list(train_img_dir.glob('*/*/*.png')))

# Model Parameters
batch_size = 32
img_height = 224
img_width = 224
img_channels = 3

# Modulation Schemes and SNRs sorted alphanumerically (the way TensorFlow reads training samples)
mod_schemes = ['16APSK', '16PAM', '16QAM', '4PAM', '64APSK', '64QAM', '8PSK', 'QPSK']
mod_schemes.sort()
snrs = [0, 5]
snrs.sort()

# Dictionary with index for each label
mod_idx = {}
for index, mod in enumerate(mod_schemes):
    mod_idx[mod] = index

# Allocate and populate labels array traversing the Root -> (SNRs) -> (Modulation Schemes) tree
labels_idx = 0
#TODO: change to train_img_count
train_labels = np.zeros(val_img_count, dtype=int)
for snr in snrs:
    for mod in mod_schemes:
        snr_mod_samples = len(list(val_img_dir.glob('{}_db/{}/*.png'.format(snr, mod))))
        train_labels[labels_idx:labels_idx+snr_mod_samples] = mod_idx[mod] * np.ones(snr_mod_samples, dtype=int)
        labels_idx += snr_mod_samples

# Form dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_img_dir,
    labels=train_labels.tolist(),
    label_mode='int',
    # validation_split=0.2,
    # subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Explore Dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(mod_schemes[labels[i]])
        plt.axis("off")


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

# Flatten, dense and softmax layers
flat_layer = tf.keras.layers.Flatten()
dense1_layer = tf.keras.layers.Dense(100, activation='relu')
dense2_layer = tf.keras.layers.Dense(100, activation='relu')
dense3_layer = tf.keras.layers.Dense(50, activation='relu')
softmax_layer = tf.keras.layers.Dense(8, activation='softmax')

# Build full model and print its summary
full_model = tf.keras.Sequential([vgg16_model, flat_layer, dense1_layer, dense2_layer, dense3_layer, softmax_layer])
full_model.summary()

# Compile and train model
full_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy,
              metrics=[tf.keras.metrics.Accuracy])
model.fit(train_ds, epochs=20, callbacks=..., validation_data=val_ds)
