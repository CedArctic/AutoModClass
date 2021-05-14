import numpy as np
import os
import pickle
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from utils.plotting import plotAccLoss
import seaborn as sn

# Import model
model = keras.models.load_model('transferLearning/VGG-16/trained_model/vgg-16-tl-ds1-13-5')

# Dataset directories
train_img_dir = pathlib.Path('dataset/training')
test_img_dir = pathlib.Path('dataset/test')

# Count images
train_img_count = len(list(train_img_dir.glob('*/*/*.png')))
test_img_count = len(list(test_img_dir.glob('*/*/*.png')))

# Dataset Parameters
img_height = 224
img_width = 224
img_channels = 3
batch_size = 32

# Modulation Schemes and SNRs sorted alphanumerically (the way TensorFlow reads training samples)
mod_schemes = ['16APSK', '16PAM', '16QAM', '4PAM', '64APSK', '64QAM', '8PSK', 'QPSK']
mod_schemes.sort()
snrs = [0, 5, 10, 15]
snrs.sort()

# Dictionary with index for each label
mod_idx = {}
for index, mod in enumerate(mod_schemes):
    mod_idx[mod] = index

# Allocate and populate labels array traversing the Root -> (SNRs) -> (Modulation Schemes) tree for the datasets
# Label arrays indices
train_labels_idx = 0
test_labels_idx = 0

# Label arrays
train_labels = np.zeros(train_img_count, dtype=int)
test_labels = np.zeros(test_img_count, dtype=int)
for snr in snrs:
    for mod in mod_schemes:
        # Number of samples for a specific SNR and modulation scheme in each dataset
        train_snr_mod_samples = len(list(train_img_dir.glob('{}_db/{}/*.png'.format(snr, mod))))
        val_snr_mod_samples = len(list(test_img_dir.glob('{}_db/{}/*.png'.format(snr, mod))))
        # Write into label arrays the appropriate number of modulation scheme indices according to the number of samples
        train_labels[train_labels_idx:train_labels_idx + train_snr_mod_samples] = mod_idx[mod] * np.ones(train_snr_mod_samples, dtype=int)
        test_labels[test_labels_idx:test_labels_idx + val_snr_mod_samples] = mod_idx[mod] * np.ones(val_snr_mod_samples, dtype=int)
        # Increment the label array indices
        train_labels_idx += train_snr_mod_samples
        test_labels_idx += val_snr_mod_samples

# Training dataset structure
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_img_dir,
    labels=train_labels.tolist(),
    label_mode='int',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False)

# Validation dataset structure
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_img_dir,
    labels=train_labels.tolist(),
    label_mode='int',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False)

# Test dataset structure
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_img_dir,
    labels=test_labels.tolist(),
    label_mode='int',
    seed=123,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Make predictions
print("Inference started")
predictions = model.predict(test_ds)

# Generate and plot confusion matrix
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
sn.heatmap(cm, annot=True, fmt='g')

# Plot training and validation accuracy and losses
history = pickle.load(open('transferLearning/VGG-16/history/trainHistoryDict', "rb"))
plotAccLoss("VGG-16", history['accuracy'], history['val_accuracy'], history['loss'], history['val_loss'])
