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

# Import validation dataset
#TODO: Switch from a validation dataset to a test dataset. This is for testing purposes only.
val_img_dir = pathlib.Path('transferLearning/VGG-16/dataset/validation')
val_img_count = len(list(val_img_dir.glob('*/*/*.png')))
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

# Allocate and populate labels array traversing the Root -> (SNRs) -> (Modulation Schemes) tree for the training and
# validation datasets
# Label arrays indices
val_labels_idx = 0

# Label arrays
val_labels = np.zeros(val_img_count, dtype=int)
for snr in snrs:
    for mod in mod_schemes:
        # Number of samples for a specific SNR and modulation scheme in each dataset
        val_snr_mod_samples = len(list(val_img_dir.glob('{}_db/{}/*.png'.format(snr, mod))))
        # Write into label arrays the appropriate number of modulation scheme indices according to the number of samples
        val_labels[val_labels_idx:val_labels_idx + val_snr_mod_samples] = mod_idx[mod] * np.ones(val_snr_mod_samples, dtype=int)
        # Increment the label array indices
        val_labels_idx += val_snr_mod_samples

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_img_dir,
    labels=val_labels.tolist(),
    label_mode='int',
    # validation_split=0.2,
    # subset="training",
    # seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False)

# Make predictions
print("Inference started")
predictions = model.predict(val_ds)

# Plot confusion matrix
cm = confusion_matrix(val_labels, predictions.argmax(axis=1))

sn.heatmap(cm, annot=True, fmt='g')

# Plot training and validation accuracy and losses
history = pickle.load(open('transferLearning/VGG-16/history/trainHistoryDict', "rb"))
plotAccLoss("VGG-16", history['accuracy'], history['val_accuracy'], history['loss'], history['val_loss'])
