import numpy as np
import os
import pathlib
import pickle
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
epochs = 100
img_height = 224
img_width = 224
img_channels = 3

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
train_labels_idx = 0
val_labels_idx = 0
# Label arrays
train_labels = np.zeros(train_img_count, dtype=int)
val_labels = np.zeros(val_img_count, dtype=int)
for snr in snrs:
    for mod in mod_schemes:
        # Number of samples for a specific SNR and modulation scheme in each dataset
        train_snr_mod_samples = len(list(train_img_dir.glob('{}_db/{}/*.png'.format(snr, mod))))
        val_snr_mod_samples = len(list(val_img_dir.glob('{}_db/{}/*.png'.format(snr, mod))))
        # Write into label arrays the appropriate number of modulation scheme indices according to the number of samples
        train_labels[train_labels_idx:train_labels_idx + train_snr_mod_samples] = mod_idx[mod] * np.ones(train_snr_mod_samples, dtype=int)
        val_labels[val_labels_idx:val_labels_idx + val_snr_mod_samples] = mod_idx[mod] * np.ones(val_snr_mod_samples, dtype=int)
        # Increment the label array indices
        train_labels_idx += train_snr_mod_samples
        val_labels_idx += val_snr_mod_samples

# Form dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_img_dir,
    labels=train_labels.tolist(),
    label_mode='int',
    # validation_split=0.2,
    # subset="training",
    # seed=123,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_img_dir,
    labels=val_labels.tolist(),
    label_mode='int',
    # validation_split=0.2,
    # subset="training",
    # seed=123,
    shuffle=True,
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
#TODO: Perhaps add ModelCheckpoint, Tensorboard, EarlyStopping and other CallBack functions
history = full_model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=val_ds)

# Save Model
os.makedirs('trained_model')
full_model.save('trained_model/vgg-16-tl-ds1-13-5')

# Save training history
os.makedirs('history')
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

np.savetxt('history/train_accuracy.csv', [train_accuracy], delimiter=',', fmt='%d', header='Training Accuracy')
np.savetxt('history/val_accuracy.csv', [val_accuracy], delimiter=',', fmt='%d', header='Validation Accuracy')
np.savetxt('history/train_loss.csv', [train_loss], delimiter=',', fmt='%d', header='Training Loss')
np.savetxt('history/val_loss.csv', [val_loss], delimiter=',', fmt='%d', header='Validation Loss')

# Save history as dictionary
with open('history/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)