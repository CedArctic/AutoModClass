import numpy as np
import os
import pickle
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import seaborn as sn

from utils.plotting import plotAccLoss
from utils.data import load_data
from transferLearning.vgg16 import vgg16

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# === Load Data ===
MODEL_NAME = 'VGG-frozen-Dropout-batch-100-GAP-DATASET-3-DYNAMIC'
print('Training Model: {}'.format(MODEL_NAME))
# Dataset Parameters
img_height = 224
img_width = 224
img_channels = 3
batch_size = 100

# Modulation Schemes and SNRs sorted alphanumerically (the way TensorFlow reads training samples)
mod_schemes = ['16APSK', '16PAM', '16QAM', '4PAM', '64APSK', '64QAM', '8PSK', 'QPSK']
mod_schemes.sort()
snrs = [0, 5, 10, 15]
snrs.sort()

# Load data
#TODO: Separate method for test datasets loading to improve performance
train_ds, val_ds, test_ds, test_labels = load_data(mod_schemes, snrs, img_height, img_width, batch_size)

# Dataset caching and prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Explore Dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(mod_schemes[labels[i]])
        plt.axis("off")


# === Training ===

# Parameters
epochs = 20

# Create model
model = vgg16(img_height, img_width, std_input=False)

# Print model summary
model.summary()

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Add early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

# Train model
#TODO: Perhaps add ModelCheckpoint, Tensorboard, EarlyStopping and other CallBack functions
history = model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=val_ds, callbacks=[early_stop])

# Save Model
if not os.path.isdir("trained_models"):
    os.makedirs('trained_models')
model.save('trained_models/{}'.format(MODEL_NAME))

# Save training history
if not os.path.isdir("history"):
    os.makedirs('history')
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

np.savetxt('history/train_accuracy.csv', train_accuracy, delimiter=',', fmt='%d', header='Training Accuracy')
np.savetxt('history/val_accuracy.csv', val_accuracy, delimiter=',', fmt='%d', header='Validation Accuracy')
np.savetxt('history/train_loss.csv', train_loss, delimiter=',', fmt='%d', header='Training Loss')
np.savetxt('history/val_loss.csv', val_loss, delimiter=',', fmt='%d', header='Validation Loss')

# Save history as dictionary
if not os.path.isdir("history/{}".format(MODEL_NAME)):
    os.makedirs('history/{}'.format(MODEL_NAME))
with open('history/{}/trainHistoryDict'.format(MODEL_NAME), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# === Inference ===
# Import model
model = keras.models.load_model('trained_models/{}'.format(MODEL_NAME))

# Make predictions
print("Inference started")
predictions = model.predict(test_ds)

# Generate and plot confusion matrix
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
sn.heatmap(cm, annot=True, fmt='g')

# Plot training and validation accuracy and losses
history = pickle.load(open('history/{}/trainHistoryDict'.format(MODEL_NAME), "rb"))
plotAccLoss(MODEL_NAME, history['accuracy'], history['val_accuracy'], history['loss'], history['val_loss'])
