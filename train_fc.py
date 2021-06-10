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
from complex_dense_9.fully_connected_cumulants import fc_cum

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# === Load Data ===
MODEL_NAME = 'fc-1'
print('Training Model: {}'.format(MODEL_NAME))
# Dataset Parameters
batch_size = 64

# Modulation Schemes and SNRs sorted alphanumerically (the way TensorFlow reads training samples)
mod_schemes = ['16APSK', '16PAM', '16QAM', '4PAM', '64APSK', '64QAM', '8PSK', 'QPSK']
mod_schemes.sort()
snrs = [0, 5, 10, 15]
snrs.sort()

# Load data
#
data = 'dataset4_cum'
X = np.zeros((480000, 16))
y = np.zeros((480000, 1))
for index, modulation in enumerate(mod_schemes):
    for i in range(15000):
        cums = np.fromfile("{}/{}/{}.cum", np.complex128)
        real = cums.real
        imag = cums.imag
        # Remove zero imaginary components of C21 and C60 - they are real numbers
        imag = np.delete(imag, (1, 4))
        y[index*15000+i] = index
        X[index*15000+i, :] = np.concatenate(cums.real, cums.imag)
# === Training ===

# Parameters
epochs = 100

# Create model
model = fc_cum()

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

# Plot training and validation accuracy and losses
history = pickle.load(open('history/{}/trainHistoryDict'.format(MODEL_NAME), "rb"))
plotAccLoss(MODEL_NAME, history['accuracy'], history['val_accuracy'], history['loss'], history['val_loss'])
