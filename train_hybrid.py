import numpy as np
import os
import pickle
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import seaborn as sn
from sklearn.utils import shuffle

from utils.plotting import plotAccLoss
from utils.data import load_data
from hybrid_model import hybrid_model


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# === Load Data ===
MODEL_NAME = 'HYBRID-1'
cnn_name = 'VGG-frozen-Dropout-batch-100-GAP-DATASET-3-DYNAMIC'
fc_name = 'fc-2-512'
print('Using CNN Model: {}'.format(cnn_name))
print('Using FC Model: {}'.format(cnn_name))
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

data = 'dataset4_cum'
X = np.zeros((480000, 18))
y = np.zeros((480000, 1))
for index, modulation in enumerate(mod_schemes):
    for index_snr, snr in enumerate(snrs):
        print("{},{}db".format(modulation, snr))
        for i in range(15000):
            # nuke complex zeros
            cums = np.fromfile("{}/{}/{}_db/{}.cum".format(data, modulation, snr, i), np.complex128)
            real = cums.real
            imag = cums.imag
            y[index*60000+15000*index_snr+i] = index
            X[index*60000+15000*index_snr+i] = np.concatenate((cums.real, cums.imag))

train_dataset = tf.data.Dataset.from_tensor_slices((X, y))


# Dataset caching and prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# === Training ===

# Parameters
epochs = 30

# Create model
model = hybrid_model(cnn_name, fc_name)

# Print model summary
model.summary()
keras.utils.plot_model(model, "hybrid_model.png", show_shapes=True)

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Add early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train model
#TODO: Perhaps add ModelCheckpoint, Tensorboard, EarlyStopping and other CallBack functions
history = model.fit({'image': train_ds, 'cumulants': cumulants}, batch_size=batch_size, epochs=epochs, validation_data=val_ds, callbacks=[early_stop])

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

# # Make predictions
# print("Inference started")
# predictions = model.predict(test_ds)
#
# # Generate and plot confusion matrix
# cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
# sn.heatmap(cm, annot=True, fmt='g')

# Plot training and validation accuracy and losses
history = pickle.load(open('history/{}/trainHistoryDict'.format(MODEL_NAME), "rb"))
plotAccLoss(MODEL_NAME, history['accuracy'], history['val_accuracy'], history['loss'], history['val_loss'])
