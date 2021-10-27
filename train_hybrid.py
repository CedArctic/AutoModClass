import numpy as np
import os
import pickle
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import seaborn as sn
from sklearn.utils import shuffle

from utils.plotting import plotAccLoss
from utils.data import load_hybrid_data, get_dataset, get_selected_dataset, get_image_dataset
from hybrid_model import hybrid_model
from transferLearning.resnet import resnet
from utils.lr_schedules import exp_decay_schedule, szegedy_schedule, cosine_annealing_schedule, step_decay_schedule, triangular_cyclic_schedule

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# === Load Data ===
MODEL_NAME = 'HYBRID_MODEL_TEST1'
cnn_name = 'RESNET-152v2-SGD-LRPLAT-LR1e-3-MOMENTUM-EXP_DECAY'
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
# #TODO: Separate method for test datasets loading to improve performance
# train_ds, val_ds, test_ds, test_labels = load_data(mod_schemes, snrs, img_height, img_width, batch_size)
#
# data = 'dataset4_cum'
# X = np.zeros((480000, 18))
# y = np.zeros((480000, 1))
# for index, modulation in enumerate(mod_schemes):
#     for index_snr, snr in enumerate(snrs):
#         print("{},{}db".format(modulation, snr))
#         for i in range(15000):
#             # nuke complex zeros
#             cums = np.fromfile("{}/{}/{}_db/{}.cum".format(data, modulation, snr, i), np.complex128)
#             real = cums.real
#             imag = cums.imag
#             y[index*60000+15000*index_snr+i] = index
#             X[index*60000+15000*index_snr+i] = np.concatenate((cums.real, cums.imag))
#
# train_dataset = tf.data.Dataset.from_tensor_slices((X, y))


# Dataset caching and prefetching
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = get_image_dataset(['tfrecords/shard_{}.tfrecords'.format(i) for i in range(16)], ordered=False)
# val_ds = get_image_dataset(['tfrecords/shard_{}.tfrecords'.format(i) for i in range(16,20)], ordered=False)
#
# train_ds = train_ds.shuffle(10000, reshuffle_each_iteration=True)
# val_ds = val_ds.shuffle(10000, reshuffle_each_iteration=True)
#
#
# train_ds = train_ds.batch(batch_size)
# val_ds = val_ds.batch(batch_size)
# # train_ds = train_ds.repeat()
# train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# Load data
#TODO: Separate method for test datasets loading to improve performance
train_ds, val_ds, test_ds, test_labels = load_hybrid_data(mod_schemes, snrs, img_height, img_width, batch_size)

# Dataset caching and prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# === Training ===

# Parameters
epochs = 30

# Create model
model = hybrid_model(cnn_name, fc_name)
# model = resnet(img_height, img_width, std_input=False)
# Print model summary
model.summary()
keras.utils.plot_model(model, "hybrid_model.png", show_shapes=True)

# Compile model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Add early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
if not os.path.isdir("checkpoints"):
    os.makedirs('checkpoints')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"checkpoints/{MODEL_NAME}_BEST",
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(szegedy_schedule)
# Train model
#TODO: Perhaps add ModelCheckpoint, Tensorboard, EarlyStopping and other CallBack functions
history = model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=val_ds,
                    callbacks=[early_stop, model_checkpoint_callback, lr_schedule])

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
