import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

model = 'fc-1-NO-BN'
batch_sizes = [64, 128, 256, 512, 1024]
dataset_size = 480000 * 0.8
# Accuracy
plt.figure(figsize=(16, 8))

for bs in batch_sizes:
    curr_model = "{}-{}".format(model, str(bs))
    history = pickle.load(open('history/{}/trainHistoryDict'.format(curr_model), "rb"))
    # Epochs
    epochs_range = np.arange(len(history['accuracy']))
    iter_per_epoch = dataset_size / bs
    iterations = epochs_range * iter_per_epoch

    plt.subplot(1, 2, 1)
    plt.ylim(0, 1)
    plt.plot(iterations, history['val_accuracy'], label='{}'.format(bs))
    plt.legend(title="batch size")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')

for bs in batch_sizes:
    curr_model = "{}-{}".format(model, str(bs))
    history = pickle.load(open('history/{}/trainHistoryDict'.format(curr_model), "rb"))
    # Epochs
    epochs_range = np.arange(len(history['accuracy']))
    iter_per_epoch = dataset_size / bs
    iterations = epochs_range * iter_per_epoch
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(iterations, history['val_loss'], label='{}'.format(bs))
    plt.legend(title="batch size")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Validation Loss')

plt.show()
