import matplotlib.pyplot as plt
import numpy as np

# Plots Accuracy and Loss timeseries across epochs in a side by side plot
def plotAccLoss(model_name, train_accuracy, val_accuracy, train_loss, val_loss):
    # Epochs
    epochs_range = np.arange(len(train_accuracy))

    # Accuracy
    plt.figure(figsize=(16, 8))
    plt.ylim(0, 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracy, label='{} Training Set Accuracy'.format(model_name))
    plt.plot(epochs_range, val_accuracy, label='{} Validation Set Accuracy'.format(model_name))
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='{} Training Loss'.format(model_name))
    plt.plot(epochs_range, val_loss, label='{} Validation Loss'.format(model_name))
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
