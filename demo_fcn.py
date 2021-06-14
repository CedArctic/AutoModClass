import matplotlib.pyplot as plt
import numpy as np
import pickle
models = ['fc-1-NO-BN', 'fc-1', 'fc-2']
dataset_size = 480000 * 0.8
fig, ax = plt.subplots(2, 3, figsize=(8, 4))
fig.tight_layout(h_pad=2)
for index, bs in enumerate((64, 128, 256, 512, 1024)):
    axi = plt.subplot(2, 4, index + 1)
    plt.ylim(0, 1)
    for model in models:
        curr_model = "{}-{}".format(model, str(bs))
        history = pickle.load(open('history/{}/trainHistoryDict'.format(curr_model), "rb"))
        # Epochs
        epochs_range = np.arange(len(history['accuracy']))
        iter_per_epoch = dataset_size / bs
        iterations = epochs_range * iter_per_epoch
        plt.plot(iterations, history['val_loss'], label='{}'.format(model))
        print("{},{}:{}".format(model,bs,max(history['val_accuracy'])))
        plt.legend(title="BN usage vs val_loss")
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Batch size: {}'.format(bs))


plt.show()
