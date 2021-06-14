import tensorflow as tf
from tensorflow import keras


# === Create Pretrained Model ===
def fc_cum():
    """
    Builds and returns a VGG-16 transfer learning model which can then be compiled and trained. Note that VGG only
    supports 3 channel images.
    :param img_height: Input image height. Minimum is 32 pixels.
    :param img_width: Input image width. Minimum is 32 pixels.
    :param std_input: Option to standardize the input from [0,255] to the [0,1] continuous space. Setting to True
    enables standardization.
    :return: Returns a TensorFlow model which still needs to be compiled and fitted afterwards.
    """
    # Build full model and print its summary
    full_model = keras.Sequential()

    full_model.add(keras.layers.Input(shape=(18,)))
    full_model.add(keras.layers.Dense(32))
    full_model.add(keras.layers.BatchNormalization(momentum=0.9))
    full_model.add(keras.layers.ReLU())
    full_model.add(keras.layers.Dense(16))
    full_model.add(keras.layers.BatchNormalization(momentum=0.9))
    full_model.add(keras.layers.ReLU())
    full_model.add(keras.layers.Dense(8, activation='softmax'))


    # full_model.summary()

    return full_model
