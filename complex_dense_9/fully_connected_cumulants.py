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

    dense1_layer = keras.layers.Dense(100, activation='relu')
    dropout1_layer = keras.layers.Dropout(0.25)
    dense2_layer = keras.layers.Dense(50, activation='relu')
    softmax_layer = keras.layers.Dense(8, activation='softmax')
    model_layers = [dense1_layer, dense2_layer, softmax_layer]

    # Build full model and print its summary
    full_model = keras.Sequential(model_layers)
    # full_model.summary()

    return full_model
