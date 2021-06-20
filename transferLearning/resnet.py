import tensorflow as tf
from tensorflow import keras


# === Create Pretrained Model ===
def resnet(img_height, img_width, std_input):
    """
    Builds and returns a VGG-16 transfer learning model which can then be compiled and trained. Note that VGG only
    supports 3 channel images.
    :param img_height: Input image height. Minimum is 32 pixels.
    :param img_width: Input image width. Minimum is 32 pixels.
    :param std_input: Option to standardize the input from [0,255] to the [0,1] continuous space. Setting to True
    enables standardization.
    :return: Returns a TensorFlow model which still needs to be compiled and fitted afterwards.
    """

    # Inputs shape in channel last format
    input_shape = (img_height, img_width, 3)

    # Full model input tensor for 3-channel images
    in_layer = keras.Input(shape=input_shape)

    # Instantiate a VGG-16 network trained on Imagenet with no "top" layers and a predefined Inputs tensor
    resnet = keras.applications.ResNet101(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=None,
        classifier_activation="softmax",
    )

    # Print VGG-16 model summary
    # vgg16_model.summary()

    # Freeze VGG-16
    resnet.trainable = False
    res_out = resnet(in_layer)
    # Flatten, dense and softmax layers
    gap_layer = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(res_out)
    # flat_layer = tf.keras.layers.Flatten()
    BN_layer_gap = tf.keras.layers.BatchNormalization(momentum=0.9)(gap_layer)
    gmp_layer = tf.keras.layers.GlobalMaxPooling2D(data_format='channels_last')(res_out)
    BN_layer_gmp = tf.keras.layers.BatchNormalization(momentum=0.9)(gmp_layer)
    conc = tf.keras.layers.Concatenate()([BN_layer_gap, BN_layer_gmp])
    dense1_layer = tf.keras.layers.Dense(100, activation='relu')(conc)
    # dropout1_layer = tf.keras.layers.Dropout(0.25)
    BN_layer2 = tf.keras.layers.BatchNormalization(momentum=0.9)(dense1_layer)
    dense2_layer = tf.keras.layers.Dense(50, activation='relu')(BN_layer2)
    softmax_layer = tf.keras.layers.Dense(8, activation='softmax')(dense2_layer)

    # # Model Layers
    # if std_input:
    #     # With Standardization Layer ([0,255] -> [0,1])
    #     std_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    #     model_layers = [in_layer, std_layer, vgg16_model,
    #                     gap_layer, dense1_layer, dropout1_layer, dense2_layer,
    #                     softmax_layer]
    # else:
    #     # Without Standardization Layer
    #     model_layers = [in_layer, vgg16_model,
    #                     gap_layer, dense1_layer, dropout1_layer, dense2_layer,
    #                     softmax_layer]

    # Build full model and print its summary
    full_model = keras.Model(inputs=in_layer, outputs=softmax_layer)
    # full_model.summary()

    return full_model
