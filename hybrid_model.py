from tensorflow import keras


def hybrid_model(cnn_model_name, fc_model_name):

    cnn = keras.models.load_model('trained_models/{}'.format(cnn_model_name))
    cnn.trainable = False
    cumulant_model = keras.models.load_model('trained_models/{}'.format(fc_model_name))
    cumulant_model.trainable = False
    cnn_input = keras.Input(shape=(224, 224, 3), name='image')
    cumulant_input = keras.Input(shape=(18, ), name='cumulants')
    cnn_output = cnn(cnn_input)
    cumulant_output = cumulant_model(cumulant_input)
    conc = keras.layers.Concatenate()([cumulant_output, cnn_output])
    dense1 = keras.layers.Dense(32)(conc)
    BN1 = keras.layers.BatchNormalization(momentum=0.9)(dense1)
    relu1 = keras.layers.ReLU()(BN1)
    dense2 = keras.layers.Dense(16)(relu1)
    BN2 = keras.layers.BatchNormalization(momentum=0.9)(dense2)
    relu2 = keras.layers.ReLU()(BN2)
    dense_soft = keras.layers.Dense(8, activation='softmax')(relu2)
    full_model = keras.Model(inputs=[cumulant_input, cnn_input], outputs=dense_soft)
    full_model.summary()
    keras.utils.plot_model(full_model, "hybrid_model.png", show_shapes=True)
    return full_model
