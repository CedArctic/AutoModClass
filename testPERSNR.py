import pathlib
import numpy as np
import tensorflow as tf
test_img_dir = pathlib.Path('dataset/dataset3_dyn_snr_test/15_db')
MODEL_NAME = 'VGG-frozen-Dropout-batch-100-GAP-DATASET-3-DYNAMIC'

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_img_dir,
        # labels=test_labels.tolist(),
        label_mode='int',
        image_size=(224, 224),
        batch_size=100,
        shuffle=False)

# Get testset labels
test_labels = np.concatenate([y for x, y in test_ds], axis=0)

model = tf.keras.models.load_model('trained_models/{}'.format(MODEL_NAME))

# Make predictions
print("Inference started")
predictions = model.predict(test_ds).argmax(axis=1)
a = (predictions == test_labels)
acc = sum(a)/len(a)
