import os
import pathlib
import numpy as np
import tensorflow as tf

# Loads data off HDD
def load_data(mod_schemes, snrs, img_height, img_width, batch_size):
    # === Parameters ===
    # Dataset directories
    train_img_dir = pathlib.Path('dataset/training')
    test_img_dir = pathlib.Path('dataset/test')

    # Count images
    train_img_count = len(list(train_img_dir.glob('*/*/*.png')))
    test_img_count = len(list(test_img_dir.glob('*/*/*.png')))

    # === Code used for the x_db/mod_scheme/ dataset structure ===
    # # Dictionary with index for each label
    # mod_idx = {}
    # for index, mod in enumerate(mod_schemes):
    #     mod_idx[mod] = index
    #
    # # Allocate and populate labels array traversing the Root -> (SNRs) -> (Modulation Schemes) tree for the datasets
    # # Label arrays indices
    # train_labels_idx = 0
    # test_labels_idx = 0
    #
    # # Label arrays
    # train_labels = np.zeros(train_img_count, dtype=int)
    # test_labels = np.zeros(test_img_count, dtype=int)
    # for snr in snrs:
    #     for mod in mod_schemes:
    #         # Number of samples for a specific SNR and modulation scheme in each dataset
    #         train_snr_mod_samples = len(list(train_img_dir.glob('{}_db/{}/*.png'.format(snr, mod))))
    #         val_snr_mod_samples = len(list(test_img_dir.glob('{}_db/{}/*.png'.format(snr, mod))))
    #         # Write into label arrays the appropriate number of modulation scheme indices according to the number of samples
    #         train_labels[train_labels_idx:train_labels_idx + train_snr_mod_samples] = mod_idx[mod] * np.ones(
    #             train_snr_mod_samples, dtype=int)
    #         test_labels[test_labels_idx:test_labels_idx + val_snr_mod_samples] = mod_idx[mod] * np.ones(
    #             val_snr_mod_samples, dtype=int)
    #         # Increment the label array indices
    #         train_labels_idx += train_snr_mod_samples
    #         test_labels_idx += val_snr_mod_samples

    # Training dataset structure
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_img_dir,
        # labels=train_labels.tolist(),
        label_mode='int',
        validation_split=0.2,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        seed=123)

    # Validation dataset structure
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_img_dir,
        # labels=train_labels.tolist(),
        label_mode='int',
        validation_split=0.2,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        seed=123)

    # Test dataset structure
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_img_dir,
        # labels=test_labels.tolist(),
        label_mode='int',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False)

    # Get testset labels
    test_labels = np.concatenate([y for x, y in test_ds], axis=0)

    return train_ds, val_ds, test_ds, test_labels
