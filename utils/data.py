import os
import pathlib
import numpy as np
import tensorflow as tf
from utils.hybrid_dataset_from_directory import hybrid_dataset_from_directory


def get_selected_dataset(ds, X_indices_np):
    # Make a tensor of type tf.int64 to match the one by Dataset.enumerate().
    X_indices_ts = tf.constant(X_indices_np, dtype=tf.int64)

    def is_index_in(index, rest):
        # Returns True if the specified index value is included in X_indices_ts.
        #
        # '==' compares the specified index value with each values in X_indices_ts.
        # The result is a boolean tensor, looks like [ False, True, ..., False ].
        # reduce_any() returns Ture if True is included in the specified tensor.
        return tf.math.reduce_any(index == X_indices_ts)

    def drop_index(index, rest):
        return rest

    # Dataset.enumerate() is similter to Python's enumerate().
    # The method adds indices to each elements. Then, the elements are filtered
    # by using the specified indices. Finally unnecessary indices are dropped.
    selected_ds = ds \
        .enumerate() \
        .filter(is_index_in) \
        .map(drop_index)
    return selected_ds


def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'snr': tf.io.FixedLenFeature([], tf.int64),
        'mod': tf.io.FixedLenFeature([], tf.int64),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'cumulants': tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(element, data)

    snr = content['snr']
    mod = content['mod']
    cumulants = content['cumulants']
    raw_image = content['raw_image']

    # get our 'feature'-- our image -- and reshape it appropriately
    img = tf.io.parse_tensor(raw_image, out_type=tf.string)
    cumulants = tf.io.parse_tensor(cumulants, out_type=tf.float64)
    # TODO: KEEP IMG DIMS IN TFRECORDS ENTRY
    # img = tf.reshape(img, shape=img_resolution)
    img = tf.io.decode_png(img, channels=3)
    cumulants = tf.reshape(cumulants, shape=(18, 1))
    inputs = (cumulants, img)
    outputs = mod
    return inputs, outputs

def parse_tfr_img_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'snr': tf.io.FixedLenFeature([], tf.int64),
        'mod': tf.io.FixedLenFeature([], tf.int64),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'cumulants': tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(element, data)

    snr = content['snr']
    mod = content['mod']
    cumulants = content['cumulants']
    raw_image = content['raw_image']

    # get our 'feature'-- our image -- and reshape it appropriately
    img = tf.io.parse_tensor(raw_image, out_type=tf.string)
    cumulants = tf.io.parse_tensor(cumulants, out_type=tf.float64)
    # TODO: KEEP IMG DIMS IN TFRECORDS ENTRY
    # img = tf.reshape(img, shape=img_resolution)
    img = tf.io.decode_png(img, channels=3)
    inputs = img
    outputs = mod
    return inputs, outputs


def get_dataset(filenames, ordered=False):
    # create the dataset
    AUTOTUNE = tf.data.AUTOTUNE
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element, num_parallel_calls=AUTOTUNE
    )

    return dataset

def get_image_dataset(filenames, ordered=False):
    # create the dataset
    AUTOTUNE = tf.data.AUTOTUNE
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_img_element, num_parallel_calls=AUTOTUNE
    )

    return dataset


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

# Loads data off HDD
def load_hybrid_data(mod_schemes, snrs, img_height, img_width, batch_size):
    # === Parameters ===
    # Dataset directories
    train_img_dir = pathlib.Path('hybrid_dataset/training/images')
    train_cum_dir = pathlib.Path('hybrid_dataset/training/cumulants')
    # test_img_dir = pathlib.Path('hybrid_dataset/test')

    # Training dataset structure
    train_ds = hybrid_dataset_from_directory(
        train_img_dir,
        train_cum_dir,
        # labels=train_labels.tolist(),
        label_mode='int',
        validation_split=0.2,
        subset="training",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        seed=123)

    # Validation dataset structure
    val_ds = hybrid_dataset_from_directory(
        train_img_dir,
        train_cum_dir,
        # labels=train_labels.tolist(),
        label_mode='int',
        validation_split=0.2,
        subset="validation",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        seed=123)

    test_ds = None
    # # Test dataset structure
    # test_ds = hybrid_dataset_from_directory(
    #     test_img_dir,
    #     # labels=test_labels.tolist(),
    #     label_mode='int',
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size,
    #     shuffle=False)
    test_labels = None
    # # Get testset labels
    # test_labels = np.concatenate([y for x, y in test_ds], axis=0)

    return train_ds, val_ds, test_ds, test_labels
