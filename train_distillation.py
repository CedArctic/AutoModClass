import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import seaborn as sn

from utils.plotting import plotAccLoss
from utils.data import load_data
from transferLearning.resnet import resnet
from knowledge_distillation.distiller import Distiller

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# === Load Data ===
STUDENT_MODEL_NAME = 'VGG-BRANCHES'
TEACHER_MODEL_NAME = 'VGG-BRANCHES'
print('Training Student Model {} using {} as the Teacher Model'.format(STUDENT_MODEL_NAME, TEACHER_MODEL_NAME))
# Dataset Parameters
img_height = 224
img_width = 224
img_channels = 3
batch_size = 100

# Modulation Schemes and SNRs sorted alphanumerically (the way TensorFlow reads training samples)
mod_schemes = ['16APSK', '16PAM', '16QAM', '4PAM', '64APSK', '64QAM', '8PSK', 'QPSK']
mod_schemes.sort()
snrs = [0, 5, 10, 15]
snrs.sort()

# Load data
#TODO: Separate method for test datasets loading to improve performance
train_ds, val_ds, test_ds, test_labels = load_data(mod_schemes, snrs, img_height, img_width, batch_size)

# Dataset caching and prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# === Training ===

# Parameters
epochs = 30

# Create student model, print summary and plot
#TODO: Switch resnet from resnet 101 to resnet 50
student_model = resnet(img_height, img_width, std_input=False)
student_model.summary()
keras.utils.plot_model(student_model, "student_cnn.png", show_shapes=True)

# Compile model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Import Teacher model
teacher_model = keras.models.load_model('trained_models/{}'.format(TEACHER_MODEL_NAME))


# Early Stopping and Model Checkpoint callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
if not os.path.isdir("checkpoints"):
    os.makedirs('checkpoints')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"checkpoints/{STUDENT_MODEL_NAME}_BEST",
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# Distillation
distiller = Distiller(student=student_model, teacher=teacher_model)
distiller.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    # else just 'accuracy' should do the job
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# Training: Distill teacher to student
history = distiller.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data=val_ds,
              callbacks=[early_stop, model_checkpoint_callback])
# distiller.fit(x_train, y_train, epochs=3)

# Evaluate student on test dataset
# distiller.evaluate(x_test, y_test)


# Save Model
if not os.path.isdir("trained_models"):
    os.makedirs('trained_models')
student_model.save('trained_models/distillation/{}'.format(STUDENT_MODEL_NAME))

# Save training history
if not os.path.isdir("history"):
    os.makedirs('history')
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

np.savetxt('history/train_accuracy.csv', train_accuracy, delimiter=',', fmt='%d', header='Training Accuracy')
np.savetxt('history/val_accuracy.csv', val_accuracy, delimiter=',', fmt='%d', header='Validation Accuracy')
np.savetxt('history/train_loss.csv', train_loss, delimiter=',', fmt='%d', header='Training Loss')
np.savetxt('history/val_loss.csv', val_loss, delimiter=',', fmt='%d', header='Validation Loss')

# Save history as dictionary
if not os.path.isdir("history/{}".format(STUDENT_MODEL_NAME)):
    os.makedirs('history/{}'.format(STUDENT_MODEL_NAME))
with open('history/{}/trainHistoryDict'.format(STUDENT_MODEL_NAME), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# === Inference ===
# Import model
model = keras.models.load_model('trained_models/{}'.format(STUDENT_MODEL_NAME))

# Make predictions
print("Inference started")
predictions = model.predict(test_ds)

# Generate and plot confusion matrix
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
sn.heatmap(cm, annot=True, fmt='g')

# Plot training and validation accuracy and losses
history = pickle.load(open('history/{}/trainHistoryDict'.format(STUDENT_MODEL_NAME), "rb"))
plotAccLoss(STUDENT_MODEL_NAME, history['accuracy'], history['val_accuracy'], history['loss'], history['val_loss'])
