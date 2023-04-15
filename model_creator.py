import numpy as np
from emnist import extract_training_samples, extract_test_samples
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
x_train, y_train = extract_training_samples('byclass')
x_test, y_test = extract_test_samples('byclass')

# Preprocess the dataset
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, 62)
y_test = keras.utils.to_categorical(y_test, 62)

# Data augmentation
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
])

# Create the neural network model
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    data_augmentation,
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(62, activation="softmax"),
])

# Learning rate scheduler
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# Train the model
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=30, validation_split=0.1)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save the model
model.save("emnist_letter_detection_optimized.h5")

