# This code is from the following stackoverflow answer
# https://stackoverflow.com/questions/61974271/how-do-i-train-a-neural-network-with-tensorflow-datasets

import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


train_data, validation_data = tfds.load('emnist/letters',
                                        split=['train', 'test'],
                                        shuffle_files=True,
                                        batch_size=32,
                                        as_supervised=True)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(47, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=20)

history = model.fit(train_data,
                    epochs=50,
                    validation_data=validation_data,
                    callbacks=[early_stopping],
                    verbose=1)
print("Saving model")
model.save('emnistmodel.h5')