from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np


# Load training and eval data
(train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_data = train_data.reshape(60000,28,28,1)
eval_data = eval_data.reshape(10000,28,28,1)
train_data = train_data/np.float32(255)
eval_data = eval_data/np.float32(255)

print(train_data.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(eval_data, eval_labels)
print(test_acc)

