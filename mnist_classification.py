import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
tf.compat.v1.disable_eager_execution()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train , x_test = x_train/255, x_test/255
print("Size of Training-set:\t\t{}".format(len(x_train)))
print("Size of Test-set:\t\t{}".format(len(x_test)))
#print(x_train[0], y_train[0])
print(y_train[0].shape)
#x = tf.placeholder(tf.float32, [None, 784]) # Don't need placeholder in TF2.0


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)