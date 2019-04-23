from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
#tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow import strided_slice
#from tensorflow.python.keras import activations
#from tensorflow.python.keras.engine import training_arrays


def cnn_model(features, labels, mode):

    #Input Layer
    #print(features["x"])
    #print(features.shape)
    input_layer = tf.reshape(features["x"], [-1, 28,28,1])

    #Convolutional Layer #1
    conv1 = tf.compat.v1.layers.conv2d(
        input_layer,
        filters = 32,
        kernel_size = [5,5],
        padding="same",
        activation='relu')

    #Maxpool Layer #1
    pool1 = tf.compat.v1.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    #Convolutional Layer #2
    conv2 = tf.compat.v1.layers.conv2d(
        pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation='relu')

    pool2 = tf.compat.v1.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    #Dense Layer

    flatten_data = tf.reshape(pool2, [-1, 7, 7, 64])
    dense = tf.compat.v1.layers.Dense(flatten_data, 1024, activation=tf.nn.relu)
    dropout = tf.compat.v1.layers.dropout(dense, rate=0.4, training= (mode == tf.estimator.ModeKeys.TRAIN))

    #Logit Layer
    logits = tf.compat.v1.layers.dense(dropout, 10)

    predictions = {
        #Generate predictions ( for PREDICT and EVAL Mode)
        "classes" : tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.

        "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.Estimator(mode=mode, predictions=predictions)

    #Calculate Loss

    loss = tf.losses.sparse_softmax_cross_entroty(labels=labels, logits=logits)

    #Configure the training Operation (Train Mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
        train_operation = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, train_op = train_operation)

    eval_metric_ops = {
        "accuracy" : tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Load training and eval data
((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
eval_data = eval_data/np.float32(255)

#create the estimator
mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir="/tmp/mnist_convnet_model")

# tensors_to_log = {"probabilities": "softmax_tensor"}
# logging_hook = tf.estimator.LoggingTensorHook(
#     tensors=tensors_to_log, every_n_iter=50)


#Train the model
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x":train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

# train one step and display the probabilties
mnist_classifier.train(input_fn=train_input_fn, steps=1)

eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
