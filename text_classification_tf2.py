from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


if not os.path.exists('imdb_reviews/subwords8k'):
    data, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
#print('data', data)
#print('info', info)

train_data, test_data = data['train'], data['test']

tokenizer = info.features['text'].encoder
print('Vocabulary size: {}'.format(tokenizer.vocab_size))

#Example of tokenizer
# sample_string = 'tensorflow is cool and awesome.'
# tokenized_string = tokenizer.encode(sample_string)
# print('encoded tokenized string : ', tokenized_string)
# original_string = tokenizer.decode(tokenized_string)
# print('Original string : ', original_string)
#
# for ts in tokenized_string:
#     print('{} --- >  {}'.format(ts, tokenizer.decode([ts])))
#Example Ends

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_data = train_data.shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, train_data.output_shapes)

test_data = test_data.padded_batch(BATCH_SIZE, test_data.output_shapes)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_data, epochs=10, validation_data=test_data)

test_loss, test_acc = model.evaluate(test_data)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def sample_predict(sentence, pad):
    tokenized_sample_pred_text = tokenizer.encode(sentence)

    if pad:
        tokenized_sample_pred_text = pad_to_size(tokenized_sample_pred_text, 64)

    predictions = model.predict(tf.expand_dims(tokenized_sample_pred_text, 0))

    return predictions

# predict on a sample text without padding.

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)


# predict on a sample text with padding

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print (predictions)

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
