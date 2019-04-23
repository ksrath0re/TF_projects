from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as pt


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


if not os.path.exists('shakespeare.txt'):
    path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                           'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text_data = open(path_to_file, 'rb').read().decode(encoding='utf-8')

#print('Length of text data : {} characters'.format(len(text_data)))

#print(text_data[:250])

vocab = sorted(set(text_data))
#print('{} unique characters'.format(len(vocab)))

# we need to map strings to a numerical representation.
# Create two lookup tables: one mapping characters to numbers,
# and another for numbers to characters.

char2index = {u: i for i, u in enumerate(vocab)}
index2char = np.array(vocab)
#print(index2char)

text_as_int = np.array([char2index[c] for c in text_data])
# e.g. 'First Citizen' = [18 47 56 57 58  1 15 47 58 47 64 43 52]


# print('{')
# for char, _ in zip(char2index, range(20)):
#     print(' {:4s}: {:3d}, '.format(repr(char), char2index[char]))
# print('... \n}')

print('{} --- Characters mapped into int --> {}'.format(repr(text_data[:13]), text_as_int[:13]))

seq_length = 100
samples_per_epoch = len(text_data)//seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
#print(char_dataset)

# for i in char_dataset.take(5):
#     print(index2char[i])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
#print(sequence[0])

# for item in sequences.take(5):
#     #print(item)
#     print(repr(''.join(index2char[item])))

dataset = sequences.map(split_input_target)

# for input_example, target_example in dataset.take(1):
#     print('Input Data: ', repr((''.join(index2char[input_example]))))
#     print('Target Data: ', repr((''.join(index2char[target_example]))))

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#print(dataset)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)


for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

print(model.summary())

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

print(sampled_indices)

print('Input: ', repr(''.join(index2char[input_example_batch[0]])))
print('Next Predictions ', repr(''.join(index2char[sampled_indices])))


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " #(batch_size, sequence_length, vocab_size)")
print("scalar loss: ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 10

#history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))



def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2index[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(index2char[predicted_id])

  return (start_string + ''.join(text_generated))

#model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

print(generate_text(model, start_string=u"ROMEO: "))
