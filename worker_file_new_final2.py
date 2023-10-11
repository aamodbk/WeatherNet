import os
import sys
import json
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set TF_CONFIG
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["172.17.0.3:9002", "172.17.0.4:9003"]
    },
    'task': {'type': 'worker', 'index': 1}
})


def get_mnist_dataset():
    batch_size = 32
    num_val_samples = 10000

    # Return the MNIST dataset in the form of a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )

def get_station_dataset():
  batch_size = 128
  data_station1 = pd.read_csv('datastation1.csv')
  data_station2 = pd.read_csv('datastation2.csv')
   
  scaler = StandardScaler()
  window_length = 5
  ds = np.append(data_station1.values[:, 5:], data_station2.values[:, 5:], axis=0)
  X = []
  for i in range(window_length, len(ds)):
    X.append(scaler.fit_transform(ds[i-window_length:i, :]))
  X = np.array(X)

  n_features = 7
  encoding_dim = 5
  test_perc = 0.1
  test_samples = int(test_perc*X.shape[0])

  x_test = X[-test_samples:]
  x_train = X[:-test_samples]

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')

  x_train_deep = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
  x_test_deep = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

  return (
    tf.data.Dataset.from_tensor_slices((x_train_deep, x_train_deep)).batch(batch_size),
    tf.data.Dataset.from_tensor_slices(([], [])).batch(batch_size),
    tf.data.Dataset.from_tensor_slices((x_test_deep, x_test_deep)).batch(batch_size),
  )

def get_compiled_mdel_autoencoder():
  window_length = 5
  n_features = 7
  encoding_dim = 5
  input_window = tf.keras.Input(shape=(window_length*n_features,))

  # x = Dense(32, activation="relu")(input_window)
  # x = BatchNormalization()(x)
  x = tf.keras.layers.Dense(16, activation="relu")(input_window)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dense(8, activation='relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  encoded = tf.keras.layers.Dense(encoding_dim, activation='relu', name='encoded_out')(x)

  # x = Dense(32, activation='relu')(encoded)
  # x = BatchNormalization()(x)
  x = tf.keras.layers.Dense(8, activation="relu", name='decoder_in')(encoded)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dense(16, activation="relu")(x)
  x = tf.keras.layers.BatchNormalization()(x)
  decoded = tf.keras.layers.Dense(window_length*n_features, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)

  autoencoder = tf.keras.Model(input_window, decoded)
  autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

  return autoencoder


def get_compiled_model_mnist():
    # Make a simple 2-layer densely-connected neural network.
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

def get_compile_model():
  pass

def get_train_dataset():
  pass

def get_test_dataset():
  pass

# Open a strategy scope and create/restore the model.
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with strategy.scope():
  model = get_compiled_model_mnist()



train_dataset, val_dataset, test_dataset = get_mnist_dataset()

model.fit(train_dataset, epochs=2, validation_data=val_dataset)

model.evaluate(test_dataset)

