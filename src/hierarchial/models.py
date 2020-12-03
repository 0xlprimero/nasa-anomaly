import tensorflow as tf
import tensorflow.keras.layers as layer

BATCH_SIZE = 128
LATENT_DIM = 10
HIDDEN_DIM = 50
num_class = 7
TIMESTEPS = 160
FEATURES = 21

def binary_model():
    binary_world = tf.keras.Sequential()
    binary_world.add(layer.InputLayer((TIMESTEPS, FEATURES)))
    binary_world.add(layer.Conv1D(32, 3))
    binary_world.add(layer.TimeDistributed(
        layer.Dense(HIDDEN_DIM, activation='relu')
        ))
    binary_world.add(layer.TimeDistributed(
        layer.Dense(HIDDEN_DIM, activation='relu')
        ))
    binary_world.add(layer.TimeDistributed(
        layer.Dense(LATENT_DIM, activation='relu')
        ))
    binary_world.add(layer.LSTM(HIDDEN_DIM, activation='tanh', input_shape=(TIMESTEPS, HIDDEN_DIM), return_sequences=False))
    binary_world.add(layer.Dense(2, activation='softmax'))
    binary_world.build((TIMESTEPS, FEATURES))
    optimizer = tf.keras.optimizers.Adam(1e-4)
    return binary_world, optimizer


def anomaly_model():
    anomaly_world = tf.keras.Sequential()
    anomaly_world.add(layer.InputLayer((TIMESTEPS, FEATURES)))
    anomaly_world.add(layer.Conv1D(32, 3, activation='relu'))
    anomaly_world.add(layer.MaxPooling1D(pool_size=2, strides=1))
    anomaly_world.add(layer.TimeDistributed(
        layer.Dense(HIDDEN_DIM, activation='relu')
        ))
    anomaly_world.add(layer.TimeDistributed(
        layer.Dense(HIDDEN_DIM, activation='relu')
        ))
    anomaly_world.add(layer.TimeDistributed(
        layer.Dense(LATENT_DIM, activation='relu')
        ))
    anomaly_world.add(layer.LSTM(HIDDEN_DIM, activation='tanh', input_shape=(TIMESTEPS, HIDDEN_DIM), return_sequences=False))
    anomaly_world.add(layer.Dense(6, activation='softmax'))
    anomaly_world.build((TIMESTEPS, FEATURES))
    optimizer = tf.keras.optimizers.Adam(1e-4)
    return anomaly_world, optimizer