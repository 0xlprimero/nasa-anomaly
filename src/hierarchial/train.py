import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layer
from tensorflow.keras import activations
import tensorflow_probability as tfp
import time
from tqdm import tqdm

import os

from src.data import read_data, create_binary_data
from .models import binary_model

DATA_PATH = os.environ['NASA_DATASET']

def compute_loss(model, x, y):
    """ Compute categorical cross-entropy loss

        Args:
            model (keras.model): the keras model object for the model
            x (np.array): features
            y (np.array): ground-truth output
    """
    cc_loss = tf.keras.losses.CategoricalCrossentropy()
    y_pred = model(x)

    return cc_loss(y, y_pred)


def compute_loss_and_batch_predictions(model, x, y):
    """ Compute categorical cross-entropy loss

        Args:
            model (keras.model): the keras model object for the model
            x (np.array): features
            y (np.array): ground-truth output
    """
    cc_loss = tf.keras.losses.CategoricalCrossentropy()
    y_pred = model(x)

    return cc_loss(y, y_pred), y_pred


def train_step(model, x, y, optimizer):
    """ Main training routine that performs gradient adjustment as well.

        Args:
            model (keras.model): the keras model object for the model
            x (np.array): features
            y (np.array): groud-truth output
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss.numpy()


def data_prep_binary_train(batch_size=128, test_batch_size=20263):
    """ Prepare the data for training for the first binary model.
    """
    x_train, y_train, x_valid, y_valid, x_test, y_test = read_data(DATA_PATH)

    x_train_binary, y_train_binary = create_binary_data(x_train, y_train)
    x_test_binary, y_test_binary = create_binary_data(x_test, y_test)
    x_valid_binary, y_valid_binary = create_binary_data(x_valid, y_valid)

    train_dataset_x = tf.data.Dataset.from_tensor_slices(x_train_binary).batch(batch_size)
    train_dataset_y = tf.data.Dataset.from_tensor_slices(y_train_binary).batch(batch_size)
    truth_dataset_y = tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size)

    test_dataset_x = tf.data.Dataset.from_tensor_slices(x_test_binary).batch(test_batch_size)
    test_dataset_y = tf.data.Dataset.from_tensor_slices(y_test_binary).batch(test_batch_size)

    valid_dataset_x = tf.data.Dataset.from_tensor_slices(x_valid_binary).batch(batch_size)
    valid_dataset_y = tf.data.Dataset.from_tensor_slices(y_valid_binary).batch(batch_size)

    return train_dataset_x, train_dataset_y, truth_dataset_y, test_dataset_x, test_dataset_y, valid_dataset_x, valid_dataset_y


def train():
    """
    """

    epochs_binary = 100
    epochs_anomalous = 250

    tf.keras.backend.clear_session()

    (train_dataset_x,
    train_dataset_y,
    truth_dataset_y,
    test_dataset_x,
    test_dataset_y,
    valid_dataset_x,
    valid_dataset_y) = data_prep_binary_train()

    model, optimizer = binary_model()

    for epoch in tqdm(range(1, epochs_binary + 1)):
        num_batches, train_loss = 0, 0
        predicted_anomalous_x, predicted_anomalous_y = np.empty((0, 160, 21), float), np.empty((0, 6), float)

        for x, y, true_y in zip(train_dataset_x, train_dataset_y, truth_dataset_y):
            train_loss += train_step(model, x, y, optimizer)
            num_batches += 1

            # the snippet under the if condition takes a snapshot of the 
            # predictions in the last epoch of training of the binary model
            if epoch == epochs_binary:
                _, y_pred = compute_loss_and_batch_predictions(model, x, y)
                y_pred_class = y_pred.numpy().argmax(1)
                for i, _y in enumerate(y_pred_class):
                    # backup every sample that is not nominal
                    if _y != 0:
                        predicted_anomalous_x = np.append(predicted_anomalous_x, [x[i]], axis=0)
                        predicted_anomalous_y = np.append(predicted_anomalous_y, [true_y[i][1:]], axis=0)

        print('\nAverage Epoch %s Train Loss: %s' % (str(epoch), str(train_loss/num_batches)))

        num_valid_batches, valid_loss = 0, 0
        for x, y in zip(valid_dataset_x, valid_dataset_y):
            valid_loss += compute_loss(model, x, y).numpy()
            num_valid_batches += 1

        print('\nAverage Epoch %s Validation Loss: %s' % (str(epoch), str(valid_loss/num_valid_batches)))