import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

import time
import os
import math

from data import read_data, create_binary_data, remove_nominal_samples
from models import binary_model, anomaly_model

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

        Args:
            batch_size (int): the batch size to be used by the generator
            test_batch_size (int): as above but for test data
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


def data_prep_anomalous_train(predicted_anomalous_x, predicted_anomalous_y, batch_size=128, test_batch_size=2321):
    """ Prepare the data for training for the second anomalous model.

        Args:
            batch_size (int): the batch size to be used by the generator
            test_batch_size (int): as above but for test data
    """
    _, _, x_valid, y_valid, x_test, y_test = read_data(DATA_PATH)

    train_dataset_x = tf.data.Dataset.from_tensor_slices(predicted_anomalous_x).batch(batch_size)
    train_dataset_y = tf.data.Dataset.from_tensor_slices(predicted_anomalous_y).batch(batch_size)

    x_test, y_test = remove_nominal_samples(x_test, y_test)
    test_dataset_x = tf.data.Dataset.from_tensor_slices(x_test).batch(test_batch_size)
    test_dataset_y = tf.data.Dataset.from_tensor_slices(y_test).batch(test_batch_size)

    x_valid, y_valid = remove_nominal_samples(x_valid, y_valid)
    valid_dataset_x = tf.data.Dataset.from_tensor_slices(x_valid).batch(batch_size)
    valid_dataset_y = tf.data.Dataset.from_tensor_slices(y_valid).batch(batch_size)

    return train_dataset_x, train_dataset_y, test_dataset_x, test_dataset_y, valid_dataset_x, valid_dataset_y


def train():
    """ Trains the hierarchial model as follows:
        1. Train the binary model.
        2. Extract anomalous samples from the prediction.
        3. Train the anomalous model on the extracted samples.
    """

    epochs_binary = 1
    epochs_anomalous = 1
    train_losses_backup_bin, valid_losses_backup_bin = [], []
    train_losses_backup_an, valid_losses_backup_an = [], []

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
                print('\nTaking a backup of the predictions. This may take some time....')
                _, y_pred = compute_loss_and_batch_predictions(model, x, y)
                y_pred_class = y_pred.numpy().argmax(1)
                for i, _y in enumerate(y_pred_class):
                    # backup every sample that is not nominal
                    if _y != 0:
                        predicted_anomalous_x = np.append(predicted_anomalous_x, [x[i]], axis=0)
                        predicted_anomalous_y = np.append(predicted_anomalous_y, [true_y[i][1:]], axis=0)

        print('\nBinary Model :: Average Epoch %s Train Loss: %s' % (str(epoch), str(train_loss/num_batches)))

        num_valid_batches, valid_loss = 0, 0
        for x, y in zip(valid_dataset_x, valid_dataset_y):
            valid_loss += compute_loss(model, x, y).numpy()
            num_valid_batches += 1

        print('Binary Model :: Average Epoch %s Validation Loss: %s' % (str(epoch), str(valid_loss/num_valid_batches)))

        train_losses_backup_bin.append(train_loss/num_batches)
        valid_losses_backup_bin.append(valid_loss/num_valid_batches)

        for x, y in zip(test_dataset_x, test_dataset_y):
            _, y_pred_test = compute_loss_and_batch_predictions(model, x, y)
            print('Binary Model :: Test Accuracy: %s' % str(accuracy_score(y.numpy().argmax(1), y_pred_test.numpy().argmax(1))))
            print('Binary Model :: Classification Report (below)')
            print(classification_report(y.numpy().argmax(1), y_pred_test.numpy().argmax(1)))

        
    # At this point, the binary model has finished training and we need to plug its results
    # into the anomalous model
    (train_dataset_x,
    train_dataset_y,
    test_dataset_x,
    test_dataset_y,
    valid_dataset_x,
    valid_dataset_y) = data_prep_anomalous_train(predicted_anomalous_x, predicted_anomalous_y)

    model, optimizer = anomaly_model()

    for epoch in tqdm(range(1, epochs_anomalous + 1)):
        num_batches, train_loss = 0, 0
        for x, y in zip(train_dataset_x, train_dataset_y):
            train_loss += train_step(model, x, y, optimizer)
            num_batches += 1

        print('\nAnomalous Model :: Average Epoch %s Train Loss: %s' % (str(epoch), str(train_loss/num_batches)))

        num_valid_batches, valid_loss = 0, 0
        for x, y in zip(valid_dataset_x, valid_dataset_y):
            valid_loss += compute_loss(model, x, y).numpy()
            num_valid_batches += 1

        print('Anomalous Model :: Average Epoch %s Validation Loss: %s' % (str(epoch), str(valid_loss/num_valid_batches)))
    
    train_losses_backup_an.append(train_loss/num_batches)
    valid_losses_backup_an.append(valid_loss/num_valid_batches)

    for x, y in zip(test_dataset_x, test_dataset_y):
        _, y_pred_test = compute_loss_and_batch_predictions(model, x, y)
        print('Binary Model :: Test Accuracy: %s' % str(accuracy_score(y.numpy().argmax(1), y_pred_test.numpy().argmax(1))))
        print('Binary Model :: Classification Report (below)')
        print(classification_report(y.numpy().argmax(1), y_pred_test.numpy().argmax(1)))
