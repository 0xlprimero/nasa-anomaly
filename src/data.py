import numpy as np
from sklearn.utils import shuffle

def read_data(data_path):
    """ Reads the data into numpy arrays

        Args:
            data_path (str): location of the dataset.npz
    """
    data = np.load(data_path)
    x_train, y_train = data['train_data'], data['train_label']
    x_valid, y_valid = data['valid_data'], data['valid_label']
    x_test, y_test = data['test_data'], data['test_label']

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def create_binary_data(x, y):
    """ Converts the dataset into a binary dataset of
        anomalous and nominal samples.

        Args:
            x (np.array): features
            y (np.array): output
    """
    x_binary, y_binary = [], []

    for _x, _y in zip(x, y):
        x_binary.append(_x)
        if _y.argmax() ==  0: # if the data is nominal
            y_binary.append(np.array([1.0, 0.0]))
        else:
            y_binary.append(np.array([0.0, 1.0]))

    return np.array(x_binary), np.array(y_binary)


def remove_nominal_samples(x, y):
    """ Removes the nominal samples from the dataset.

        Args:
            x (np.array): features
            y (np.array): output
    """
    x_anomalous, y_anomalous = [], []

    for _x, _y in zip(x, y):
        if _y.argmax() != 0: # if the data is anomalous
            x_anomalous.append(_x)
            y_anomalous.append(_y[1:]) # remove the one-hot encoding for the nominal samples

    return np.array(x_anomalous), np.array(y_anomalous)


def one_hot_encode(y_idx):
    """ Converts a label to a list. For example, `4` becomes `[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]`.

        Args:
            y_idx (int): the label that needs to be one-hot encoded.
    """
    label = np.zeros(7)
    label[y_idx] = 1

    return label


def pick_nominal_samples_at_random(percentage, x, y, num_nominal=53834):
    """ Picks a percentage of nominal samples at random.

        Args:
            percentage (int): the percentage of nominal samples that are desired
            x (np.array): the features
            y (np.array): the output
            num_nominal (int): the total number of nominal samples in the dataset.
    """
    x_sampled, y_sampled = [], []
    num_pick = int(percentage * 0.1 * num_nominal)
    cnt = 0
    x, y = shuffle(x, y)
    for _x, _y in zip(x, y):
        if cnt == num_pick and _y.argmax() == 0:
            # we have reached the desired number of nominal samples
            # and do not need to collect them any further
            continue
        elif _y.argmax() == 0:
            x_sampled.append(_x)
            y_sampled.append(_y)
            cnt += 1
        else:
            x_sampled.append(_x)
            y_sampled.append(_y)

    return np.array(x_sampled), np.array(y_sampled)