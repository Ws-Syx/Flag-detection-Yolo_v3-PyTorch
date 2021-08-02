import numpy as np


def save_dict(filename, dictionary):
    labels = np.array(list(dictionary.keys()))
    np.save(filename, labels)


def load_dict(filename):
    labels = np.load(filename)
    return {name: index for index, name in zip(range(len(labels)), labels)}


def load_class(filename):
    return np.load(filename)
