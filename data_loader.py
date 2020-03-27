"""Loads data from csv file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_train_data():
    data = pd.read_csv('./data/train.csv')

    return data


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        pre_data = []
        for j in indices:
            pre_data.append(dataset[j])
        data.append(np.reshape(pre_data, (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


if __name__ == "__main__":
    train_data = load_train_data()
    print(train_data.info())
    print(train_data.describe())
    print(train_data.head())
    plt.plot(train_data['Date'], train_data['ConfirmedCases'])
    plt.show()
    cases_data = train_data["ConfirmedCases"]
    data_set = univariate_data(cases_data, 0, None, 5, 0)
    print(data_set)
