"""Predicts the number of confiremd cases."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

from rnn_predictor import RNNPredictor


def predict(x):
    predictor = RNNPredictor(256)
    predictor.load_weights("./checkpoint/rnn_predictor")

    f = open('mean_std.txt', 'rb')
    mean_std = pickle.load(f)
    f.close()
    x = (x - mean_std["mean"]) / mean_std["std"]
    y = predictor(x)

    return (y * mean_std["std"]) + mean_std["mean"]


if __name__ == "__main__":
    data = [[[418045], [467653], [529591], [593291], [660706]]]

    print(predict(data))

