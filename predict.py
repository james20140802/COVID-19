"""Predicts the number of confiremd cases."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import pandas as pd

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
    data = np.array([[467653], [529591], [593291], [660706], [720117]])
    date = 0
    history = []

    while True:
        last_data = data[-1]
        prediction = predict(np.float32([data])).numpy()[0][0]

        tmp_data = data[1:]
        data = np.append(tmp_data, [[prediction]], axis=0)

        date += 1
        print('Day: {:d} Expected Confirmed Cases: {:f}'.format(date, prediction))
        history.append(prediction)

        if (prediction - last_data) < 10:
            break

    data_frame = pd.DataFrame(data)
    data_frame.to_csv("history.csv", header=False, index=False)
