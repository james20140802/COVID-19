"""Trains the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from data_loader import load_train_data, univariate_data
from rnn_predictor import RNNPredictor

raw_data = load_train_data()
data, labels = univariate_data(raw_data["ConfirmedCases"], 0, None, 5, 0)

val_data, val_labels = [list(data[20]), list(data[40]), list(data[63])], [labels[20], labels[40], labels[63]]
train_data, train_labels = np.delete(data, [20, 40, 63], axis=0), np.delete(labels, [20, 40, 63], axis=0)

val_data_set = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
data_set = tf.data.Dataset.from_tensor_slices((train_data, train_labels))

data_set = data_set.shuffle(len(list(data_set.as_numpy_iterator())), reshuffle_each_iteration=True)
data_set = data_set.repeat(5)

predictor = RNNPredictor(256)

epochs = range(10000)

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(lr=1e-2, clipvalue=1)


def loss(model, x, y, training=True):
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


train_loss_results = []
best_loss = float("inf")
same_loss_count = 0

if tf.io.gfile.exists("./checkpoint/rnn_predictor.index"):
    predictor.load_weights("./checkpoint/rnn_predictor")

for epoch in epochs:
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_val_loss_avg = tf.keras.metrics.Mean()

    for data, labels in data_set.batch(32):
        loss_value, grads = grad(predictor, tf.cast(data, tf.float32), tf.cast(labels, tf.float32))
        optimizer.apply_gradients(zip(grads, predictor.trainable_variables))

        epoch_loss_avg(loss_value)

    train_loss_results.append(epoch_loss_avg)

    for data, label in val_data_set.batch(1):
        val_loss = loss(predictor, [data], [label], training=False)
        epoch_val_loss_avg(val_loss)

    if epoch_val_loss_avg.result() < best_loss:
        predictor.save_weights("./checkpoint/rnn_predictor")
        best_loss = epoch_val_loss_avg.result()
        same_loss_count = 0
    else:
        same_loss_count += 1

    print("Epoch: {:d} Loss: {:f} Validation Loss: {:f}".format((epoch+1), epoch_loss_avg.result(),
                                                                epoch_val_loss_avg.result()))

    if same_loss_count > 500:
        print("End the training. Best Loss: {:f}.".format(best_loss))
        break
