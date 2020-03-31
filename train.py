"""Trains the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from data_loader import load_train_data, univariate_data
from rnn_predictor import RNNPredictor

raw_data = load_train_data()
data, labels = univariate_data(raw_data["ConfirmedCases"], 0, None, 5, 0)
val_data, val_label = data[-1], labels[-1]
train_data, train_labels = data[:-1], labels[:-1]
data_set = tf.data.Dataset.from_tensor_slices((train_data, train_labels))

predictor = RNNPredictor(256)

epochs = range(10000)

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(lr=1e-2)


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
val_loss = 0

if tf.io.gfile.exists("./checkpoint/rnn_predictor.index"):
    predictor.load_weights("./checkpoint/rnn_predictor")

for epoch in epochs:
    epoch_loss_avg = tf.keras.metrics.Mean()

    for data, labels in data_set.batch(32):
        loss_value, grads = grad(predictor, tf.cast(data, tf.float32), tf.cast(labels, tf.float32))
        optimizer.apply_gradients(zip(grads, predictor.trainable_variables))

        epoch_loss_avg(loss_value)

    train_loss_results.append(epoch_loss_avg)

    val_loss = loss(predictor, tf.expand_dims(val_data, 0), [val_label], training=False)

    if val_loss < best_loss:
        predictor.save_weights("./checkpoint/rnn_predictor")
        best_loss = val_loss
        same_loss_count = 0
    else:
        same_loss_count += 1

    print("Epoch: {:d} Loss: {:f} Validation Loss: {:f}".format((epoch+1), epoch_loss_avg.result(), val_loss))

    if same_loss_count > 1000:
        print("End the training. Best Loss: {:f}.".format(best_loss))
        break
