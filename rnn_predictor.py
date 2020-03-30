"""Predicts confirmed cases and fatalities by RNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class RNNPredictor(tf.keras.Model):
    """Predicts confirmed cases by using RNN."""

    def __init__(self, units):
        """Initialize the model."""

        super(RNNPredictor, self).__init__()

        self.units = units
        self.gru1 = tf.keras.layers.GRU(units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        self.gru2 = tf.keras.layers.GRU(units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        self.gru3 = tf.keras.layers.GRU(units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        self.gru4 = tf.keras.layers.GRU(units, dropout=0.2, recurrent_dropout=0.2)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.gru1(inputs, training=training)
        x = self.gru2(x, training=training)
        x = self.gru3(x, training=training)
        x = self.gru4(x, training=training)
        x = self.dense(x, training=training)

        return x
