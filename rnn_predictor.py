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
        self.gru1 = tf.keras.layers.GRU(units, return_sequences=True)
        self.gru2 = tf.keras.layers.GRU(units, return_sequences=True)
        self.gru3 = tf.keras.layers.GRU(units, return_sequences=True)
        self.gru4 = tf.keras.layers.GRU(units)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.gru1(inputs)
        x = self.gru2(x)
        x = self.gru3(x)
        x = self.gru4(x)
        x = self.dense(x)

        return x
