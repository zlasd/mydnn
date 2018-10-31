

import numpy as np
import tensorflow as tf

from tensorflow.initializers import random_normal

from util.config import Config


class SoftmaxConfig(Config):
    NAME = "softmax"
    LEARNING_RATE = 0.001


def softmax(X, config):
    """ build softmax model
    """
    X_in =  tf.layers.flatten(X)

    W = tf.get_variable("W", shape=(X_in.shape.as_list()[1], config.NUM_CLASSES),
            initializer=random_normal, dtype=tf.float32)
    b = tf.get_variable("b", shape=(config.NUM_CLASSES, ))

    logits = tf.nn.softmax(tf.add(tf.matmul(X_in, W), b))
    
    return logits
