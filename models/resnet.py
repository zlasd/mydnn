# pylint: disable=E0611

import numpy as np
import tensorflow as tf

from tensorflow.layers import conv2d, max_pooling2d, \
                     batch_normalization, flatten, dense
from tensorflow.contrib.layers import xavier_initializer

from util.config import Config


class ResnetConfig(Config):
    name = "resnet"
    LEARNING_RATE = 0.01
    EPOCH = 300
    SAVE_PER_EPOCH = 10

def ConvLayer(x, c_out, ks, stride=1):
    y = conv2d(x, c_out, ks, strides=stride, padding="SAME",
                kernel_initializer=xavier_initializer())
    return batch_normalization(y)


def SqueezeExcitation(x, c_out, ratio):
    """ Squeeze and excitation building block
        <https://arxiv.org/pdf/1709.01507.pdf>
    """
    with tf.name_scope("SEBlock"):
        squeeze = tf.reduce_mean(x, axis=[1,2])

        excitation = dense(squeeze, c_out//ratio, activation=tf.nn.relu)
        excitation = dense(excitation, c_out, activation=tf.sigmoid)
        excitation = tf.reshape(excitation, [-1, 1, 1, c_out])

        scale = x * excitation
    return scale


def residualBlock(x, c_out, stride=1, useSE=True, ns=None):
    """ residual block
    Args:
        x input tensor
        c_out output channels
        stride first conv layer's stride
        ns name scope
    Return:
        residual tensor plus input tensor
    """
    if ns == None:
        ns = "ResBlock"

    c_in = x.get_shape().as_list()[3]
    with tf.name_scope(ns):
        shortcut = x

        y = ConvLayer(x, c_out, 3, stride=stride)
        y = tf.nn.relu(y)
        y = ConvLayer(y, c_out, 3)

        if stride != 1:
            shortcut = ConvLayer(x, c_out, 3, stride=stride)
        elif c_in != c_out:
            shortcut = ConvLayer(x, c_out, 1)

        if useSE:
            y = SqueezeExcitation(y, c_out, 4)

        y = shortcut + y
        y = tf.nn.relu(y)
    
    return y


def make_layer(block_fn, X, c_out, nlayers, stride=1):
    """ make several layers using block_fn
    Args:
        block_fn building block function
        X input tensor
        c_out output channel
        nlayers num of layers (>= 1)
        stride the first layer's stride
    Return:
        output tensor
    """
    with tf.name_scope("ResLayer"):
        y = block_fn(X, c_out, stride=stride)
        for _ in range(nlayers-1):
            y = residualBlock(y, c_out)
    return y



def resnet(X, config):
    """ resnet for cifar10 introduced by original paper
        <http://arxiv.org/abs/1512.03385>
    """
    with tf.name_scope(config.name):
        stem = ConvLayer(X, 16, 3)
        res1 = make_layer(residualBlock, stem, 16, 3)
        res2 = make_layer(residualBlock, res1, 32, 3, stride=2)
        res3 = make_layer(residualBlock, res2, 64, 3, stride=2)

        avg_global_pool = tf.reduce_mean(res3, axis=[1,2])
        logits = flatten(avg_global_pool)
        logits = dense(logits, config.NUM_CLASSES)
    
    return logits, None
