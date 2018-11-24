
import numpy as np
import tensorflow as tf

from tensorflow.layers import conv2d, max_pooling2d, \
                        average_pooling2d, dropout, dense, flatten
from tensorflow.initializers import random_normal

from util.config import Config

class GoogLeNetConfig(Config):
    NAME = "googlenet"
    LEARNING_RATE = 0.01
    EPOCH = 300
    SAVE_PER_EPOCH = 10


inceptionCount = 0
def inceptionBlock(X_in, c1, c3_r, c3, c5_r, c5, p3_r, ns=None):
    """ inception building block
    Args:
        X_in the input tensor
        c1...p3_r conv layer filter size of inception,
                    '_r' means reduction 1x1 conv layer
        ns name scope
    """
    global inceptionCount
    inceptionCount += 1
    if ns is None:
        ns = "inception{}".format(inceptionCount)
    
    with tf.name_scope(ns):
        conv1 = conv2d(X_in, c1, 1, padding='SAME')

        conv3 = conv2d(X_in, c3_r, 1, padding='SAME')
        conv3 = tf.nn.relu(conv3)
        conv3 = conv2d(conv3, c3, 3, padding='SAME')

        conv5 = conv2d(X_in, c5_r, 1, padding='SAME')
        conv5 = tf.nn.relu(conv5)
        conv5 = conv2d(conv5, c5, 5, padding='SAME')

        maxpool = max_pooling2d(X_in, 3, 1, padding='SAME')
        maxpool = conv2d(maxpool, p3_r, 1, padding='SAME')

        logits = tf.concat([conv1, conv3, conv5, maxpool], 3)
        logits = tf.nn.relu(logits)

    return logits

def googlenet(X, config):
    """ googlenet implementation.
    """
    with tf.name_scope('googlenet'):
        block1 = conv2d(X, 64, 7, strides=2, padding='SAME')
        block1 = tf.nn.relu(block1)
        block1 = max_pooling2d(block1, 3, 2, padding='SAME')
        block1 = tf.nn.local_response_normalization(block1)

        block2 = conv2d(block1, 64, 1, padding='SAME')
        block2 = tf.nn.relu(block2)
        block2 = conv2d(block2, 192, 3, padding='SAME')
        block2 = tf.nn.relu(block2)
        block2 = tf.nn.local_response_normalization(block2)
        block2 = max_pooling2d(block2, 3, 2, padding='SAME')
        block2 = tf.nn.relu(block2)

        # inception x2
        block3 = inceptionBlock(block2, c1=64, c3_r=96,
                                 c3=128, c5_r=16, c5=32, p3_r=32)
        block3 = inceptionBlock(block3, c1=128, c3_r=128,
                                 c3=192, c5_r=32, c5=96, p3_r=64)
        block3 = max_pooling2d(block3, 3, 2, padding='SAME')

        # inception x5
        block4 = inceptionBlock(block3, c1=192, c3_r=96,
                                 c3=208, c5_r=16, c5=48, p3_r=64)
        block4 = inceptionBlock(block4, c1=160, c3_r=112,
                                 c3=224, c5_r=24, c5=64, p3_r=64)
        block4 = inceptionBlock(block4, c1=128, c3_r=128,
                                 c3=256, c5_r=24, c5=64, p3_r=64)
        block4 = inceptionBlock(block4, c1=112, c3_r=144,
                                 c3=288, c5_r=32, c5=64, p3_r=64)
        block4 = inceptionBlock(block4, c1=256, c3_r=160,
                                 c3=320, c5_r=32, c5=128, p3_r=128)
        block4 = max_pooling2d(block4, 3, 2, padding='SAME')

        # inception x2 with average pooling
        block5 = inceptionBlock(block4, c1=256, c3_r=160,
                                 c3=320, c5_r=32, c5=128, p3_r=128)
        block5 = inceptionBlock(block5, c1=384, c3_r=192,
                                 c3=384, c5_r=48, c5=128, p3_r=128)
        block5 = average_pooling2d(block5, 7, 1, padding='SAME')
        block5 = dropout(block5, rate=0.4, training=config.training)

        logits = flatten(block5)
        logits = dense(logits, config.NUM_CLASSES)
    
    return logits, None



