# pylint: disable=no-member

import os

import tensorflow as tf
import numpy as np

from util.dataset import CIFAR10

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def data_to_tfrecord(images, labels, name):
    """ convert images and labels to tf.Record format file
    """
    images = np.float32(images)
    num_examples = images.shape[0]
    
    filename = os.path.join(os.getcwd(), name+'.tfrecord')
    print('writing to {}...'.format(filename))

    options_gzip = tf.python_io.TFRecordOptions(
                        tf.python_io.TFRecordCompressionType.GZIP)

    with tf.python_io.TFRecordWriter(filename, options=options_gzip) as writer:
        for i in range(num_examples):
            image_stream = images[i].tobytes()
            example = tf.train.Example(features=tf.train.Features(
                feature={"raw":_bytes_feature(image_stream),
                         "label":_int64_feature(labels[i])}
            ))
            writer.write(example.SerializeToString())

def to_tfrecord():
    cifar10 = CIFAR10('data/')
    cifar10.loadTrainData()
    cifar10.loadTestData()

    data_to_tfrecord(cifar10.train_X, cifar10.train_y, "data/train")
    data_to_tfrecord(cifar10.valid_X, cifar10.valid_y, "data/valid")
    data_to_tfrecord(cifar10.test_X, cifar10.test_y, "data/test")

if __name__ == 'main':
    to_tfrecord()
