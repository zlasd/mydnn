
import os
import time
import shutil

from util.dataset import CIFAR10
from util.cifar10tf import CIFAR10TF
from models.manager import Manager, ManagerTF
from models.softmax import SoftmaxConfig, softmax
from models.googlenet import googlenet, GoogLeNetConfig

from convert_to_tfrecord import data_to_tfrecord

def testLoadDataset():
    print('test load datasets...')

    t1 = time.time()
    cifar10 = CIFAR10('data/')
    cifar10.loadTrainData()
    t2 = time.time()
    print('time consuming: {}'.format(t2-t1))
    print('image shapes: {}'.format(cifar10.train_X[0].shape))
    print('train sample: {}'.format(cifar10.train_y.shape[0]))
    print('valid sample: {}'.format(cifar10.valid_y.shape[0]))


def testSoftmax():
    conf = SoftmaxConfig()
    conf.EPOCH = 20
    conf.SAVE_PER_EPOCH = 5
    cifar10 = CIFAR10('data/')
    manager = Manager(conf, cifar10, softmax)

    manager.compile()
    manager.training()


def testGoogLeNet():
    conf = GoogLeNetConfig()
    conf.EPOCH = 100
    conf.SAVE_PER_EPOCH = 10
    conf.LEARNING_RATE = 1e-2
    cifar10 = CIFAR10('data/')
    manager = Manager(conf, cifar10, googlenet)

    manager.compile()
    manager.training()

def testToRecord():
    cifar10 = CIFAR10('data/')
    cifar10.loadTrainData()
    cifar10.loadTestData()

    data_to_tfrecord(cifar10.train_X, cifar10.train_y, "data/train")
    data_to_tfrecord(cifar10.valid_X, cifar10.valid_y, "data/valid")
    data_to_tfrecord(cifar10.test_X, cifar10.test_y, "data/test")

def testManagerTF():
    conf = GoogLeNetConfig()
    conf.EPOCH = 100
    conf.SAVE_PER_EPOCH = 10
    conf.LEARNING_RATE = 1e-2

    cifar10 = CIFAR10TF('data/')
    manager = ManagerTF(conf, cifar10, googlenet)

    manager.compile()
    manager.training()

def testTFRecord():
    import tensorflow as tf

    cifar10 = CIFAR10TF('data/')
    cifar10.buildIterator(32)

    tensor_X, tensor_y = cifar10.iterator.get_next()
    tensor_y = tf.squeeze(tensor_y)

    sess = tf.Session()
    sess.run(cifar10.train_iter_init)

    X, y = sess.run([tensor_X, tensor_y])
    print(X.shape)
    print(y.shape)

    


def test():
    # testSoftmax()
    # testTFRecord()
    testManagerTF()


if __name__ == '__main__':
    test()
