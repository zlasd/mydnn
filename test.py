
import os
import time
import shutil

from util.dataset import CIFAR10
from models.manager import Manager
from models.softmax import SoftmaxConfig, softmax
from models.googlenet import googlenet, GoogLeNetConfig

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


def test():
    # testSoftmax()
    testGoogLeNet()


if __name__ == '__main__':
    test()
