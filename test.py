
import os
import time
import shutil

from util.dataset import CIFAR10

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


def test():
    testLoadDataset()


if __name__ == '__main__':
    test()
