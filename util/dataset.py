
import os
import random

import numpy as np

from matplotlib.image import imread # can be replaced by imread in some else package.
from sklearn.model_selection import train_test_split


class Dataset(object):
    """ Inspired by https://github.com/matterport/Mask_RCNN
    The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use.
    """
    def __init__(self, data_dir):
        self.dir = data_dir
        self.train_dir = os.path.join(data_dir, "train")
        self.test_dir = os.path.join(data_dir, "test")
        self.classes = []
    
    def getBatch(self, bs, mode="train"):
        return None, None


class CIFAR10(Dataset):
    """ CIFAR-10 datasets interface.
    """
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self._loadClasses()        
        self.IMAGE_HEIGHT = 32
        self.IMAGE_WIDTH = 32
        self.train_idx = 0
        self.valid_idx = 0
        self.test_idx = 0


    def loadTrainData(self):
        """ Load cifar-10 training/validation data.
        The data will be stored in self.{train, valid}_{X, y}
        """
        train_list = os.listdir(self.train_dir)

        train_X = np.zeros([len(train_list), self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3])
        train_y = np.zeros([len(train_list)], dtype=np.uint8)

        # load training sample
        for i in range(len(train_list)):
            img_name = train_list[i]
            img_path = os.path.join(self.train_dir, img_name)
            img_class = img_name.split(".")[0].split("_")[1]
            train_X[i] = imread(img_path)
            train_y[i] = self.classes.index(img_class)

        # randomly choose samples
        rnd_idx = np.random.permutation(train_X.shape[0])
        train_X = train_X[rnd_idx]
        train_y = train_y[rnd_idx]

        self.train_X, self.valid_X, self.train_y, self.valid_y \
                = train_test_split(train_X, train_y, test_size=0.2)

        print('image shapes: {}'.format(self.train_X[0].shape))
        print('image data type: {}'.format(self.train_X.dtype))
        print('train sample: {}'.format(self.train_y.shape[0]))
        print('valid sample: {}'.format(self.valid_y.shape[0]))


    def discardTrainData(self):
        """ Discard training data to reduce memory consumption.
        """
        del self.train_X
        del self.train_y
        del self.valid_X
        del self.valid_y


    def loadTestData(self):
        """Load cifar-10 test data
        """
        test_list = os.listdir(self.test_dir)

        test_X = np.zeros([len(test_list), 32, 32, 3])
        test_y = np.zeros([len(test_list)], dtype=np.uint8)

        # load test sample
        for i in range(len(test_list)):
            img_name = test_list[i]
            img_path = os.path.join(self.test_dir, img_name)
            img_class = img_name.split(".")[0].split("_")[1]
            test_X[i] = imread(img_path)
            test_y[i] = self.classes.index(img_class)
            
        # randomly choose samples
        rnd_idx = np.random.permutation(test_X.shape[0])
        self.test_X = test_X[rnd_idx]
        self.test_y = test_y[rnd_idx]

        print('image shapes: {}'.format(self.test_X[0].shape))
        print('test sample: {}'.format(self.test_y.shape[0]))



    def discardTestData(self):
        """ Discard test data
        """
        del self.test_X
        del self.test_y


    def getBatch(self, bs, mode="train"):
        """ randomly sampling a mini-batch from train/valid/test set.
        """
        assert mode in ["train", "valid", "test"]

        if mode == "train":
            idx = np.random.choice(self.train_y.shape[0], bs)
            return self.train_X[idx], self.train_y[idx]
        elif mode == "valid":
            idx = np.random.choice(self.valid_y.shape[0], bs)
            return self.valid_X[idx], self.valid_y[idx]
        else:
            idx = np.random.choice(self.test_y.shape[0], bs)
            return self.test_X[idx], self.test_y[idx]


    def nextBatch(self, bs, mode="train"):
        """ traverse dataset from the begining
        """
        assert mode in ["train", "valid", "test"]

        if mode == "train":
            start = self.train_idx
            end = start + bs
            self.train_idx += bs
            X = self.train_X
            y = self.train_y
        elif mode == "valid":
            start = self.valid_idx
            end = start + bs
            self.valid_idx += bs
            X = self.valid_X
            y = self.valid_y
        else:
            start = self.test_idx
            end = start + bs
            self.test_idx += bs
            X = self.test_X
            y = self.test_y
        return (X[start:end], y[start:end]) \
                if end <= X.shape[0] else False
        

    def resetIdx(self, mode="train"):
        """ reset dataset's index for traverse
        """
        assert mode in ["train", "valid", "test"]

        if mode == "train":
            self.train_idx = 0
        elif mode == "valid":
            self.valid_idx = 0
        else:
            self.test_idx = 0


    def _loadClasses(self):
        f = open(os.path.join(self.dir, "labels.txt"), "r")
        self.classes = list(map(lambda x:x.strip(), f.readlines()))
