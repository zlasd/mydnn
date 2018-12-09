
import os

import tensorflow as tf

from util.dataset import CIFAR10


def parse_exmp(serial_exmp):
    """ dataset parser
    """
    feats = tf.parse_single_example(serial_exmp, features={
        'raw':tf.FixedLenFeature([], tf.string),
        'label':tf.FixedLenFeature([1],tf.int64)})
    image = tf.decode_raw(feats['raw'], tf.float32)
    image = tf.reshape(image, [32, 32, 3])
    label = feats['label']
    return image, label


def get_dataset(fname):
    """ get dataset.
    """
    dataset = tf.data.TFRecordDataset(fname)
    dataset = dataset.map(parse_exmp)
    dataset = dataset.shuffle(10000)
    return dataset



class CIFAR10TF(CIFAR10):
    """ CIFAR10 dataset with tf.data pipeline, using TFRecord
    """
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.train_fn = os.path.join(data_dir, 'train.tfrecord')
        self.valid_fn = os.path.join(data_dir, 'valid.tfrecord')
        self.test_fn = os.path.join(data_dir, 'test.tfrecord')
        self.iterator = None


    def loadData(self):
        """ load train/valid/test tfrecord and generate iterator
        """
        self.train_tf = get_dataset(self.train_fn)
        self.valid_tf = get_dataset(self.valid_fn)
        self.test_tf = get_dataset(self.test_fn)


    def buildIterator(self, bs):
        """ build iterator and set batch size
        """
        self.loadData()
        self.train_tf = self.train_tf.batch(bs)
        self.valid_tf = self.valid_tf.batch(bs)
        self.test_tf = self.test_tf.batch(bs)
        
        if self.iterator is None:
            self.iterator = tf.data.Iterator.from_structure(
                                        self.train_tf.output_types, 
                                        self.train_tf.output_shapes)
        
        self.train_iter_init = self.iterator.make_initializer(self.train_tf)
        self.valid_iter_init = self.iterator.make_initializer(self.valid_tf)
        self.test_iter_init = self.iterator.make_initializer(self.test_tf)


    def getInitializer(self, mode="train"):
        """ return corresponding iterator initializer
        """
        assert mode in ["train", "valid", "test"]

        if mode == "train":
            return self.train_iter_init
        elif mode == "valid":
            return self.valid_iter_init
        
        return self.test_iter_init
    