import tensorflow as tf

from util.dataset import CIFAR10


def parse_exmp(serial_exmp):
        feats = tf.parse_single_example(serial_exmp, features={
        'raw':tf.FixedLenFeature([], tf.string),
        'label':tf.FixedLenFeature([1],tf.int64)})
    image = tf.decode_raw(feats['raw'], tf.float32)
    label = feats['label']
    return image, label

class CIFAR10_TF(CIFAR10):
    """ CIFAR10 dataset with tf.data pipeline, using TFRecord
    """
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.train_tf = os.path.join(data_dir, 'train.tfrecord')
        self.valid_tf = os.path.join(data_dir, 'valid.tfrecord')
        self.test_tf = os.path.join(data_dir, 'test.tfrecord')


    def getBatch(self, bs, mode="train"):
        pass
    