
import os

import tensorflow as tf


class BaseManager(object):
    """ Base class for task manager.
    """
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.graph = tf.Graph()



class Manager(BaseManager):
    """ Task manager for training and evaluatiing process.
    """
    def __init__(self, config, dataset, model_fn):
        """ config : utils.config.Config
            dataset : utils.dataset.Dataset
            model: function
        """
        super().__init__(config, dataset)
        self.buildModel(model_fn)


    def buildModel(self, model_fn):
        """ set up model output and model loss.
        """
        with self.graph.as_default():
            self.tensor_X = tf.placeholder(tf.float32, shape=( \
                None, self.dataset.IMAGE_HEIGHT, self.dataset.IMAGE_WIDTH, 3))
            self.tensor_y = tf.placeholder(tf.uint8, shape=(None, ))

            y, _ = model_fn(self.tensor_X, self.config)
            self.model_out = y
            
            if self.config.MODE == "training":
                y_ = tf.one_hot(self.tensor_y, self.config.NUM_CLASSES)
                loss = tf.losses.log_loss(y_, y)
                self.loss = loss

            equals = tf.equal(tf.arg_max(y, 1), tf.argmax(y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))


    def compile(self, optimizer=None):
        """ load data and create seesion.
        """
        assert self.config.MODE in ["training", "inference"]

        self.sess = tf.Session(graph=self.graph)
        
        if self.config.MODE == "inference": 
            self.dataset.loadTestData()
            return
        
        self.dataset.loadTrainData()

        with self.graph.as_default():
            if optimizer is None:
                optimizer = tf.train.RMSPropOptimizer(self.config.LEARNING_RATE, momentum=0.9)
            self.train_step = optimizer.minimize(self.loss)
            tf.summary.scalar("log loss", self.loss)
            tf.summary.histogram("prob", self.model_out)
            self.summary = tf.summary.merge_all()
            self.trainWriter = tf.summary.FileWriter(
                    self.config.TENSORBOARD_DIR+'/train', self.graph)
            self.testWriter = tf.summary.FileWriter(
                    self.config.TENSORBOARD_DIR+'/test', self.graph)

            self.sess.run(tf.global_variables_initializer())


    def loadWeights(self, path):
        """ load model weights from file.
        """
        with self.graph.as_default():
            try:
                name = path.split('/')[-1]
                epoch = int(name.split('-')[-1])
                self.config.PRETRAINED_EPOCH = epoch
            except Exception:
                pass
            tf.train.Saver().restore(self.sess, path)


    def saveWeights(self, epoch):
        """ save model weights to file.
        """
        epoch += self.config.PRETRAINED_EPOCH
        path = '{}/{}-{}'.format(self.config.CHECKPOINT_DIR,
                                         self.config.NAME, epoch)
        path = os.path.join(os.getcwd(), path)

        with self.graph.as_default():
            save_path = tf.train.Saver().save(self.sess, path)
        print("Model saved in: {}".format(save_path))


    def training(self):
        """ training model using model loss.
        """
        config = self.config

        for i in range(config.EPOCH):
            print("Epoch {} training...".format(i+1))
            self.dataset.resetIdx()
            batch = self.dataset.nextBatch(self.config.BATCH_SIZE)
            while batch:
                train_batch_X, train_batch_y = batch
                summary, _ = self.sess.run([self.summary, self.train_step],
                    feed_dict={
                        self.tensor_X : train_batch_X,
                        self.tensor_y: train_batch_y
                })
                batch = self.dataset.nextBatch(self.config.BATCH_SIZE)
            
            self.trainWriter.add_summary(summary, i)
                
            mean_acc = 0.0
            count = 0
            self.dataset.resetIdx(mode="valid")
            batch = self.dataset.nextBatch(self.config.BATCH_SIZE, mode="valid")
            while batch:
                valid_batch_X, valid_batch_y = batch
                summary, valid_acc = self.sess.run([self.summary, self.accuracy],
                    feed_dict={
                        self.tensor_X : valid_batch_X,
                        self.tensor_y: valid_batch_y
                })
                mean_acc += valid_acc
                count += 1
                batch = self.dataset.nextBatch(self.config.BATCH_SIZE, mode="valid")
            mean_acc /= count
            self.testWriter.add_summary(summary, i)
            print("Epoch {} validation accuracy: {}".format(i+1, mean_acc))

            if (i+1) % self.config.SAVE_PER_EPOCH == 0:
                self.saveWeights(i+1)


    def predict(self, custom_X=None):
        pass
