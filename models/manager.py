
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
            model_fn: function
        """
        super().__init__(config, dataset)
        self.buildModel(model_fn)

    def _get_tensor_X_y(self):
        """ return the input tensor of the model.
            Subclass can change how to feed input data
            by overriding this method.
        """
        X = tf.placeholder(tf.float32, shape=( \
                None, self.dataset.IMAGE_HEIGHT, self.dataset.IMAGE_WIDTH, 3))
        # padding x to upsample
        X = tf.pad(X, [(0, 0), (32, 32), (32, 32), (0, 0)])
        y = tf.placeholder(tf.uint8, shape=(None, ))
        return X, y


    def buildModel(self, model_fn):
        """ set up model output and model loss.
        """
        with self.graph.as_default():
            self.tensor_X, self.tensor_y = self._get_tensor_X_y()
            self.config.training = tf.placeholder(tf.bool)

            y, _ = model_fn(self.tensor_X, self.config)
            self.model_out = y
            
            if self.config.MODE == "train":
                y_ = tf.one_hot(self.tensor_y, self.config.NUM_CLASSES)
                loss = tf.losses.softmax_cross_entropy(y_, y)
                self.loss = loss

            equals = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))


    def compile(self):
        """ load data and create seesion.
        """
        assert self.config.MODE in ["train", "inference"]

        self.sess = tf.Session(graph=self.graph)
        
        if self.config.MODE == "inference": 
            self.dataset.loadTestData()
            return
        
        self.dataset.loadTrainData()

        with self.graph.as_default():
            # define training step
            global_step = tf.Variable(0, trainable=False)
            self.variable_lr = tf.train.exponential_decay(self.config.LEARNING_RATE,
                                           global_step=global_step,
                                           decay_steps=5,decay_rate=0.95)
            self.variable_add_step = global_step.assign_add(1)
            self.optimizer = tf.train.MomentumOptimizer(self.variable_lr, momentum=0.9)
            self.train_step = self.optimizer.minimize(self.loss)

            # define tensorboard summaries
            tf.summary.scalar("log loss", self.loss)
            tf.summary.histogram("prob", self.model_out)
            self.summary = tf.summary.merge_all()
            self.trainWriter = tf.summary.FileWriter(
                    self.config.TENSORBOARD_DIR+'/train', self.graph)
            self.testWriter = tf.summary.FileWriter(
                    self.config.TENSORBOARD_DIR+'/test', self.graph)

            # define saver
            self.saver = tf.train.Saver()

            # init
            self.sess.run(tf.global_variables_initializer())


    def loadWeights(self, path):
        """ load model weights from file.
        """
        with self.graph.as_default():
            tf.train.Saver().restore(self.sess, path)


    def saveWeights(self, epoch):
        """ save model weights to file.
        """
        path = '{}/{}'.format(self.config.CHECKPOINT_DIR,
                                         self.config.NAME)
        path = os.path.join(os.getcwd(), path)

        with self.graph.as_default():
            save_path = self.saver.save(self.sess, path)
        print("Model saved in: {}".format(save_path))


    def training(self):
        """ training model using model loss.
            Subclass can overide this method if neccessary.
        """
        config = self.config

        for i in range(config.EPOCH):
            print("Epoch {} training...".format(i+1))
            self.sess.run([self.variable_add_step, self.variable_lr])

            self.dataset.resetIdx()
            batch = self.dataset.nextBatch(self.config.BATCH_SIZE)
            while batch:
                train_batch_X, train_batch_y = batch
                summary, _ = self.sess.run([self.summary, self.train_step],
                    feed_dict={
                        self.tensor_X : train_batch_X,
                        self.tensor_y: train_batch_y,
                        self.config.training: True
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
                        self.tensor_y: valid_batch_y,
                        self.config.training: False
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
        """ predict custom samples using this model.
        """
        pass



class ManagerTF(Manager):
    """ model manager with tf.data pipeline.
    """
    def __init__(self, config, dataset, model_fn):
        """ config : utils.config.Config
            dataset : utils.dataset.Dataset
            model_fn: function
        """
        super().__init__(config, dataset, model_fn)
    

    def _get_tensor_X_y(self):
        self.dataset.buildIterator(bs=self.config.BATCH_SIZE)
        X, y = self.dataset.iterator.get_next()
        y = tf.squeeze(y)
        return X, y


    def training(self):
        """ training model using model loss.
            Using tf.data API
        """
        config = self.config

        for i in range(config.EPOCH):
            print("Epoch {} training...".format(i+1))
            self.sess.run([self.variable_add_step, self.variable_lr])

            self.sess.run(self.dataset.getInitializer())
            n_batch = 0
            mean_loss = 0
            try:
                while True:
                    summary, loss, _ = self.sess.run([self.summary, self.loss, self.train_step],
                                        feed_dict={self.config.training: True})
                    n_batch += 1
                    mean_loss += loss
            except tf.errors.OutOfRangeError:
                mean_loss /= n_batch
            
            print('Epoch {} average loss: {}'.format(i+1, mean_loss))
            self.trainWriter.add_summary(summary, i)
            

            self.sess.run(self.dataset.getInitializer(mode="valid"))
            mean_acc = 0.0
            n_batch = 0

            try:
                while True:
                    summary, valid_acc = self.sess.run([self.summary, self.accuracy],
                                        feed_dict={self.config.training: False})
                    mean_acc += valid_acc
                    n_batch += 1
            except tf.errors.OutOfRangeError:
                mean_acc /= n_batch
            
            print("Epoch {} validation accuracy: {}".format(i+1, mean_acc))
            self.testWriter.add_summary(summary, i)

            if (i+1) % self.config.SAVE_PER_EPOCH == 0:
                self.saveWeights(i+1)
        
        self.evaluate()


    def evaluate(self):
        """ evaluate model using test data.
        """
        self.sess.run(self.dataset.getInitializer(mode="test"))
        mean_acc = 0.0
        n_batch = 0

        try:
            while True:
                test_acc = self.sess.run(self.accuracy,
                                feed_dict={self.config.training: False})
                mean_acc += test_acc
                n_batch += 1
        except tf.errors.OutOfRangeError:
            mean_acc /= n_batch
        
        print("\n=================")
        print("Final report")
        print("training epoch: {}".format(self.config.EPOCH))
        print("training batch size: {}".format(self.config.BATCH_SIZE))
        print("test accuracy: {}".format(mean_acc))
        print("=================")


