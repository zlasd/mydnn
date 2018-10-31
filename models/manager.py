

import tensorflow as tf


class BaseManager(object):
    """ Base class for task manager.
    """
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)



class Manager(BaseManager):
    """ Task manager for training and evaluatiing process.
    """
    def __init__(self, config, dataset, model=None):
        """ config : utils.config.Config
            dataset : utils.dataset.Dataset
            model: function
        """
        super().__init__(config, dataset)
        self.buildModel(model)


    def buildModel(self, model):
        """ set up model output and model loss.
        """
        with self.graph.as_default():
            self.tensor_X = tf.placeholder(tf.float32, shape=( \
                None, self.dataset.IMAGE_HEIGHT, self.dataset.IMAGE_WIDTH, 3))
            self.tensor_y = tf.placeholder(tf.uint8, shape=(None, ))

            y = model(self.tensor_X, self.config)
            self.model_out = y
            
            if self.config.MODE == "training":
                y_ = tf.one_hot(self.tensor_y, self.config.NUM_CLASSES)
                loss = tf.losses.log_loss(y_, y)
                self.loss = loss
            else:
                equals = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
                self.accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))


    def compile(self, optimizer=None):
        if self.config.MODE != "training": 
            self.dataset.loadTestData()
            return
        
        self.dataset.loadTrainData()

        with self.graph.as_default():
            if optimizer is None:
                optimizer = tf.train.RMSPropOptimizer(self.config.LEARNING_RATE, momentum=0.9)
            self.train_step = optimizer.minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())


    def loadWeights(self, path):
        """ load model weights from file.
        """
        pass

    def saveWeigths(self, path):
        """ save model weights to file.
        """
        pass

    def training(self):
        """ training model using model loss.
        """
        config = self.config

        for i in range(config.EPOCH):
            for j in range(config.STEPS_PER_EPOCH):
                train_batch_X, train_batch_y = self.dataset.getBatch(
                                                        self.config.BATCH_SIZE)
                self.sess.run(self.train_step, feed_dict={
                    self.tensor_X : train_batch_X,
                    self.tensor_y: train_batch_y
                })
                
            mean_acc = 0.0
            for j in range(config.VALIDATION_STEP):
                valid_batch_X, valid_batch_y = self.dataset.getBatch(
                                        self.config.BATCH_SIZE, mode="valid")
                valid_acc = self.sess.run(self.accuracy, feed_dict={
                    self.tensor_X : valid_batch_X,
                    self.tensor_y: valid_batch_y
                })
                mean_acc += valid_acc
            print("Epoch {} validation accuracy: {}".format(i, mean_acc))

        tf.train.Saver()


    def predict(self, custom_X=None):
        pass
