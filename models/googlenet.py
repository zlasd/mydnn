
import numpy as np
import tensorflow as tf


from tensorflow.initializers import random_normal

from util.config import Config

class GoogleNetConfig(Config):
    NAME = "googlenet"
    LEARNING_RATE = 0.01
    EPOCH = 300
    SAVE_PER_EPOCH = 10

def googlenet(X, config):
    pass