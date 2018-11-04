


class Config(object):
    """ Inspired by https://github.com/matterport/Mask_RCNN
    Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    NAME = "model"
    CHECKPOINT_DIR = "checkpoints"
    TENSORBOARD_DIR = "tensorboard"

    BATCH_SIZE = 128
    EPOCH = 10
    PRETRAINED_EPOCH = 0
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEP = 50
    SAVE_PER_EPOCH = 5

    NUM_CLASSES = 10
    LEARNING_RATE = 0.01
    PADDING = "SAME"

    MODE = "training"

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

