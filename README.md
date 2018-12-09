# MyDNN

This repository contains my implementations of deep neural networks.

I will also use the networks to solve the CIFAR-10 classification problem in terms of checking their sanity.

## Requirements

```
numpy
matplotlib (to use imread)
tensorflow
scikit-learn (to use train_test_split)
```


## CIFAR-10

The datasets should be loaded in folder ```data```.

The original directories stucture is as follow:
```
+---data
|   +---labels.txt      # labels in cifar-10
|   +---train
|   |   +--- 0_frog.png
|   |   +--- 1_truck.png
|   |   +--- ...
|   +---test
|   |   +--- 0_cat.png
|   |   +--- 1_ship.png 
|   |   +--- ...
```

You can download it from [fast.ai](https://course.fast.ai/datasets).

You can also use ```tf.data.TFRecordDataset```.

Firstly, prepare dataset as above and run ```conver_to_tfrecord.py``` script. Then, you can use ```CIFAR10TF``` and ```ManagerTF``` which use ```tf.data``` pipeline. The usage of those classes can be founded in ```test.py```.

## Networks

GoogLeNet

![googlenet](tb_example.png)

ResNet and SENet are added. Some test should be done further.

## TODO

1. implement InceptionV3
3. add another datasets
4. add object detection task
