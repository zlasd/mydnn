# MyDNN

This repository contains my implementations of deep neural networks.

I will also use the networks to solve the CIFAR-10 classification problem in terms of checking their sanity.

## Networks

As a new project, it currently has no implementation yet.

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

Use the script  ```dataset_preprocess.py``` to refine directories structures. Then the images will be grouped by classes.

## TODO

1. add config class
2. add data access object
3. add a benchmark model
4. implement Inception
