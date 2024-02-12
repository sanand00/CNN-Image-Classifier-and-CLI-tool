# CNN-Image-Classifier-and-CLI-tool

A Convolutional Neural Network for classifying an image as a cat or a dog. The CNN was built using PyTorch and its architecture was inspired by [VGG16](https://arxiv.org/abs/1409.1556) and [ResNet](https://arxiv.org/abs/1512.03385). The training was conducted using 32x32, cat and dog images from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. Furthermore a simple data augmentation procedure was condudcted, consisting of a random horizontal flip followed by a 4 pixel padding and then a random square crop.

In addition I've created a simply CLI based tool, to easily use the provided pre-trained model, to classify a provided image. The CLI tool can be used by cloning the repository and running 'cli-app.py' 
