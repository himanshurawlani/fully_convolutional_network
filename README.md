# Implementing a fully convolutional network (FCN) in TensorFlow
The code in this repository is developed and tested on Ubuntu 18.04.3 LTS with Python 3.6.7. Following are the packages required to setup the environment:
* TensorFlow 2.x (tensorflow==2.0.0)
* OpenCV (opencv-python==4.1.2.30)
* Scikit learn (sklearn==0.21.3)
* Numpy (numpy==1.16.2)

Please refer the blogpost here for a detailed explanation of the project. It covers the following topics:
1. Building a fully convolutional network (FCN) in TensorFlow using Keras
2. Downloading and splitting a sample dataset
3. Creating a generator in Keras to load and process a batch of data in memory
4. Training the network with variable batch dimensions
5. Deploying the model using TensorFlow Serving
