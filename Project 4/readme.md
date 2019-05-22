#  Architectural Basics

Properly designing a neural network is of uttermost importance in deep learning. There is a wide spread belief that deep neural networks are magic boxes which ingest data and come up with intricate features and variables to solve the problem in hand. This is not entirely true. A neural network has many parameters and infinite ways in which they can interact with each other, which means there are infinitely many ways to design a network. This makes a complete grid search over the entire parameter space infeasible. So best approach would be to hand engineer the network in terms of number of layer, type of layers and many more other parameters.

Designing a good model usually involves a lot of trial and error. It is still more of an art than science, and people have their own ways of designing models. The basic thing to keep in mind beforing finalizing on an architecture is to check whether it capable for handling the complexity of the problem in hand.

We will see few steps and tips that we need to consider while designing a convolutional neural network architecture for an 
image classification problem. We are still in an early phase of deep learning, which means the list is not exhaustive.

The following steps are to be followed in the given order.

## 1. Receptive Field :
The first and foremost thing in designing a convolutional neural network is to understand the receptive field required it. **Receptive field** is a portion of sensory space that can elicit neuronal responses when stimulated. More specifically, it is defined as the region in the input space that a particular CNN’s feature is looking at. The final layer of the network should be able to see the entire image to say something about it, hence the global receptive field at the final layer should be as big as the image. Incase, if the objects of interest are always smaller than the image, then the required receptive field needs to be as big as the object in the image. Basically, the network should be able to see the full picture before commenting on it.

Eg : Let's say we are doing a image classification problem on MNIST data with image sizes of 28x28.

## 2. 3x3 Convolutions :

## 3. MaxPooling :

## When do we stop convolutions and go ahead with a larger kernel or some other alternative (

## 4. How many layers:

## 5. Position of MaxPooling :
The distance of MaxPooling from Prediction

## 6. Concept of Transition Layers : 

## 7. Position of Transition Layer :

## 8. Kernels and how do we decide the number of kernels?

## 1x1 Convolutions,

## SoftMax,

## Image Normalization,

## Batch Size, and effects of batch size

## Adam vs SGD

## Learning Rate,

## When to add validation checks

## Number of Epochs and when to increase them,

## How do we know our network is not going well, comparatively, very early

## When do we introduce DropOut, or when do we know we have some overfitting

## DropOut

## 9. Batch Normalization,

## The distance of Batch Normalization from Prediction,

## LR schedule and concept behind it
