#  Architectural Basics

Properly designing a neural network is of uttermost importance in deep learning. There is a widespread belief that deep neural networks are magic boxes which ingest data and come up with intricate features and variables to solve the problem in hand. This is not entirely true. A neural network has many parameters and infinite ways in which they can interact with each other, which means there are infinitely many ways to design a network. This makes a complete grid search over the entire parameter space infeasible. So the best approach would be to hand engineer the network in terms of the number of layer, type of layers and many more other parameters.

Designing a good model usually involves a lot of trial and error. It is still more of an art than science, and people have their own ways of designing models. The basic thing to keep in mind before finalizing on architecture is to check whether it is capable of handling the complexity of the problem in hand.

We will see a few steps and tips that we need to consider while designing a convolutional neural network architecture for an image classification problem. We are still in an early phase of deep learning, which means the list is not exhaustive.

The following steps are to be followed in the given order.

## 1. Receptive Field :
The first and foremost thing in designing a convolutional neural network is to understand the receptive field required it. **Receptive field** is a portion of sensory space that can elicit neuronal responses when stimulated. More specifically, it is defined as the region in the input space that a particular CNN’s feature is looking at. The final layer of the network should be able to see the entire image to say something about it, hence the global receptive field at the final layer should be as big as the image. In case, if the objects of interest are always smaller than the image, then the required receptive field needs to be as big as the object in the image. Basically, the network should be able to see the full picture before commenting on it.

Eg: Let's say we are doing an image classification problem on MNIST data with image sizes of 28x28. So ideally, we need to have a receptive field of 28x28. But if you closely examine the images, we can see that the numbers in the images are not of full size. The edges of the images have blank spaces which is irrelevant for correctly identifying the label of the image. By further analysis, we can see that the receptive field required for this problem would around 23x23.

## 2. Convolution Size :
Once we have finalized on the receptive field requirement, then the next task is to decide on the convolutional filter size. **Filters**  or **kernels** are used in image processing for blurring, sharpening, embossing, edge detection and more. In the initial layers of CNN, filters, when convolved across an image, captures low-level features like vertical, horizontal and diagonal edges. Filter in the subsequent layers will be able to detect abstract and complex features. Each filter is a matrix of numbers corresponding to a feature that the filter is looking for. They can be any sizes like **1x1, 3x3, 5x5, 7x7, 11x11** etc. Usually, we don't use even sized filter since they don't have an axis of symmetry and hence less useful in extracting features. We mostly use 3x3 filters when designing the network due to the following advantages of 3x3 over others.

### Why should we mostly use 3x3 Kernels?

3x3 Convolution** is a type of **filter/kernel** used in **deep CNNs**. It is basically a matrix of size **3x3xinput_channel** convolved over the previous input layers. 

![alt+text](https://github.com/rinazbelhaj/EVA/blob/master/Images/3x3%20Convolution.png?raw=true "3x3 Convolution")

1.  **3x3 kernels have least number of parameters [9 Parameters]**</br>
    3x3 kernels are the smallest possible kernel that can detect patterns in an image. This has 9 parameters which need to be optmized       while training. Any higher order filter can be represented as a multiple of 3x3 filters thereby reducing the parameters. Fewer           parameters result in the network to  be trained faster.</br>
    
    Eg : 5x5 filter can be represented as a series of two 3x3 filters. </br>
    5x5 filter : 25 parameters </br>
    3x3 filter x 2 : 9 x 2 = 18 parameters </br>
2.  **3x3 kernels are accelerated on GPUs and TPUs** </br>
    Since most of the researchers and companies using CNNs use 3x3 filter, hardware manufacturers have optimized their GPUs and TPUs to     perform faster operations on these filters. This resulted in more and more people using 3x3 filter more.

## 3. MaxPooling :
**Max pooling** is a sample-based discretization process. The objective is to down-sample an input representation, reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned. Max pooling is done by applying a max filter to (usually) non-overlapping subregions of the initial representation.

It is common to periodically insert a max Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. The Pooling Layer operates independently on every depth slice of the input and resizes it spatially, using the MAX operation. The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. 

![alt+text](https://github.com/rinazbelhaj/EVA/blob/master/Images/image.png?raw=true "3x3 Convolution")

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
